"""
Parallel execution capabilities for the travel planner system.

This module implements parallel execution of agent tasks using LangGraph's
parallel branch capabilities, allowing multiple agents to work simultaneously
on different aspects of the travel planning process.
"""

import asyncio
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from travel_planner.agents.base import BaseAgent
from travel_planner.orchestration.states.workflow_stages import (
    PARALLEL_SEARCH_COMPLETED,
)
from travel_planner.utils.logging import get_logger

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from travel_planner.orchestration.states.planning_state import TravelPlanningState
else:
    # Forward references for type hints
    TravelPlanningState = Any

# Type for state update functions
T = TypeVar("T")
UpdateFunction = Callable[[T], T]

logger = get_logger(__name__)


class ParallelTask(Enum):
    """Enum representing different parallel tasks in the travel planning process."""

    FLIGHT_SEARCH = "flight_search"
    ACCOMMODATION = "accommodation"
    TRANSPORTATION = "transportation"
    ACTIVITIES = "activities"
    BUDGET = "budget"


class ParallelResult(BaseModel):
    """Model for storing results from parallel task execution."""

    task_type: ParallelTask
    result: dict[str, Any]
    error: str | None = None
    completed: bool = False


async def execute_in_parallel(
    tasks: list[tuple[BaseAgent, dict[str, Any]]], state: TravelPlanningState
) -> dict[str, Any]:
    """
    Execute multiple agent tasks in parallel using asyncio.

    Args:
        tasks: List of (agent, parameters) tuples to execute
        state: Current travel planning state

    Returns:
        Combined results from all parallel tasks
    """

    async def execute_task(agent: BaseAgent, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a single agent task with retry logic."""
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        # Define which exceptions should trigger retries
        retryable_exceptions = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        )

        # Define retry decorator
        @retry(
            stop=stop_after_attempt(3),  # Try up to 3 times
            wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff
            retry=retry_if_exception_type(retryable_exceptions),
            reraise=True,  # Re-raise the last exception if all retries fail
        )
        async def execute_with_retry():
            try:
                return await agent.process(**params, context=state)
            except retryable_exceptions as e:
                logger.warning(f"Retryable error in {agent.name}: {e!s}. Will retry.")
                raise  # Re-raise to trigger retry
            except Exception as e:
                logger.error(f"Non-retryable error in {agent.name}: {e!s}")
                raise  # Re-raise to be caught by the outer try-except

        try:
            # Try to execute with retry logic
            result = await execute_with_retry()
            return {agent.name: {"result": result, "error": None, "retries": 0}}
        except Exception as e:
            # Catch and log all exceptions after retries are exhausted
            logger.error(f"Error in parallel task {agent.name} after retries: {e!s}")
            return {agent.name: {"result": None, "error": str(e), "retries": 3}}

    # Create a list of coroutines to execute
    coroutines = [execute_task(agent, params) for agent, params in tasks]

    # Execute all coroutines in parallel with timeout protection
    try:
        results = await asyncio.gather(*coroutines)

        # Combine results into a single dictionary
        combined_results = {}
        for result in results:
            combined_results.update(result)

        return combined_results
    except TimeoutError:
        logger.error("Parallel execution timed out")
        return {"error": "Execution timeout exceeded"}


async def parallel_search_tasks(state: TravelPlanningState) -> TravelPlanningState:
    """
    Execute search-related tasks in parallel (flights, accommodation, activities).

    This function sets up parallel execution of multiple agent tasks using asyncio
    for concurrent processing, which significantly reduces the overall processing time
    for travel planning.

    Args:
        state: Current travel planning state

    Returns:
        Updated state with search results
    """
    # Import specialized agents
    from travel_planner.agents.accommodation import AccommodationAgent
    from travel_planner.agents.activity_planning import ActivityPlanningAgent
    from travel_planner.agents.flight_search import FlightSearchAgent
    from travel_planner.agents.transportation import TransportationAgent

    logger.info("Setting up parallel search tasks")

    # Create a deep copy of the state to avoid mutations during parallel processing
    # This helps prevent race conditions and unexpected state changes
    working_state = state.model_copy(deep=True)

    try:
        # Create agent instances with lazy loading for better resource utilization
        agents = {
            "flight": FlightSearchAgent(),
            "accommodation": AccommodationAgent(),
            "transportation": TransportationAgent(),
            "activity": ActivityPlanningAgent(),
        }

        # Set up task list with agents and their parameters
        # Include specific timeouts for each agent type
        tasks = [
            (agents["flight"], {"query": working_state.query, "timeout": 60}),
            (agents["accommodation"], {"query": working_state.query, "timeout": 60}),
            (agents["transportation"], {"query": working_state.query, "timeout": 45}),
            (
                agents["activity"],
                {
                    "query": working_state.query,
                    "preferences": working_state.preferences,
                    "timeout": 60,
                },
            ),
        ]

        logger.info(f"Executing {len(tasks)} tasks in parallel")

        # Execute all tasks in parallel with overall timeout
        async with asyncio.timeout(180):  # 3 minute overall timeout
            results = await execute_in_parallel(tasks, working_state)

        # Check for overall errors in parallel execution
        if "error" in results and not any(k != "error" for k in results):
            logger.error(f"All parallel tasks failed: {results['error']}")
            working_state.error = f"Parallel execution error: {results['error']}"
            if not working_state.plan or not working_state.plan.alerts:
                if not working_state.plan:
                    from travel_planner.data.models import TravelPlan

                    working_state.plan = TravelPlan()
                working_state.plan.alerts = []
            working_state.plan.alerts.append(
                f"Error in parallel search: {results['error']}"
            )
            working_state.current_stage = "error"
            return working_state

        # Merge results into the state
        updated_state = merge_parallel_results(working_state, results)

        # Update the current stage
        updated_state.current_stage = PARALLEL_SEARCH_COMPLETED

        logger.info("Parallel search tasks completed successfully")
        return updated_state

    except TimeoutError:
        logger.error("Parallel search tasks timed out after 3 minutes")
        working_state.error = "Parallel search timed out"
        if not working_state.plan or not working_state.plan.alerts:
            if not working_state.plan:
                from travel_planner.data.models import TravelPlan

                working_state.plan = TravelPlan()
            working_state.plan.alerts = []
        working_state.plan.alerts.append(
            "Search operations timed out. Some results may be incomplete."
        )
        working_state.current_stage = "error"
        return working_state

    except Exception as e:
        logger.error(f"Unexpected error in parallel search: {e!s}")
        working_state.error = f"Unexpected error: {e!s}"
        if not working_state.plan or not working_state.plan.alerts:
            if not working_state.plan:
                from travel_planner.data.models import TravelPlan

                working_state.plan = TravelPlan()
            working_state.plan.alerts = []
        working_state.plan.alerts.append(f"Unexpected error in search: {e!s}")
        working_state.current_stage = "error"
        return working_state


def merge_parallel_results(
    state: TravelPlanningState, results: dict[str, Any]
) -> TravelPlanningState:
    """
    Merge results from parallel execution into the state.

    This function takes the results from parallel agent executions and merges them
    into a consolidated state, ensuring that all data is properly integrated.

    Args:
        state: Current travel planning state
        results: Results from parallel execution

    Returns:
        Updated state with merged results
    """
    # Create a copy of the state to update
    updated_state = state.model_copy(deep=True)

    # Initialize the plan if not yet created
    updated_state = _ensure_plan_initialized(updated_state)

    # Process results from each agent
    updated_state = _process_flight_results(updated_state, results)
    updated_state = _process_accommodation_results(updated_state, results)
    updated_state = _process_transportation_results(updated_state, results)
    updated_state = _process_activity_results(updated_state, results)

    # Handle errors
    updated_state = _process_parallel_errors(updated_state, results)

    return updated_state


def _ensure_plan_initialized(state: TravelPlanningState) -> TravelPlanningState:
    """Ensure the travel plan is initialized in the state."""
    if state.plan is None:
        from travel_planner.data.models import TravelPlan

        state.plan = TravelPlan()
    return state


def _process_flight_results(
    state: TravelPlanningState, results: dict[str, Any]
) -> TravelPlanningState:
    """Process flight search results from parallel execution."""
    if "FlightSearchAgent" not in results or not results["FlightSearchAgent"]["result"]:
        return state

    flight_data = results["FlightSearchAgent"]["result"]

    if "flights" in flight_data:
        state.plan.flights = flight_data["flights"]
    elif "flight_options" in flight_data:
        state.plan.flights = flight_data["flight_options"]

    return state


def _process_accommodation_results(
    state: TravelPlanningState, results: dict[str, Any]
) -> TravelPlanningState:
    """Process accommodation results from parallel execution."""
    if (
        "AccommodationAgent" not in results
        or not results["AccommodationAgent"]["result"]
    ):
        return state

    accom_data = results["AccommodationAgent"]["result"]

    if "accommodations" in accom_data:
        state.plan.accommodation = accom_data["accommodations"]
    elif "accommodation_options" in accom_data:
        state.plan.accommodation = accom_data["accommodation_options"]

    return state


def _process_transportation_results(
    state: TravelPlanningState, results: dict[str, Any]
) -> TravelPlanningState:
    """Process transportation results from parallel execution."""
    if (
        "TransportationAgent" not in results
        or not results["TransportationAgent"]["result"]
    ):
        return state

    transport_data = results["TransportationAgent"]["result"]

    if "transportation" in transport_data:
        state.plan.transportation = transport_data["transportation"]
    elif "transportation_options" in transport_data:
        state.plan.transportation = transport_data["transportation_options"]

    return state


def _process_activity_results(
    state: TravelPlanningState, results: dict[str, Any]
) -> TravelPlanningState:
    """Process activity planning results from parallel execution."""
    if (
        "ActivityPlanningAgent" not in results
        or not results["ActivityPlanningAgent"]["result"]
    ):
        return state

    activity_data = results["ActivityPlanningAgent"]["result"]

    if "activities" in activity_data:
        state.plan.activities = activity_data["activities"]
    elif "daily_itineraries" in activity_data:
        state.plan.activities = activity_data["daily_itineraries"]

    return state


def _process_parallel_errors(
    state: TravelPlanningState, results: dict[str, Any]
) -> TravelPlanningState:
    """Process errors from parallel execution and add them to the state."""
    errors = []

    for agent_name, result in results.items():
        if result.get("error"):
            errors.append(f"{agent_name}: {result['error']}")

    if errors:
        if not state.plan.alerts:
            state.plan.alerts = []
        state.plan.alerts.extend(errors)

    return state


def combine_parallel_branch_results(
    state: TravelPlanningState, branch_results: dict[str, ParallelResult]
) -> TravelPlanningState:
    """
    Combine results from a LangGraph parallel branch execution.

    Args:
        state: Current travel planning state
        branch_results: Results from parallel branch execution

    Returns:
        Updated state with combined results
    """
    # Create a copy of the state to update
    updated_state = state.model_copy(deep=True)

    # Initialize the plan if not yet created
    updated_state = _ensure_plan_initialized(updated_state)

    # Ensure there are results to process
    if not _validate_branch_results(branch_results):
        logger.warning("No results from parallel branch execution")
        return updated_state

    # Extract and organize results by task type
    results_by_task = _organize_branch_results(branch_results)

    # Process results from each task type
    updated_state = _process_branch_flight_results(updated_state, results_by_task)
    updated_state = _process_branch_accommodation_results(
        updated_state, results_by_task
    )
    updated_state = _process_branch_transportation_results(
        updated_state, results_by_task
    )
    updated_state = _process_branch_activity_results(updated_state, results_by_task)
    updated_state = _process_branch_budget_results(updated_state, results_by_task)

    # Process errors
    updated_state = _process_branch_errors(updated_state, results_by_task)

    # Update workflow stage
    updated_state = _update_workflow_stage(updated_state)

    return updated_state


def _validate_branch_results(branch_results: dict[str, ParallelResult]) -> bool:
    """Validate that the branch results contain actual result data."""
    # Improved validation that checks if there's any result in the branch results
    if not branch_results:
        return False

    # Check for the 'result' key at the top level
    if branch_results.get("result"):
        return True

    # Check if any of the values in branch_results are ParallelResult objects
    # that have completed successfully
    for key, value in branch_results.items():
        if isinstance(value, ParallelResult) and value.completed and not value.error:
            return True

    return False


def _organize_branch_results(
    branch_results: dict[str, ParallelResult],
) -> dict[ParallelTask, ParallelResult]:
    """Organize parallel branch results by task type."""
    results_by_task = {}
    for task_result in branch_results.values():
        if isinstance(task_result, ParallelResult):
            task_type = task_result.task_type
            results_by_task[task_type] = task_result
    return results_by_task


def _process_branch_flight_results(
    state: TravelPlanningState, results_by_task: dict[ParallelTask, ParallelResult]
) -> TravelPlanningState:
    """Process flight search results from branch execution."""
    if ParallelTask.FLIGHT_SEARCH not in results_by_task:
        return state

    flight_result = results_by_task[ParallelTask.FLIGHT_SEARCH]
    if flight_result.completed and not flight_result.error:
        state.plan.flights = flight_result.result.get("flight_options", [])

    return state


def _process_branch_accommodation_results(
    state: TravelPlanningState, results_by_task: dict[ParallelTask, ParallelResult]
) -> TravelPlanningState:
    """Process accommodation results from branch execution."""
    if ParallelTask.ACCOMMODATION not in results_by_task:
        return state

    accom_result = results_by_task[ParallelTask.ACCOMMODATION]
    if accom_result.completed and not accom_result.error:
        state.plan.accommodation = accom_result.result.get("accommodations", [])

    return state


def _process_branch_transportation_results(
    state: TravelPlanningState, results_by_task: dict[ParallelTask, ParallelResult]
) -> TravelPlanningState:
    """Process transportation results from branch execution."""
    if ParallelTask.TRANSPORTATION not in results_by_task:
        return state

    transport_result = results_by_task[ParallelTask.TRANSPORTATION]
    if transport_result.completed and not transport_result.error:
        state.plan.transportation = transport_result.result.get(
            "transportation_options", {}
        )

    return state


def _process_branch_activity_results(
    state: TravelPlanningState, results_by_task: dict[ParallelTask, ParallelResult]
) -> TravelPlanningState:
    """Process activity results from branch execution."""
    if ParallelTask.ACTIVITIES not in results_by_task:
        return state

    activity_result = results_by_task[ParallelTask.ACTIVITIES]
    if activity_result.completed and not activity_result.error:
        state.plan.activities = activity_result.result.get("daily_itineraries", {})

    return state


def _process_branch_budget_results(
    state: TravelPlanningState, results_by_task: dict[ParallelTask, ParallelResult]
) -> TravelPlanningState:
    """Process budget results from branch execution."""
    if ParallelTask.BUDGET not in results_by_task:
        return state

    budget_result = results_by_task[ParallelTask.BUDGET]
    if budget_result.completed and not budget_result.error:
        state.plan.budget = budget_result.result.get("report", {})

    return state


def _process_branch_errors(
    state: TravelPlanningState, results_by_task: dict[ParallelTask, ParallelResult]
) -> TravelPlanningState:
    """Process errors from branch execution and add them to the state."""
    errors = []

    for task_type, task_result in results_by_task.items():
        if task_result.error:
            errors.append(f"{task_type.value}: {task_result.error}")

    if errors:
        if not state.plan.alerts:
            state.plan.alerts = []
        state.plan.alerts.extend(errors)

    return state


def _update_workflow_stage(state: TravelPlanningState) -> TravelPlanningState:
    """Update the workflow stage in the state."""
    state.current_stage = PARALLEL_SEARCH_COMPLETED
    return state
