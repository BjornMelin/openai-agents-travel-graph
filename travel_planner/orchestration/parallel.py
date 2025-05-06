"""
Parallel execution capabilities for the travel planner system.

This module implements parallel execution of agent tasks using LangGraph's
parallel branch capabilities, allowing multiple agents to work simultaneously
on different aspects of the travel planning process.
"""

import asyncio
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from langgraph.graph.branches.parallel import ParallelBranch
from pydantic import BaseModel

from travel_planner.agents.base import BaseAgent
from travel_planner.orchestration.state_graph import TravelPlanningState
from travel_planner.utils.logging import get_logger

# Type for state update functions
T = TypeVar('T')
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


def create_parallel_branch(state: TravelPlanningState) -> ParallelBranch:
    """
    Create a parallel branch for executing multiple agents simultaneously.
    
    This function creates a LangGraph ParallelBranch that enables concurrent execution
    of multiple agents, allowing faster processing of independent tasks.
    
    Args:
        state: Current travel planning state
        
    Returns:
        ParallelBranch instance with agents to execute in parallel
    """
    # Create a parallel branch for travel planning tasks
    branch = ParallelBranch("parallel_planning")
    
    # Add nodes for different aspects of travel planning that can run in parallel
    branch.add_node(ParallelTask.FLIGHT_SEARCH.value, flight_search_task)
    branch.add_node(ParallelTask.ACCOMMODATION.value, accommodation_task)
    branch.add_node(ParallelTask.TRANSPORTATION.value, transportation_task)
    branch.add_node(ParallelTask.ACTIVITIES.value, activities_task)
    
    return branch


def flight_search_task(state: TravelPlanningState) -> dict[str, ParallelResult]:
    """
    Execute flight search task in parallel branch.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Dictionary with task results
    """
    from travel_planner.agents.flight_search import FlightSearchAgent
    
    try:
        agent = FlightSearchAgent()
        result = agent.invoke(state)
        
        return {
            "result": ParallelResult(
                task_type=ParallelTask.FLIGHT_SEARCH,
                result=result,
                completed=True
            )
        }
    except Exception as e:
        logger.error(f"Error in flight search task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.FLIGHT_SEARCH,
                result={},
                error=str(e),
                completed=False
            )
        }


def accommodation_task(state: TravelPlanningState) -> dict[str, ParallelResult]:
    """
    Execute accommodation search task in parallel branch.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Dictionary with task results
    """
    from travel_planner.agents.accommodation import AccommodationAgent
    
    try:
        agent = AccommodationAgent()
        result = agent.invoke(state)
        
        return {
            "result": ParallelResult(
                task_type=ParallelTask.ACCOMMODATION,
                result=result,
                completed=True
            )
        }
    except Exception as e:
        logger.error(f"Error in accommodation task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.ACCOMMODATION,
                result={},
                error=str(e),
                completed=False
            )
        }


def transportation_task(state: TravelPlanningState) -> dict[str, ParallelResult]:
    """
    Execute transportation planning task in parallel branch.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Dictionary with task results
    """
    from travel_planner.agents.transportation import TransportationAgent
    
    try:
        agent = TransportationAgent()
        result = agent.invoke(state)
        
        return {
            "result": ParallelResult(
                task_type=ParallelTask.TRANSPORTATION,
                result=result,
                completed=True
            )
        }
    except Exception as e:
        logger.error(f"Error in transportation task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.TRANSPORTATION,
                result={},
                error=str(e),
                completed=False
            )
        }


def activities_task(state: TravelPlanningState) -> dict[str, ParallelResult]:
    """
    Execute activity planning task in parallel branch.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Dictionary with task results
    """
    from travel_planner.agents.activity_planning import ActivityPlanningAgent
    
    try:
        agent = ActivityPlanningAgent()
        result = agent.invoke(state)
        
        return {
            "result": ParallelResult(
                task_type=ParallelTask.ACTIVITIES,
                result=result,
                completed=True
            )
        }
    except Exception as e:
        logger.error(f"Error in activities task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.ACTIVITIES,
                result={},
                error=str(e),
                completed=False
            )
        }


def budget_task(state: TravelPlanningState) -> dict[str, ParallelResult]:
    """
    Execute budget management task (this usually runs after other tasks are complete).
    
    Args:
        state: Current travel planning state
        
    Returns:
        Dictionary with task results
    """
    from travel_planner.agents.budget_management import BudgetManagementAgent
    
    try:
        agent = BudgetManagementAgent()
        result = agent.invoke(state)
        
        return {
            "result": ParallelResult(
                task_type=ParallelTask.BUDGET,
                result=result,
                completed=True
            )
        }
    except Exception as e:
        logger.error(f"Error in budget task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.BUDGET,
                result={},
                error=str(e),
                completed=False
            )
        }


async def execute_in_parallel(
    tasks: list[tuple[BaseAgent, dict[str, Any]]],
    state: TravelPlanningState
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
        try:
            result = await agent.process(**params, context=state)
            return {agent.name: {"result": result, "error": None}}
        except Exception as e:
            logger.error(f"Error in parallel task {agent.name}: {e!s}")
            return {agent.name: {"result": None, "error": str(e)}}
    
    # Create a list of coroutines to execute
    coroutines = [execute_task(agent, params) for agent, params in tasks]
    
    # Execute all coroutines in parallel
    results = await asyncio.gather(*coroutines)
    
    # Combine results into a single dictionary
    combined_results = {}
    for result in results:
        combined_results.update(result)
    
    return combined_results


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
    
    # Create agent instances
    flight_agent = FlightSearchAgent()
    accommodation_agent = AccommodationAgent()
    transportation_agent = TransportationAgent()
    activity_agent = ActivityPlanningAgent()
    
    # Set up task list with agents and their parameters
    tasks = [
        (flight_agent, {"query": state.query}),
        (accommodation_agent, {"query": state.query}),
        (transportation_agent, {"query": state.query}),
        (activity_agent, {"query": state.query, "preferences": state.preferences})
    ]
    
    logger.info(f"Executing {len(tasks)} tasks in parallel")
    
    # Execute all tasks in parallel
    results = await execute_in_parallel(tasks, state)
    
    # Merge results into the state
    updated_state = merge_parallel_results(state, results)
    
    # Update the current stage
    updated_state.current_stage = "parallel_search_completed"
    
    logger.info("Parallel search tasks completed")
    return updated_state


def merge_parallel_results(state: TravelPlanningState, results: dict[str, Any]) -> TravelPlanningState:
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
    if updated_state.plan is None:
        from travel_planner.data.models import TravelPlan
        updated_state.plan = TravelPlan()
    
    # Process flight search results
    if "FlightSearchAgent" in results and results["FlightSearchAgent"]["result"]:
        flight_data = results["FlightSearchAgent"]["result"]
        if "flights" in flight_data:
            updated_state.plan.flights = flight_data["flights"]
        if "flight_options" in flight_data:
            updated_state.plan.flights = flight_data["flight_options"]
    
    # Process accommodation results
    if "AccommodationAgent" in results and results["AccommodationAgent"]["result"]:
        accom_data = results["AccommodationAgent"]["result"]
        if "accommodations" in accom_data:
            updated_state.plan.accommodation = accom_data["accommodations"]
        if "accommodation_options" in accom_data:
            updated_state.plan.accommodation = accom_data["accommodation_options"]
    
    # Process transportation results
    if "TransportationAgent" in results and results["TransportationAgent"]["result"]:
        transport_data = results["TransportationAgent"]["result"]
        if "transportation" in transport_data:
            updated_state.plan.transportation = transport_data["transportation"]
        if "transportation_options" in transport_data:
            updated_state.plan.transportation = transport_data["transportation_options"]
    
    # Process activity results
    if "ActivityPlanningAgent" in results and results["ActivityPlanningAgent"]["result"]:
        activity_data = results["ActivityPlanningAgent"]["result"]
        if "activities" in activity_data:
            updated_state.plan.activities = activity_data["activities"]
        if "daily_itineraries" in activity_data:
            updated_state.plan.activities = activity_data["daily_itineraries"]
    
    # Check for any errors and add them to the state
    errors = []
    for agent_name, result in results.items():
        if result.get("error"):
            errors.append(f"{agent_name}: {result['error']}")
    
    if errors:
        if not updated_state.plan.alerts:
            updated_state.plan.alerts = []
        updated_state.plan.alerts.extend(errors)
    
    return updated_state


def combine_parallel_branch_results(state: TravelPlanningState, branch_results: dict[str, ParallelResult]) -> TravelPlanningState:
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
    if updated_state.plan is None:
        from travel_planner.data.models import TravelPlan
        updated_state.plan = TravelPlan()
    
    # Process results from each task in the branch
    result = branch_results.get("result")
    if not result:
        logger.warning("No results from parallel branch execution")
        return updated_state
    
    # Organize results by task type
    results_by_task = {}
    for task_result in branch_results.values():
        if isinstance(task_result, ParallelResult):
            task_type = task_result.task_type
            results_by_task[task_type] = task_result
    
    # Update the state with results from each task
    # Flight search results
    if ParallelTask.FLIGHT_SEARCH in results_by_task:
        flight_result = results_by_task[ParallelTask.FLIGHT_SEARCH]
        if flight_result.completed and not flight_result.error:
            updated_state.plan.flights = flight_result.result.get("flight_options", [])
    
    # Accommodation results
    if ParallelTask.ACCOMMODATION in results_by_task:
        accom_result = results_by_task[ParallelTask.ACCOMMODATION]
        if accom_result.completed and not accom_result.error:
            updated_state.plan.accommodation = accom_result.result.get("accommodations", [])
    
    # Transportation results
    if ParallelTask.TRANSPORTATION in results_by_task:
        transport_result = results_by_task[ParallelTask.TRANSPORTATION]
        if transport_result.completed and not transport_result.error:
            updated_state.plan.transportation = transport_result.result.get("transportation_options", {})
    
    # Activities results
    if ParallelTask.ACTIVITIES in results_by_task:
        activity_result = results_by_task[ParallelTask.ACTIVITIES]
        if activity_result.completed and not activity_result.error:
            updated_state.plan.activities = activity_result.result.get("daily_itineraries", {})
    
    # Budget results
    if ParallelTask.BUDGET in results_by_task:
        budget_result = results_by_task[ParallelTask.BUDGET]
        if budget_result.completed and not budget_result.error:
            updated_state.plan.budget = budget_result.result.get("report", {})
    
    # Collect any errors and add them to the state
    errors = []
    for task_type, task_result in results_by_task.items():
        if task_result.error:
            errors.append(f"{task_type.value}: {task_result.error}")
    
    if errors:
        if not updated_state.plan.alerts:
            updated_state.plan.alerts = []
        updated_state.plan.alerts.extend(errors)
    
    # Update the current stage
    updated_state.current_stage = "parallel_tasks_completed"
    
    return updated_state