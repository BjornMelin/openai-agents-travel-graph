"""
Flight search node implementation for the travel planning workflow.

This module defines functions for searching flights using the flight
search agent, both for individual execution and as part of parallel processing.
"""

from travel_planner.agents.flight_search import FlightSearchAgent
from travel_planner.orchestration.nodes.base_node import (
    create_node_function,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


# Using the factory function for a simpler implementation
flight_search = create_node_function(
    agent_class=FlightSearchAgent,
    task_name="flight_search",
    complete_stage=WorkflowStage.FLIGHTS_SEARCHED,
    result_field="flight_options",
    plan_field="flights",
    message_template="Found {count} flight options",
)


def flight_search_task(state: TravelPlanningState) -> dict[str, any]:
    """
    Execute flight search task in parallel branch.

    Args:
        state: Current travel planning state

    Returns:
        Dictionary with task results
    """
    from travel_planner.orchestration.parallel import ParallelResult, ParallelTask

    try:
        agent = FlightSearchAgent()
        result = agent.invoke(state)

        return {
            "result": ParallelResult(
                task_type=ParallelTask.FLIGHT_SEARCH, result=result, completed=True
            )
        }
    except Exception as e:
        logger.error(f"Error in flight search task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.FLIGHT_SEARCH,
                result={},
                error=str(e),
                completed=False,
            )
        }
