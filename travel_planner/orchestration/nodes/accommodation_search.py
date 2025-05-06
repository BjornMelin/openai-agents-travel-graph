"""
Accommodation search node implementation for the travel planning workflow.

This module defines functions for searching accommodations using the accommodation
agent, both for individual execution and as part of parallel processing.
"""

from travel_planner.agents.accommodation import AccommodationAgent
from travel_planner.orchestration.nodes.base_node import (
    create_node_function,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


# Using the factory function for a simpler implementation
accommodation_search = create_node_function(
    agent_class=AccommodationAgent,
    task_name="accommodation_search",
    complete_stage=WorkflowStage.ACCOMMODATION_SEARCHED,
    result_field="accommodations",
    plan_field="accommodation",
    message_template="Found {count} accommodation options",
)


def accommodation_task(state: TravelPlanningState) -> dict[str, any]:
    """
    Execute accommodation search task in parallel branch.

    Args:
        state: Current travel planning state

    Returns:
        Dictionary with task results
    """
    from travel_planner.orchestration.parallel import ParallelResult, ParallelTask

    try:
        agent = AccommodationAgent()
        result = agent.invoke(state)

        return {
            "result": ParallelResult(
                task_type=ParallelTask.ACCOMMODATION, result=result, completed=True
            )
        }
    except Exception as e:
        logger.error(f"Error in accommodation task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.ACCOMMODATION,
                result={},
                error=str(e),
                completed=False,
            )
        }
