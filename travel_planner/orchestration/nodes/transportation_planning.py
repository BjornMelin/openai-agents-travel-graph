"""
Transportation planning node implementation for the travel planning workflow.

This module defines functions for planning transportation using the transportation
agent, both for individual execution and as part of parallel processing.
"""

from travel_planner.agents.transportation import TransportationAgent
from travel_planner.orchestration.nodes.base_node import (
    create_node_function,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


# Using the factory function for a simpler implementation
transportation_planning = create_node_function(
    agent_class=TransportationAgent,
    task_name="transportation_planning",
    complete_stage=WorkflowStage.TRANSPORTATION_PLANNED,
    result_field="transportation_options",
    plan_field="transportation",
    message_template="Local transportation planned with {count} options",
)


def transportation_task(state: TravelPlanningState) -> dict[str, any]:
    """
    Execute transportation planning task in parallel branch.

    Args:
        state: Current travel planning state

    Returns:
        Dictionary with task results
    """
    from travel_planner.orchestration.parallel import ParallelResult, ParallelTask

    try:
        agent = TransportationAgent()
        result = agent.invoke(state)

        return {
            "result": ParallelResult(
                task_type=ParallelTask.TRANSPORTATION, result=result, completed=True
            )
        }
    except Exception as e:
        logger.error(f"Error in transportation task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.TRANSPORTATION,
                result={},
                error=str(e),
                completed=False,
            )
        }
