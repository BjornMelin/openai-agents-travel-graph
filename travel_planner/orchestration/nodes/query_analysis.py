"""
Query analysis node implementation for the travel planning workflow.

This module defines the function that analyzes user queries to extract travel
requirements and preferences using the orchestrator agent.
"""

from travel_planner.agents.orchestrator import OrchestratorAgent
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def query_analysis(state: TravelPlanningState) -> TravelPlanningState:
    """
    Analyze the user query to understand requirements and preferences.

    Args:
        state: Current travel planning state

    Returns:
        Updated travel planning state
    """
    logger.info("Starting query analysis")

    orchestrator = OrchestratorAgent()
    result = orchestrator.invoke(state)

    # Update state with query analysis results
    state.query = result.get("query", state.query)
    state.preferences = result.get("preferences", state.preferences)
    state.update_stage(WorkflowStage.QUERY_ANALYZED)

    has_destination = state.query and state.query.destination
    destination = state.query.destination if has_destination else "Unknown"
    logger.info(f"Query analyzed. Destination: {destination}")

    # Add the result to conversation history for context
    state.conversation_history.append(
        {
            "role": "system",
            "content": (
                f"Query analyzed: {destination}"
                if destination != "Unknown"
                else "Query analyzed: Destination research needed"
            ),
        }
    )

    # Add the task result
    state.add_task_result("query_analysis", result)

    return state
