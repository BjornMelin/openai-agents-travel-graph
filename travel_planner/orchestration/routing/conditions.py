"""
Routing conditions for the travel planner workflow.

This module contains functions that determine workflow routing based on state analysis,
including error detection, recovery decisions, and human intervention requirements.
"""

from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage

# Constants
MAX_ERROR_COUNT = 3  # Maximum number of errors before recovery is not possible
# Number of errors that trigger human intervention
HUMAN_INTERVENTION_ERROR_THRESHOLD = 2


def query_research_needed(state: TravelPlanningState) -> str:
    """
    Determine if destination research is needed based on query analysis.

    Args:
        state: Current travel planning state

    Returns:
        Next stage in the workflow
    """
    # This is a router function that helps determine the next stage
    # based on the current state
    if not state.query or not state.query.destination:
        return "research_destination"
    return "flight_search"


def has_error(state: TravelPlanningState) -> str:
    """
    Check if the state has an error.

    Args:
        state: Current travel planning state

    Returns:
        "true" if the state has an error, "false" otherwise
    """
    if state.error or state.current_stage == WorkflowStage.ERROR:
        return "true"
    return "false"


def error_recoverable(state: TravelPlanningState) -> str:
    """
    Determine if the error in the state is recoverable.

    Args:
        state: Current travel planning state

    Returns:
        "true" if the error is recoverable, "false" otherwise
    """
    # Get the stage that had the error
    error_stage = str(state.previous_stage) if state.previous_stage else "unknown"

    # Check if we can retry this stage
    if state.should_retry(error_stage):
        return "true"

    # If error count is too high, not recoverable
    if state.error_count > MAX_ERROR_COUNT:
        return "false"

    return "true"


def recover_to_stage(state: TravelPlanningState) -> str:
    """
    Determine which stage to recover to after an error.

    Args:
        state: Current travel planning state

    Returns:
        Name of the stage to recover to
    """
    # Get the stage that had the error
    error_stage = str(state.previous_stage) if state.previous_stage else "analyze_query"

    # Clear the error state
    state.error = None
    state.update_stage(
        state.previous_stage if state.previous_stage else WorkflowStage.START
    )

    # Add a note to the conversation history
    state.conversation_history.append(
        {"role": "system", "content": f"Recovering from error, retrying {error_stage}"}
    )

    return error_stage


def needs_human_intervention(state: TravelPlanningState) -> str:
    """
    Determine if human intervention is needed.

    Args:
        state: Current travel planning state

    Returns:
        "true" if human intervention is needed, "false" otherwise
    """
    # Check if the state has requested guidance
    if state.guidance_requested:
        return "true"

    # Check if we have too many errors (might need human help)
    if state.error_count >= HUMAN_INTERVENTION_ERROR_THRESHOLD:
        return "true"

    # Check if we have an interrupted state
    if state.interrupted:
        return "true"

    return "false"


def continue_after_intervention(state: TravelPlanningState) -> str:
    """
    Determine where to continue after human intervention.

    Args:
        state: Current travel planning state

    Returns:
        Name of the node to continue at
    """
    # Clear intervention flags
    state.guidance_requested = False

    # If we were interrupted, stay interrupted until explicitly resumed
    if state.interrupted:
        return "END"

    # Get the stage we should return to (previous stage or start)
    return_stage = (
        str(state.previous_stage) if state.previous_stage else "analyze_query"
    )

    # Add note to conversation history
    state.conversation_history.append(
        {
            "role": "system",
            "content": f"Continuing workflow at {return_stage} after human intervention",
        }
    )

    return return_stage


def plan_complete(state: TravelPlanningState) -> bool:
    """
    Check if the travel plan is complete.

    Args:
        state: Current travel planning state

    Returns:
        True if the plan is complete, False otherwise
    """
    # Check if the workflow has been marked as complete
    if state.current_stage == WorkflowStage.COMPLETE:
        return True

    # Check if all required components of the travel plan are present
    if not state.plan:
        return False

    required_fields = [
        state.plan.destination,
        state.plan.flights,
        state.plan.accommodation,
        state.plan.activities,
        state.plan.transportation,
        state.plan.budget,
    ]

    # Check if all required fields are present
    fields_complete = all(required_fields)

    # If all fields are complete but state isn't marked complete, update it
    if fields_complete and state.current_stage != WorkflowStage.COMPLETE:
        state.update_stage(WorkflowStage.COMPLETE)

    return fields_complete
