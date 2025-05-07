"""
Error handling and recovery for the travel planner workflow.

This module implements functions for handling errors and recovering from
interruptions in the workflow, including checkpointing and resumption.
"""

from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def handle_error(state: TravelPlanningState) -> TravelPlanningState:
    """
    Handle an error in the workflow.

    Args:
        state: Current travel planning state

    Returns:
        Updated state with error handling
    """
    from travel_planner.orchestration.serialization.checkpoint import (
        save_state_checkpoint,
    )

    # Get the error message
    error_message = state.error or "Unknown error"

    # Add error to the conversation history
    state.conversation_history.append(
        {"role": "system", "content": f"Error occurred: {error_message}"}
    )

    # Create checkpoint for potential recovery
    checkpoint_id = save_state_checkpoint(state)

    # Update checkpoint ID in state
    state.checkpoint_id = checkpoint_id

    logger.error(f"Handled error: {error_message}. Created checkpoint: {checkpoint_id}")

    return state


def handle_interruption(state: TravelPlanningState) -> TravelPlanningState:
    """
    Handle a workflow interruption.

    Args:
        state: Current travel planning state

    Returns:
        Updated state with interruption handling
    """
    from travel_planner.orchestration.serialization.checkpoint import (
        save_state_checkpoint,
    )

    # Mark the state as interrupted if not already
    if not state.interrupted:
        state.mark_interrupted("User requested interruption")

    # Create a checkpoint and persist it
    checkpoint_id = save_state_checkpoint(state)

    # Update checkpoint ID in state
    state.checkpoint_id = checkpoint_id

    # Add note to conversation history
    state.conversation_history.append(
        {
            "role": "system",
            "content": (
                f"Workflow interrupted: {state.interruption_reason}. "
                f"Checkpoint ID: {checkpoint_id}"
            ),
        }
    )

    logger.info(
        f"Handled interruption: {state.interruption_reason}. "
        f"Created checkpoint: {checkpoint_id}"
    )

    return state
