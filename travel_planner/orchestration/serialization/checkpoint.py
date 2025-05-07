"""
Checkpoint system for the travel planner state graph.

This module provides functionality for persisting and recovering workflow state
checkpoints, allowing for resumption of interrupted workflows and reliable
error recovery.
"""

import json
import os
from datetime import datetime
from typing import Any

from travel_planner.data.models import TravelPlan, TravelQuery, UserPreferences
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)

# Default directory for storing checkpoints
DEFAULT_CHECKPOINT_DIR = os.path.expanduser("~/.travel_planner/checkpoints")


class CheckpointManager:
    """
    Manages workflow state checkpoints for persistence and recovery.

    This class provides the ability to save workflow state checkpoints to
    disk, load them for workflow resumption, and manage checkpoint cleanup.
    """

    def __init__(self, checkpoint_dir: str | None = None):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoints (optional)
        """
        self.checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Keep track of active checkpoints
        self.active_checkpoints: dict[str, dict[str, Any]] = {}

    def save_checkpoint(self, state: TravelPlanningState) -> str:
        """
        Save a checkpoint of the current workflow state.

        Args:
            state: Current workflow state to save

        Returns:
            Checkpoint ID of the saved checkpoint
        """
        # Create checkpoint data
        checkpoint_data = state.create_checkpoint()
        checkpoint_id = checkpoint_data["checkpoint_id"]

        # Add additional metadata
        checkpoint_data["timestamp"] = datetime.now().isoformat()
        checkpoint_data["workflow_stage"] = str(state.current_stage)

        # Save to in-memory cache
        self.active_checkpoints[checkpoint_id] = checkpoint_data

        # Save to disk
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Saved checkpoint {checkpoint_id} at stage {state.current_stage}")

        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> TravelPlanningState:
        """
        Load a workflow state from a checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to load

        Returns:
            Reconstructed workflow state

        Raises:
            ValueError: If the checkpoint doesn't exist
        """
        # Check in-memory cache first
        if checkpoint_id in self.active_checkpoints:
            checkpoint_data = self.active_checkpoints[checkpoint_id]
        else:
            # Check on disk
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Checkpoint {checkpoint_id} not found")

            # Load checkpoint data
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            # Add to in-memory cache
            self.active_checkpoints[checkpoint_id] = checkpoint_data

        # Reconstruct the state
        state = self._reconstruct_state_from_checkpoint(checkpoint_data)

        logger.info(f"Loaded checkpoint {checkpoint_id} at stage {state.current_stage}")

        return state

    def list_checkpoints(
        self, stage: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        List available checkpoints with optional filtering.

        Args:
            stage: Filter by workflow stage (optional)
            limit: Maximum number of checkpoints to return

        Returns:
            List of checkpoint metadata dictionaries
        """
        checkpoints = []

        # First gather all checkpoints from disk
        for filename in os.listdir(self.checkpoint_dir):
            if not filename.endswith(".json"):
                continue

            checkpoint_path = os.path.join(self.checkpoint_dir, filename)

            try:
                with open(checkpoint_path) as f:
                    metadata = json.load(f)

                    # Apply stage filter if provided
                    if stage and metadata.get("workflow_stage") != stage:
                        continue

                    # Extract minimal metadata
                    checkpoint_info = {
                        "checkpoint_id": metadata.get("checkpoint_id"),
                        "timestamp": metadata.get("timestamp"),
                        "workflow_stage": metadata.get("workflow_stage"),
                        "plan_id": metadata.get("plan", {})
                        .get("metadata", {})
                        .get("id"),
                        "destination": metadata.get("plan", {})
                        .get("destination", {})
                        .get("name"),
                        "error": metadata.get("error"),
                    }

                    checkpoints.append(checkpoint_info)
            except Exception as e:
                logger.error(f"Error reading checkpoint {filename}: {e}")

        # Sort by timestamp (newest first) and limit
        checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return checkpoints[:limit]

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        # Remove from in-memory cache
        if checkpoint_id in self.active_checkpoints:
            del self.active_checkpoints[checkpoint_id]

        # Remove from disk
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True

        logger.warning(f"Checkpoint {checkpoint_id} not found for deletion")
        return False

    def cleanup_old_checkpoints(self, max_age_days: int = 7) -> int:
        """
        Clean up old checkpoints.

        Args:
            max_age_days: Maximum age of checkpoints to keep (in days)

        Returns:
            Number of checkpoints deleted
        """
        now = datetime.now()
        deleted_count = 0

        for filename in os.listdir(self.checkpoint_dir):
            if not filename.endswith(".json"):
                continue

            checkpoint_path = os.path.join(self.checkpoint_dir, filename)

            try:
                # Check file age
                file_mtime = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
                age_days = (now - file_mtime).days

                if age_days > max_age_days:
                    # Load the file to get checkpoint ID
                    with open(checkpoint_path) as f:
                        metadata = json.load(f)
                        checkpoint_id = metadata.get("checkpoint_id")

                    # Delete the checkpoint
                    if checkpoint_id and self.delete_checkpoint(checkpoint_id):
                        deleted_count += 1
            except Exception as e:
                logger.error(f"Error cleaning up checkpoint {filename}: {e}")

        logger.info(f"Cleaned up {deleted_count} old checkpoints")
        return deleted_count

    def _parse_datetime(self, datetime_str: str | None) -> datetime | None:
        """Parse datetime string to datetime object."""
        return datetime.fromisoformat(datetime_str) if datetime_str else None

    def _reconstruct_state_from_checkpoint(
        self, checkpoint_data: dict[str, Any]
    ) -> TravelPlanningState:
        """
        Reconstruct a TravelPlanningState from checkpoint data.

        Args:
            checkpoint_data: Dictionary with checkpoint data

        Returns:
            Reconstructed TravelPlanningState
        """
        # Extract plan data if available
        plan_data = checkpoint_data.get("plan")
        plan = TravelPlan.model_validate(plan_data) if plan_data else None

        # Extract query data if available
        query_data = checkpoint_data.get("query")
        query = TravelQuery.model_validate(query_data) if query_data else None

        # Extract preferences data if available
        preferences_data = checkpoint_data.get("preferences")
        preferences = (
            UserPreferences.model_validate(preferences_data)
            if preferences_data
            else None
        )

        # Convert string stage to enum
        current_stage_str = checkpoint_data.get("current_stage", "start")
        current_stage = (
            WorkflowStage(current_stage_str)
            if current_stage_str
            else WorkflowStage.START
        )

        # Extract previous stage if available
        previous_stage_str = checkpoint_data.get("previous_stage")
        previous_stage = (
            WorkflowStage(previous_stage_str) if previous_stage_str else None
        )

        # Reconstruct state
        state = TravelPlanningState(
            query=query,
            preferences=preferences,
            plan=plan,
            conversation_history=checkpoint_data.get("conversation_history", []),
            current_stage=current_stage,
            previous_stage=previous_stage,
            error=checkpoint_data.get("error"),
            error_count=checkpoint_data.get("error_count", 0),
            retry_count=checkpoint_data.get("retry_count", {}),
            interrupted=checkpoint_data.get("interrupted", False),
            interruption_reason=checkpoint_data.get("interruption_reason"),
            checkpoint_id=checkpoint_data.get("checkpoint_id"),
            progress=checkpoint_data.get("progress", 0.0),
            stage_progress=checkpoint_data.get("stage_progress", {}),
            parallel_tasks=checkpoint_data.get("parallel_tasks", []),
            completed_tasks=checkpoint_data.get("completed_tasks", []),
            task_results=checkpoint_data.get("task_results", {}),
            human_feedback=checkpoint_data.get("human_feedback", []),
            guidance_requested=checkpoint_data.get("guidance_requested", False),
            start_time=self._parse_datetime(checkpoint_data.get("start_time")),
            last_update_time=self._parse_datetime(
                checkpoint_data.get("last_update_time")
            ),
        )

        # Reconstruct stage_times as datetimes
        state.stage_times = {}
        for stage_name, time_str in checkpoint_data.get("stage_times", {}).items():
            if time_str:
                state.stage_times[stage_name] = datetime.fromisoformat(time_str)

        return state


# Create a singleton instance
default_checkpoint_manager = CheckpointManager()


def save_state_checkpoint(state: TravelPlanningState) -> str:
    """
    Save a checkpoint of the current workflow state using the default manager.

    Args:
        state: Current workflow state to save

    Returns:
        Checkpoint ID of the saved checkpoint
    """
    return default_checkpoint_manager.save_checkpoint(state)


def load_state_checkpoint(checkpoint_id: str) -> TravelPlanningState:
    """
    Load a workflow state from a checkpoint using the default manager.

    Args:
        checkpoint_id: ID of the checkpoint to load

    Returns:
        Reconstructed workflow state
    """
    return default_checkpoint_manager.load_checkpoint(checkpoint_id)


def list_state_checkpoints(
    stage: str | None = None, limit: int = 10
) -> list[dict[str, Any]]:
    """
    List available checkpoints with optional filtering using the default manager.

    Args:
        stage: Filter by workflow stage (optional)
        limit: Maximum number of checkpoints to return

    Returns:
        List of checkpoint metadata dictionaries
    """
    return default_checkpoint_manager.list_checkpoints(stage=stage, limit=limit)


def delete_state_checkpoint(checkpoint_id: str) -> bool:
    """
    Delete a checkpoint using the default manager.

    Args:
        checkpoint_id: ID of the checkpoint to delete

    Returns:
        True if deleted successfully, False otherwise
    """
    return default_checkpoint_manager.delete_checkpoint(checkpoint_id)
