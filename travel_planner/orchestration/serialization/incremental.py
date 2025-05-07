"""
Incremental checkpoint system for the travel planner workflow.

This module provides an optimized checkpoint system that only stores
changes since the last checkpoint, significantly reducing storage
requirements and performance impact for large state objects.
"""

import json
from datetime import datetime
from typing import Any

from travel_planner.orchestration.serialization.checkpoint import CheckpointManager
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


class IncrementalCheckpoint:
    """
    Data structure for an incremental checkpoint.

    Stores only the differences between a state and its parent state,
    rather than the entire state.
    """

    def __init__(
        self,
        checkpoint_id: str,
        data: dict[str, Any],
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize a new incremental checkpoint.

        Args:
            checkpoint_id: Unique ID for this checkpoint
            data: Dictionary of changes since parent
            parent_id: ID of the parent checkpoint, if any
            metadata: Additional metadata for the checkpoint
        """
        self.checkpoint_id = checkpoint_id
        self.data = data
        self.parent_id = parent_id
        self.metadata = metadata or {
            "timestamp": datetime.now().isoformat(),
            "size": len(json.dumps(data)),
        }


class IncrementalCheckpointManager:
    """
    Manages incremental checkpoints for the travel planning workflow.

    This manager creates and restores checkpoints that only store the
    differences between successive states, reducing storage requirements
    and improving performance for large state objects.
    """

    def __init__(self, base_manager: CheckpointManager | None = None):
        """
        Initialize the incremental checkpoint manager.

        Args:
            base_manager: Optional base checkpoint manager
        """
        from travel_planner.orchestration.serialization.checkpoint import (
            default_checkpoint_manager,
        )

        self.base_manager = base_manager or default_checkpoint_manager
        self.checkpoints: dict[str, IncrementalCheckpoint] = {}
        self.last_full_checkpoint_id: str | None = None
        self.chain_length = 0
        # Maximum number of incremental checkpoints before a full one
        self.max_chain_length = 5

    def save_checkpoint(self, state: TravelPlanningState) -> str:
        """
        Save an incremental checkpoint of the workflow state.

        Args:
            state: Current workflow state

        Returns:
            ID of the new checkpoint
        """
        # Generate checkpoint ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plan_id = state.plan.id if state.plan else "noplan"
        checkpoint_id = f"incr_{state.current_stage.value}_{timestamp}_{plan_id}"

        # Check if we need a full checkpoint
        if (
            not self.last_full_checkpoint_id
            or self.chain_length >= self.max_chain_length
        ):
            # Create a full checkpoint
            full_checkpoint_id = self.base_manager.save_checkpoint(state)
            self.last_full_checkpoint_id = full_checkpoint_id
            self.chain_length = 0

            # Create metadata for the incremental checkpoint
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "stage": str(state.current_stage),
                "is_full": True,
                "full_checkpoint_id": full_checkpoint_id,
            }

            # Store reference to the full checkpoint
            self.checkpoints[checkpoint_id] = IncrementalCheckpoint(
                checkpoint_id=checkpoint_id,
                data={},  # No incremental data for a full checkpoint
                parent_id=full_checkpoint_id,
                metadata=metadata,
            )

            logger.info(f"Created full checkpoint: {full_checkpoint_id}")
            return checkpoint_id

        # Create incremental checkpoint
        if self.last_full_checkpoint_id:
            # Load the previous state
            previous_state = self.base_manager.load_checkpoint(
                self.last_full_checkpoint_id
            )

            # Calculate differences
            diff_data = self._calculate_state_diff(previous_state, state)

            # Create checkpoint
            self.checkpoints[checkpoint_id] = IncrementalCheckpoint(
                checkpoint_id=checkpoint_id,
                data=diff_data,
                parent_id=self.last_full_checkpoint_id,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "stage": str(state.current_stage),
                    "is_incremental": True,
                    "parent_checkpoint_id": self.last_full_checkpoint_id,
                    "diff_size": len(json.dumps(diff_data)),
                },
            )

            self.chain_length += 1
            logger.info(
                f"Created incremental checkpoint: {checkpoint_id} "
                f"(based on {self.last_full_checkpoint_id})"
            )
            return checkpoint_id
        else:
            # No previous checkpoint, create a full one
            return self.save_checkpoint(state)

    def load_checkpoint(self, checkpoint_id: str) -> TravelPlanningState:
        """
        Load a state from an incremental checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to load

        Returns:
            Reconstructed workflow state

        Raises:
            ValueError: If the checkpoint doesn't exist
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        checkpoint = self.checkpoints[checkpoint_id]

        # If this is a reference to a full checkpoint, load it directly
        if "full_checkpoint_id" in checkpoint.metadata:
            full_id = checkpoint.metadata["full_checkpoint_id"]
            return self.base_manager.load_checkpoint(full_id)

        # Load the parent checkpoint
        if not checkpoint.parent_id:
            raise ValueError(f"Incremental checkpoint {checkpoint_id} has no parent")

        # Load the base state
        base_state = self.base_manager.load_checkpoint(checkpoint.parent_id)

        # Apply the incremental changes
        updated_state = self._apply_state_diff(base_state, checkpoint.data)

        logger.info(f"Loaded incremental checkpoint: {checkpoint_id}")
        return updated_state

    def _calculate_state_diff(
        self, base_state: TravelPlanningState, current_state: TravelPlanningState
    ) -> dict[str, Any]:
        """
        Calculate the differences between two states.

        Args:
            base_state: Previous state
            current_state: Current state

        Returns:
            Dictionary of changes
        """
        diff = {}

        # Convert states to dictionaries for comparison
        base_dict = base_state.model_dump()
        current_dict = current_state.model_dump()

        # Find changed/added fields
        for key, value in current_dict.items():
            if key not in base_dict or base_dict[key] != value:
                diff[key] = value

        return diff

    def _apply_state_diff(
        self, base_state: TravelPlanningState, diff: dict[str, Any]
    ) -> TravelPlanningState:
        """
        Apply a diff to a base state.

        Args:
            base_state: Base state to apply changes to
            diff: Dictionary of changes to apply

        Returns:
            Updated state
        """
        # Create a copy of the base state
        updated_state = base_state.model_copy(deep=True)

        # Apply changes
        for key, value in diff.items():
            if hasattr(updated_state, key):
                setattr(updated_state, key, value)

        return updated_state


# Create a singleton instance
incremental_checkpoint_manager = IncrementalCheckpointManager()


def save_incremental_checkpoint(state: TravelPlanningState) -> str:
    """
    Save an incremental checkpoint using the default manager.

    Args:
        state: Current workflow state

    Returns:
        ID of the new checkpoint
    """
    return incremental_checkpoint_manager.save_checkpoint(state)


def load_incremental_checkpoint(checkpoint_id: str) -> TravelPlanningState:
    """
    Load a state from an incremental checkpoint using the default manager.

    Args:
        checkpoint_id: ID of the checkpoint to load

    Returns:
        Reconstructed workflow state
    """
    return incremental_checkpoint_manager.load_checkpoint(checkpoint_id)
