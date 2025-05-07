"""
State representation for the travel planner workflow.

This module defines the TravelPlanningState class, which represents the
complete state of a travel planning workflow, including progress tracking,
error handling, and checkpointing capabilities.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from travel_planner.data.models import TravelPlan, TravelQuery, UserPreferences
from travel_planner.orchestration.states.workflow_stages import WorkflowStage


class TravelPlanningState(BaseModel):
    """
    State representation for the travel planning workflow.

    This state is passed between nodes in the LangGraph state graph and
    maintains the full context of the planning process. It includes
    progress tracking, interruption handling, and checkpointing capabilities.
    """

    # Core travel data
    query: TravelQuery | None = None
    preferences: UserPreferences | None = None
    plan: TravelPlan | None = None

    # Conversation and history tracking
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)

    # Workflow state management
    current_stage: WorkflowStage = WorkflowStage.START
    previous_stage: WorkflowStage | None = None
    start_time: datetime | None = None
    last_update_time: datetime | None = None
    stage_times: dict[str, datetime] = Field(default_factory=dict)

    # Error handling and recovery
    error: str | None = None
    error_count: int = 0
    retry_count: dict[str, int] = Field(default_factory=dict)

    # Interruption handling
    interrupted: bool = False
    interruption_reason: str | None = None
    checkpoint_id: str | None = None

    # Progress tracking
    progress: float = 0.0  # 0.0 to 1.0
    stage_progress: dict[str, float] = Field(default_factory=dict)

    # Parallel execution tracking
    parallel_tasks: list[str] = Field(default_factory=list)
    completed_tasks: list[str] = Field(default_factory=list)
    task_results: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Human feedback and guidance
    human_feedback: list[dict[str, Any]] = Field(default_factory=list)
    guidance_requested: bool = False

    def __init__(self, **data):
        """Initialize the travel planning state with timing information."""
        super().__init__(**data)
        current_time = datetime.now()
        if self.start_time is None:
            self.start_time = current_time
        self.last_update_time = current_time
        self.stage_times[str(self.current_stage)] = current_time

    def update_stage(self, new_stage: WorkflowStage) -> None:
        """
        Update the current stage and related timing information.

        Args:
            new_stage: The new workflow stage
        """
        self.previous_stage = self.current_stage
        self.current_stage = new_stage
        current_time = datetime.now()
        self.last_update_time = current_time
        self.stage_times[str(new_stage)] = current_time

        # Update progress based on stage
        stage_weights = {
            WorkflowStage.START: 0.0,
            WorkflowStage.QUERY_ANALYZED: 0.1,
            WorkflowStage.DESTINATION_RESEARCHED: 0.2,
            WorkflowStage.FLIGHTS_SEARCHED: 0.4,
            WorkflowStage.ACCOMMODATION_SEARCHED: 0.5,
            WorkflowStage.TRANSPORTATION_PLANNED: 0.6,
            WorkflowStage.ACTIVITIES_PLANNED: 0.8,
            WorkflowStage.BUDGET_MANAGED: 0.9,
            WorkflowStage.COMPLETE: 1.0,
            # Equivalent to completing several stages at once
            WorkflowStage.PARALLEL_SEARCH_COMPLETED: 0.6,
        }
        self.progress = stage_weights.get(new_stage, self.progress)

    def mark_interrupted(self, reason: str) -> None:
        """
        Mark the state as interrupted with a specific reason.

        Args:
            reason: The reason for the interruption
        """
        self.interrupted = True
        self.interruption_reason = reason
        previous_stage = self.current_stage
        self.update_stage(WorkflowStage.INTERRUPTED)
        # Preserve the actual stage we were interrupted at
        self.previous_stage = previous_stage
        self.checkpoint_id = f"checkpoint_{uuid.uuid4().hex}"

    def mark_error(self, error_message: str) -> None:
        """
        Mark the state with an error.

        Args:
            error_message: Description of the error
        """
        self.error = error_message
        self.error_count += 1
        previous_stage = self.current_stage
        self.update_stage(WorkflowStage.ERROR)
        # Preserve the stage where the error occurred
        self.previous_stage = previous_stage

    def add_human_feedback(self, feedback: dict[str, Any]) -> None:
        """
        Add human feedback to the state.

        Args:
            feedback: Dictionary with feedback information
        """
        feedback["timestamp"] = datetime.now().isoformat()
        self.human_feedback.append(feedback)

        # Add feedback to conversation history for context
        self.conversation_history.append(
            {"role": "human", "content": feedback.get("content", ""), "feedback": True}
        )

    def add_task_result(
        self, task_name: str, result: dict[str, Any], error: str | None = None
    ) -> None:
        """
        Add a task result to the state.

        Args:
            task_name: Name of the completed task
            result: Result data from the task
            error: Optional error information
        """
        self.task_results[task_name] = {
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }

        if task_name not in self.completed_tasks:
            self.completed_tasks.append(task_name)

    def should_retry(self, stage: str, max_retries: int = 3) -> bool:
        """
        Determine if a failed stage should be retried.

        Args:
            stage: The stage that failed
            max_retries: Maximum number of retries allowed

        Returns:
            True if a retry should be attempted, False otherwise
        """
        current_retries = self.retry_count.get(stage, 0)
        if current_retries < max_retries:
            self.retry_count[stage] = current_retries + 1
            return True
        return False

    def create_checkpoint(self) -> dict[str, Any]:
        """
        Create a serializable checkpoint of the current state.

        Returns:
            Dictionary with serialized state data
        """
        checkpoint_data = self.model_dump()

        # Generate a unique checkpoint ID if not already set
        if not self.checkpoint_id:
            self.checkpoint_id = f"checkpoint_{uuid.uuid4().hex}"

        checkpoint_data["checkpoint_id"] = self.checkpoint_id
        checkpoint_data["checkpoint_time"] = datetime.now().isoformat()

        # Convert datetime objects to ISO format strings for serialization
        if self.start_time:
            checkpoint_data["start_time"] = self.start_time.isoformat()

        if self.last_update_time:
            checkpoint_data["last_update_time"] = self.last_update_time.isoformat()

        # Convert stage_times datetime objects to strings
        if self.stage_times:
            checkpoint_data["stage_times"] = {
                k: v.isoformat() if v else None for k, v in self.stage_times.items()
            }

        return checkpoint_data

    def from_checkpoint(self, checkpoint_data: dict[str, Any]) -> None:
        """
        Load state from a checkpoint.

        Args:
            checkpoint_data: Checkpoint data to load
        """
        # Handle special fields first

        # Convert stage strings to enums
        current_stage_str = checkpoint_data.get("current_stage", "start")
        checkpoint_data["current_stage"] = (
            WorkflowStage(current_stage_str)
            if current_stage_str
            else WorkflowStage.START
        )

        previous_stage_str = checkpoint_data.get("previous_stage")
        if previous_stage_str:
            checkpoint_data["previous_stage"] = WorkflowStage(previous_stage_str)

        # Convert ISO timestamp strings back to datetime objects
        start_time_str = checkpoint_data.get("start_time")
        if start_time_str:
            self.start_time = datetime.fromisoformat(start_time_str)

        last_update_time_str = checkpoint_data.get("last_update_time")
        if last_update_time_str:
            self.last_update_time = datetime.fromisoformat(last_update_time_str)

        # Convert stage_times strings back to datetime objects
        stage_times_data = checkpoint_data.get("stage_times", {})
        for stage, time_str in stage_times_data.items():
            if time_str:
                self.stage_times[stage] = datetime.fromisoformat(time_str)

        # Now update all the basic attributes
        excluded_fields = ["start_time", "last_update_time", "stage_times"]
        for key, value in checkpoint_data.items():
            if key not in excluded_fields and hasattr(self, key):
                setattr(self, key, value)
