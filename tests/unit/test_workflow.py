"""
Unit tests for the travel planner workflow module.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import mock LangGraph before importing any modules that depend on it
from tests.unit.mock_langgraph import (
    InterruptibleError,
    NodeError,
    ValidationError,
    setup_mock_langgraph,
)

# Set up mock LangGraph modules
setup_mock_langgraph()

# Now import the modules that depend on LangGraph
from travel_planner.data.models import TravelPlan, TravelQuery  # noqa: E402
from travel_planner.orchestration.states.planning_state import (  # noqa: E402
    TravelPlanningState,
)
from travel_planner.orchestration.states.workflow_stages import (  # noqa: E402
    WorkflowStage,
)
from travel_planner.orchestration.workflow import TravelWorkflow  # noqa: E402


@pytest.fixture
def mock_graph():
    """Mock LangGraph state graph."""
    mock = MagicMock()
    # Set up arun to return an async generator that yields a final state event
    mock_event_stream = AsyncMock()

    async def mock_anext(*args, **kwargs):
        return mock_event_stream

    mock.arun.return_value = AsyncMock(__anext__=mock_anext)

    # Set up the event stream to yield an event with the final state
    async def mock_event_generator():
        yield {
            "type": "node",
            "node": "END",
            "state": TravelPlanningState(
                query=TravelQuery(raw_query="Test query"),
                plan=TravelPlan(metadata={"status": "completed"}),
                stage=WorkflowStage.COMPLETE,
            ),
        }

    mock_event_stream.__aiter__.return_value = mock_event_generator()

    return mock


@pytest.fixture
def workflow(mock_graph):
    """Create a TravelWorkflow instance with mocked components."""
    with (
        patch("travel_planner.orchestration.workflow.register_default_agents"),
        patch(
            "travel_planner.orchestration.workflow.create_planning_graph",
            return_value=mock_graph,
        ),
    ):
        return TravelWorkflow()


def test_process_query_success(workflow):
    """Test successful query processing."""
    # Mock the _execute_graph method to simply return a successful result
    test_plan = TravelPlan(metadata={"status": "completed"})

    test_state = TravelPlanningState(
        query=TravelQuery(raw_query="Plan a trip to Paris"), plan=test_plan
    )

    workflow._execute_graph = MagicMock(return_value=test_state)

    # Execute the workflow with a test query
    result = workflow.process_query("Plan a trip to Paris")

    # Verify the result is a valid travel plan
    assert isinstance(result, TravelPlan)
    assert result.metadata["status"] == "completed"


def test_process_query_validation_error(workflow):
    """Test handling of validation errors."""
    # Setup mock to raise ValidationError
    error = ValidationError("Invalid state format")
    workflow._execute_graph = MagicMock(side_effect=error)

    # Execute the workflow and verify error handling
    result = workflow.process_query("Plan a trip to Paris")

    # Verify error information in the resulting plan
    assert "error" in result.metadata
    assert result.metadata["error_type"] == "validation_error"
    assert len(result.alerts) == 1
    assert "Error" in result.alerts[0]


def test_process_query_node_error(workflow):
    """Test handling of node execution errors."""
    # Setup mock to raise NodeError
    error = NodeError("Failed to execute node")
    error.node_name = "destination_research"
    workflow._execute_graph = MagicMock(side_effect=error)

    # Execute the workflow and verify error handling
    result = workflow.process_query("Plan a trip to Paris")

    # Verify error information in the resulting plan
    assert "error" in result.metadata
    assert result.metadata["error_type"] == "node_error_destination_research"
    assert len(result.alerts) == 1
    assert "Error" in result.alerts[0]


def test_process_query_interruption(workflow):
    """Test handling of workflow interruptions."""
    # Setup mock to raise InterruptibleError
    error = InterruptibleError("User requested interruption")
    workflow._execute_graph = MagicMock(side_effect=error)

    # Mock the checkpoint saving function
    with patch(
        "travel_planner.orchestration.workflow.save_state_checkpoint",
        return_value="test_checkpoint_id",
    ):
        # Execute the workflow and verify interruption handling
        result = workflow.process_query("Plan a trip to Paris")

        # Verify interruption information in the resulting plan
        assert result.metadata["interrupted"] is True
        assert "interruption_reason" in result.metadata
        assert result.metadata["resumable"] is True
        assert "checkpoint_id" in result.metadata
        assert len(result.alerts) == 1
        assert "interruption" in result.alerts[0].lower()


def test_process_query_unexpected_error(workflow):
    """Test handling of unexpected errors."""
    # Setup mock to raise an unexpected exception
    error = Exception("Unexpected system error")
    workflow._execute_graph = MagicMock(side_effect=error)

    # Execute the workflow and verify error handling
    result = workflow.process_query("Plan a trip to Paris")

    # Verify error information in the resulting plan
    assert "error" in result.metadata
    assert result.metadata["error_type"] == "unexpected_error"
    assert len(result.alerts) == 1
    assert "Error" in result.alerts[0]


def test_resume_workflow_success(workflow, mock_graph):
    """Test successful workflow resumption."""
    # Mock the checkpoint loading function
    resumed_state = TravelPlanningState(
        query=TravelQuery(raw_query="Test query"),
        stage=WorkflowStage.DESTINATION_RESEARCHED,
        interrupted=True,
    )

    # Create the result state and plan
    final_state = TravelPlanningState(
        query=TravelQuery(raw_query="Test query"),
        plan=TravelPlan(metadata={"status": "completed"}),
        stage=WorkflowStage.COMPLETE,
    )

    with (
        patch(
            "travel_planner.orchestration.serialization.checkpoint.load_state_checkpoint",
            return_value=resumed_state,
        ),
        patch.object(workflow, "_execute_graph", return_value=final_state),
    ):
        # Resume the workflow with a test checkpoint
        result = workflow.resume_workflow("test_checkpoint_id")

        # Verify the result is a valid travel plan
        assert isinstance(result, TravelPlan)
        assert result.metadata["status"] == "completed"

        # Verify the state was properly updated before resumption
        assert resumed_state.interrupted is False
        assert resumed_state.interruption_reason is None
        assert len(resumed_state.conversation_history) == 1


def test_resume_workflow_error(workflow):
    """Test error handling during workflow resumption."""
    # Mock the checkpoint loading function to raise an exception
    error = Exception("Failed to load checkpoint")

    with patch(
        "travel_planner.orchestration.serialization.checkpoint.load_state_checkpoint",
        side_effect=error,
    ):
        # Attempt to resume the workflow
        result = workflow.resume_workflow("test_checkpoint_id")

        # Verify error information in the resulting plan
        assert "error" in result.metadata
        assert result.metadata["error_type"] == "resume_error"
        assert result.metadata["checkpoint_id"] == "test_checkpoint_id"
        assert len(result.alerts) == 1
        assert "Error resuming workflow" in result.alerts[0]
