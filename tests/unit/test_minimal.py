"""
Minimal test for the state graph and parallel refactoring.
"""

from unittest.mock import MagicMock


# Test that the parallel module is importable
def test_parallel_importable():
    """Verify that the parallel module can be imported."""
    from travel_planner.orchestration.parallel import ParallelResult, ParallelTask

    # Create instances to verify they work
    task = ParallelTask.FLIGHT_SEARCH
    result = ParallelResult(task_type=task, result={"test": "data"}, completed=True)

    assert task == ParallelTask.FLIGHT_SEARCH
    assert result.task_type == task
    assert result.result == {"test": "data"}
    assert result.completed is True


# Test that the core agent registry is importable
def test_agent_registry_importable():
    """Verify that the agent registry can be imported and used."""
    from travel_planner.orchestration.core.agent_registry import (
        AgentRegistry,
        get_agent,
        register_agent,
    )

    # Create a mock agent
    mock_agent = MagicMock()
    mock_agent.name = "test_agent"

    # Test registry operations
    registry = AgentRegistry()
    registry.clear()

    # Register the agent
    register_agent("test_agent", mock_agent)

    # Get the agent back
    retrieved_agent = get_agent("test_agent")

    # Verify it's the same agent
    assert retrieved_agent == mock_agent
