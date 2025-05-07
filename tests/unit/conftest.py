"""
Test configuration for unit tests.
"""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_langgraph():
    """Mock LangGraph imports for testing."""
    # Mocking langgraph imported modules
    mock_langgraph = MagicMock()
    mock_graph = MagicMock()
    mock_branches = MagicMock()
    mock_human = MagicMock()
    mock_human.human_in_the_loop.return_value = MagicMock()

    # Create StateGraph mock
    mock_state_graph = MagicMock()
    mock_graph_instance = MagicMock()
    mock_state_graph.return_value = mock_graph_instance
    mock_graph_instance.add_node.return_value = mock_graph_instance
    mock_graph_instance.add_edge.return_value = mock_graph_instance
    mock_graph_instance.add_conditional_edges.return_value = mock_graph_instance
    mock_graph_instance.compile.return_value = mock_graph_instance

    mock_graph.StateGraph = mock_state_graph
    mock_branches.human = mock_human

    # Mock langgraph errors
    mock_errors = MagicMock()
    mock_errors.GraphError = Exception
    mock_errors.InterruptibleError = Exception
    mock_errors.NodeError = Exception
    mock_errors.ValidationError = Exception

    # Setup module structure
    sys.modules["langgraph"] = mock_langgraph
    sys.modules["langgraph.graph"] = mock_graph
    sys.modules["langgraph.graph.branches"] = mock_branches
    sys.modules["langgraph.graph.branches.human"] = mock_human
    sys.modules["langgraph.errors"] = mock_errors

    # Return the mocks for use in tests
    return {
        "langgraph": mock_langgraph,
        "graph": mock_graph,
        "state_graph": mock_state_graph,
        "human": mock_human,
    }
