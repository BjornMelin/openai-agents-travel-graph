"""
Mock implementations for LangGraph to support testing.

This module provides mock classes and functions to simulate LangGraph functionality
in tests without requiring the actual LangGraph package. It should be imported
before any modules that depend on LangGraph.
"""

import sys
from unittest.mock import MagicMock


# LangGraph graph module mock
class StateGraph(MagicMock):
    """Mock for LangGraph StateGraph."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.add_node = MagicMock(return_value=self)
        self.add_edge = MagicMock(return_value=self)
        self.add_conditional_edges = MagicMock(return_value=self)
        self.add_branch = MagicMock(return_value=self)
        self.compile = MagicMock(return_value=self)


# LangGraph errors
class GraphError(Exception):
    """Mock for LangGraph GraphError."""

    pass


class InterruptibleError(Exception):
    """Mock for LangGraph InterruptibleError."""

    pass


class NodeError(Exception):
    """Mock for LangGraph NodeError."""

    def __init__(self, message, node_name=None):
        super().__init__(message)
        self.node_name = node_name


class ValidationError(Exception):
    """Mock for LangGraph ValidationError."""

    pass


# LangGraph human in the loop mock
def human_in_the_loop():
    """Mock for LangGraph human_in_the_loop function."""
    return MagicMock()


# LangGraph END constant
END = "END"


# Mock for ParallelBranch
class ParallelBranch(MagicMock):
    """Mock for LangGraph ParallelBranch."""

    pass


# Set up the module structure
def setup_mock_langgraph():
    """Set up the mock LangGraph modules and classes."""
    # Create module structure
    mock_langgraph = MagicMock()
    mock_graph = MagicMock()
    mock_branches = MagicMock()
    mock_human = MagicMock()
    mock_parallel = MagicMock()

    # Assign mock classes and functions
    mock_graph.StateGraph = StateGraph
    mock_graph.END = END

    mock_errors = MagicMock()
    mock_errors.GraphError = GraphError
    mock_errors.InterruptibleError = InterruptibleError
    mock_errors.NodeError = NodeError
    mock_errors.ValidationError = ValidationError

    mock_human.human_in_the_loop = human_in_the_loop
    mock_branches.human = mock_human

    mock_parallel.ParallelBranch = ParallelBranch
    mock_branches.parallel = mock_parallel

    # Set up module structure
    mock_langgraph.graph = mock_graph
    mock_langgraph.errors = mock_errors
    mock_langgraph.graph.branches = mock_branches
    mock_langgraph.graph.branches.human = mock_human
    mock_langgraph.graph.branches.parallel = mock_parallel

    # Register in sys.modules
    sys.modules["langgraph"] = mock_langgraph
    sys.modules["langgraph.graph"] = mock_graph
    sys.modules["langgraph.errors"] = mock_errors
    sys.modules["langgraph.graph.branches"] = mock_branches
    sys.modules["langgraph.graph.branches.human"] = mock_human
    sys.modules["langgraph.graph.branches.parallel"] = mock_parallel

    return {
        "langgraph": mock_langgraph,
        "graph": mock_graph,
        "errors": mock_errors,
        "branches": mock_branches,
        "human": mock_human,
    }
