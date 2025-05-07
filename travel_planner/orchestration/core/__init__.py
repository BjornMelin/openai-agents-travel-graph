"""
Core orchestration components for the travel planner workflow.

This package contains the core orchestration components for the travel planning workflow,
including the graph builder and agent registry for dependency injection.
"""

# Only import registry components to avoid import errors with graph_builder in tests
from travel_planner.orchestration.core.agent_registry import (
    AgentRegistry,
    default_agent_registry,
    get_agent,
    register_agent,
    register_default_agents,
)

# Import graph_builder conditionally to prevent import errors in tests
try:
    from travel_planner.orchestration.core.graph_builder import create_planning_graph

    _has_graph_builder = True
except ImportError:
    # Create a placeholder for tests
    def create_planning_graph():
        """Placeholder for tests."""
        raise NotImplementedError("Graph builder not available")

    _has_graph_builder = False

__all__ = [
    "AgentRegistry",
    "create_planning_graph",
    "default_agent_registry",
    "get_agent",
    "register_agent",
    "register_default_agents",
]
