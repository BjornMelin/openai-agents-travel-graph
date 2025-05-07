"""
Agent registry for the travel planner system.

This module provides a central registry for agent access, implementing
dependency injection to reduce circular dependencies and improve testability.
"""

from travel_planner.agents.base import BaseAgent
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


class AgentRegistry:
    """
    Central registry for accessing agent instances.

    This class implements the service locator pattern for agent access,
    providing a centralized way to obtain agent instances and facilitating
    dependency injection for testing.
    """

    def __init__(self):
        """Initialize the agent registry."""
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent_type: str, agent: BaseAgent) -> None:
        """
        Register an agent in the registry.

        Args:
            agent_type: The type identifier for the agent
            agent: The agent instance to register
        """
        logger.debug(f"Registering agent: {agent_type} ({agent.__class__.__name__})")
        self._agents[agent_type] = agent

    def get(self, agent_type: str) -> BaseAgent:
        """
        Get an agent from the registry.

        Args:
            agent_type: The type identifier for the agent

        Returns:
            The registered agent instance

        Raises:
            ValueError: If the agent type is not registered
        """
        if agent_type not in self._agents:
            raise ValueError(f"Agent type '{agent_type}' not registered")
        return self._agents[agent_type]

    def register_defaults(self) -> None:
        """Register all default agents in the registry."""
        from travel_planner.agents.accommodation import AccommodationAgent
        from travel_planner.agents.activity_planning import ActivityPlanningAgent
        from travel_planner.agents.budget_management import BudgetManagementAgent
        from travel_planner.agents.destination_research import DestinationResearchAgent
        from travel_planner.agents.flight_search import FlightSearchAgent
        from travel_planner.agents.orchestrator import OrchestratorAgent
        from travel_planner.agents.transportation import TransportationAgent

        self.register("orchestrator", OrchestratorAgent())
        self.register("destination_research", DestinationResearchAgent())
        self.register("flight_search", FlightSearchAgent())
        self.register("accommodation", AccommodationAgent())
        self.register("transportation", TransportationAgent())
        self.register("activity_planning", ActivityPlanningAgent())
        self.register("budget_management", BudgetManagementAgent())

        logger.info("Default agents registered")

    def clear(self) -> None:
        """Clear all registered agents (useful for testing)."""
        self._agents.clear()
        logger.debug("Agent registry cleared")


# Create a singleton instance
default_agent_registry = AgentRegistry()


def get_agent(agent_type: str) -> BaseAgent:
    """
    Get an agent from the default registry.

    Args:
        agent_type: The type identifier for the agent

    Returns:
        The registered agent instance
    """
    return default_agent_registry.get(agent_type)


def register_agent(agent_type: str, agent: BaseAgent) -> None:
    """
    Register an agent in the default registry.

    Args:
        agent_type: The type identifier for the agent
        agent: The agent instance to register
    """
    default_agent_registry.register(agent_type, agent)


def register_default_agents() -> None:
    """Register all default agents in the default registry."""
    default_agent_registry.register_defaults()
