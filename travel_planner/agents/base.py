"""
Base agent class for the travel planner system.

This module implements the foundational Agent class that all specialized
agents in the travel planner system will inherit from. It provides common
functionality and standardized interfaces for all agents.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union

from openai import OpenAI
from pydantic import BaseModel, Field

# Type variable for context
T = TypeVar('T')

class AgentContext(BaseModel):
    """Base class for agent context that can be passed between agents."""
    pass


class TravelPlannerAgentException(Exception):
    """Base exception for all agent-related exceptions."""
    pass


class InvalidConfigurationException(TravelPlannerAgentException):
    """Exception raised when agent configuration is invalid."""
    pass


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    instructions: str
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    tools: List[Any] = field(default_factory=list)


class BaseAgent(Generic[T]):
    """
    Base class for all travel planner agents.
    
    This class provides the foundation for specialized agents that handle
    different aspects of travel planning, such as destination research,
    flight search, accommodation booking, etc.
    
    The BaseAgent implements common functionality like handling API clients,
    error management, and providing a standardized interface for all agents.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        context_type: Optional[Type[T]] = None,
    ):
        """
        Initialize a base agent.
        
        Args:
            config: Configuration for the agent
            context_type: Type of context this agent handles (optional)
        """
        self.config = config
        self.client = OpenAI()
        self.context_type = context_type or AgentContext
    
    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self.config.name
    
    @property
    def instructions(self) -> str:
        """Get the instructions for the agent."""
        return self.config.instructions
    
    async def run(self, input_data: Union[str, List[Dict[str, Any]]], context: Optional[T] = None) -> Any:
        """
        Run the agent with the provided input and context.
        
        Args:
            input_data: User input or conversation history
            context: Optional context for the agent
            
        Returns:
            Agent response or result
        """
        # This would typically integrate with the OpenAI Agents SDK
        # For now, this is a simple implementation
        raise NotImplementedError("Subclasses must implement run method")
    
    async def process(self, *args, **kwargs) -> Any:
        """
        Process the input according to the agent's specialized function.
        
        This is the main method that specialized agents will implement
        to perform their specific tasks.
        """
        raise NotImplementedError("Subclasses must implement process method")
    
    def _validate_config(self) -> bool:
        """Validate the agent configuration."""
        if not self.config.name:
            raise InvalidConfigurationException("Agent name cannot be empty")
        if not self.config.instructions:
            raise InvalidConfigurationException("Agent instructions cannot be empty")
        return True

    def _prepare_messages(self, input_data: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Prepare messages for the API call.
        
        Args:
            input_data: User input or conversation history
            
        Returns:
            List of messages formatted for the API
        """
        if isinstance(input_data, str):
            messages = [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": input_data}
            ]
        else:
            # If input_data is already a list of messages, add system message if not present
            if input_data and input_data[0].get("role") != "system":
                messages = [{"role": "system", "content": self.instructions}] + input_data
            else:
                messages = input_data
        
        return messages