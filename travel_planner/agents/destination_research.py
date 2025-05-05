"""
Destination Research Agent for the travel planner system.

This module implements the specialized agent responsible for researching 
destination information, analyzing travel advisories, providing weather 
insights, and identifying points of interest for potential travel destinations.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

from travel_planner.agents.base import BaseAgent, AgentConfig, AgentContext
from travel_planner.utils import (
    AgentLogger, 
    handle_errors, 
    with_retry,
    AgentExecutionError,
    safe_serialize
)


@dataclass
class DestinationInfo:
    """Information about a travel destination."""
    name: str
    country: str
    description: str = ""
    weather: Dict[str, Any] = field(default_factory=dict)
    best_times_to_visit: List[str] = field(default_factory=list)
    points_of_interest: List[Dict[str, Any]] = field(default_factory=list)
    local_transportation: List[Dict[str, Any]] = field(default_factory=list)
    travel_advisories: List[Dict[str, Any]] = field(default_factory=list)
    visa_requirements: str = ""
    language: str = ""
    currency: str = ""
    timezone: str = ""
    cost_index: float = 0.0  # Relative cost of living index


@dataclass
class DestinationContext(AgentContext):
    """Context for the destination research agent."""
    query: str = ""
    destinations: List[DestinationInfo] = field(default_factory=list)
    selected_destination: Optional[DestinationInfo] = None
    travel_dates: Dict[str, str] = field(default_factory=dict)
    search_results: Dict[str, Any] = field(default_factory=dict)


class DestinationResearchAgent(BaseAgent[DestinationContext]):
    """
    Specialized agent for researching travel destinations.
    
    This agent is responsible for:
    1. Analyzing user preferences to suggest appropriate destinations
    2. Researching detailed information about destinations
    3. Checking travel advisories and visa requirements
    4. Providing weather and seasonal information
    5. Identifying key points of interest and activities
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the destination research agent.
        
        Args:
            config: Configuration for the agent (optional)
        """
        default_config = AgentConfig(
            name="Destination Research",
            instructions=(
                "You are an AI destination research specialist for travel planning. "
                "Your expertise is in providing comprehensive, accurate information about "
                "travel destinations worldwide. Research and analyze destinations based on "
                "user preferences, provide detailed information about points of interest, "
                "local travel conditions, weather patterns, and travel advisories. "
                "Your goal is to help travelers make informed decisions about their destinations."
            ),
            tools=[
                # We would typically define tool functions here for:
                # - Searching travel information
                # - Checking weather forecasts 
                # - Looking up travel advisories
                # - Finding points of interest
            ]
        )
        super().__init__(config or default_config, DestinationContext)
        self.logger = AgentLogger(self.name)
    
    async def run(
        self, 
        input_data: Union[str, List[Dict[str, Any]]], 
        context: Optional[DestinationContext] = None
    ) -> Dict[str, Any]:
        """
        Run the destination research agent with the provided input and context.
        
        Args:
            input_data: User input or conversation history
            context: Optional destination research context
            
        Returns:
            Updated context and research results
        """
        self.logger.info(f"Running destination research agent with input: {input_data if isinstance(input_data, str) else '...'}")
        
        # Initialize context if not provided
        if context is None:
            context = DestinationContext()
        
        # If input is a string, set it as the query
        if isinstance(input_data, str):
            context.query = input_data
        
        try:
            result = await self.process(input_data, context)
            return {
                "context": context,
                "result": result,
            }
        except Exception as e:
            error_msg = f"Error in destination research agent: {str(e)}"
            self.logger.error(error_msg)
            raise AgentExecutionError(error_msg, self.name, original_error=e)
    
    @handle_errors(error_cls=AgentExecutionError)
    async def process(
        self, 
        input_data: Union[str, List[Dict[str, Any]]], 
        context: DestinationContext
    ) -> Dict[str, Any]:
        """
        Process the destination research request.
        
        Args:
            input_data: User input or conversation history
            context: Destination research context
            
        Returns:
            Research results
        """
        self.logger.info(f"Processing destination research for query: {context.query}")
        
        # Prepare messages for the model
        messages = self._prepare_messages(input_data)
        
        # Determine the type of request (destination suggestion or detailed research)
        if not context.selected_destination:
            # First, suggest destinations based on user preferences
            result = await self._suggest_destinations(context)
        else:
            # If a destination is already selected, gather detailed information about it
            result = await self._research_destination(context.selected_destination.name, context)
        
        return result
    
    async def _suggest_destinations(self, context: DestinationContext) -> Dict[str, Any]:
        """
        Suggest destinations based on user preferences.
        
        Args:
            context: Destination research context
            
        Returns:
            Dictionary with suggested destinations and reasoning
        """
        self.logger.info(f"Suggesting destinations for query: {context.query}")
        
        # Prepare a specific prompt for destination suggestions
        suggestion_prompt = (
            "Based on the user's preferences, suggest 3-5 suitable travel destinations. "
            "For each destination, provide a brief description explaining why it matches "
            "their preferences, the best time to visit, and any notable attractions. "
            "Format the output as a structured JSON object."
        )
        
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": context.query},
            {"role": "system", "content": suggestion_prompt},
        ]
        
        response = await self._call_model(messages)
        
        # Process response to extract suggested destinations
        # In a real implementation, this would parse the structured output
        # and create DestinationInfo objects
        
        # For now, we'll return the raw response
        return {"suggestions": response.get("content", "")}
    
    async def _research_destination(self, destination: str, context: DestinationContext) -> Dict[str, Any]:
        """
        Research detailed information about a specific destination.
        
        Args:
            destination: Name of the destination to research
            context: Destination research context
            
        Returns:
            Dictionary with detailed destination information
        """
        self.logger.info(f"Researching destination: {destination}")
        
        # Prepare a specific prompt for detailed destination research
        research_prompt = (
            f"Provide comprehensive information about {destination} as a travel destination. "
            "Include details about the location, weather, best times to visit, main attractions, "
            "local transportation options, visa requirements, local currency, language, and any "
            "relevant travel advisories. Format the output as a structured JSON object."
        )
        
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": f"Research {destination} as a travel destination"},
            {"role": "system", "content": research_prompt},
        ]
        
        response = await self._call_model(messages)
        
        # Process response to extract destination information
        # In a real implementation, this would parse the structured output
        # and create a DestinationInfo object
        
        # For now, we'll return the raw response
        return {"research": response.get("content", "")}
    
    @with_retry(max_attempts=3)
    async def _call_model(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Call the OpenAI API with the given messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Model response
        """
        self.logger.info(f"Calling model with {len(messages)} messages")
        
        # Log inputs for debugging
        self.logger.log_llm_input(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )
        
        try:
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            # Log the response
            self.logger.log_llm_output(model=self.config.model, response=response)
            
            # Extract the content from the response
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                return {"content": content}
            
            return {"content": "No response generated."}
            
        except Exception as e:
            self.logger.error(f"Error calling model: {str(e)}")
            raise
    
    # In a complete implementation, we would add methods for:
    # - Checking weather forecasts
    # - Retrieving travel advisories
    # - Searching for points of interest
    # - Analyzing visa requirements
    # These would typically use specialized APIs and tools