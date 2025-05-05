"""
Flight Search Agent for the travel planner system.

This module implements the specialized agent responsible for searching,
comparing, and recommending flight options for the travel itinerary.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from travel_planner.agents.base import BaseAgent, AgentConfig, AgentContext
from travel_planner.utils import (
    AgentLogger, 
    handle_errors, 
    with_retry,
    AgentExecutionError,
    safe_serialize,
    format_price
)


class CabinClass(str, Enum):
    """Flight cabin classes."""
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


@dataclass
class FlightLeg:
    """A single flight leg (segment)."""
    airline: str
    flight_number: str
    departure_airport: str
    departure_time: str
    arrival_airport: str
    arrival_time: str
    duration_minutes: int
    aircraft: Optional[str] = None


@dataclass
class FlightOption:
    """A flight option with one or more legs."""
    id: str
    price: float
    currency: str
    cabin_class: CabinClass
    legs: List[FlightLeg]
    layover_count: int
    total_duration_minutes: int
    baggage_allowance: Optional[str] = None
    refundable: bool = False
    changeable: bool = False
    eco_friendly: bool = False
    amenities: List[str] = field(default_factory=list)
    
    @property
    def formatted_price(self) -> str:
        """Get the formatted price with currency symbol."""
        return format_price(self.price, self.currency)
    
    @property
    def formatted_duration(self) -> str:
        """Get the formatted total duration as hours and minutes."""
        hours, minutes = divmod(self.total_duration_minutes, 60)
        return f"{hours}h {minutes}m"


@dataclass
class FlightSearchContext(AgentContext):
    """Context for the flight search agent."""
    origin: str = ""
    destination: str = ""
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    travelers: int = 1
    cabin_class: CabinClass = CabinClass.ECONOMY
    max_price: Optional[float] = None
    currency: str = "USD"
    preferred_airlines: List[str] = field(default_factory=list)
    flight_options: List[FlightOption] = field(default_factory=list)
    selected_flight: Optional[FlightOption] = None
    search_params: Dict[str, Any] = field(default_factory=dict)
    search_results_raw: Dict[str, Any] = field(default_factory=dict)


class FlightSearchAgent(BaseAgent[FlightSearchContext]):
    """
    Specialized agent for flight search and booking.
    
    This agent is responsible for:
    1. Searching multiple flight booking sites for optimal options
    2. Filtering results based on user preferences
    3. Providing price comparisons and recommendations
    4. Monitoring flight prices if needed
    5. Presenting flight options with key details
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the flight search agent.
        
        Args:
            config: Configuration for the agent (optional)
        """
        default_config = AgentConfig(
            name="Flight Search",
            instructions=(
                "You are an AI flight search specialist for travel planning. "
                "Your expertise is in finding and comparing flight options across "
                "multiple airlines and booking platforms. Provide comprehensive flight "
                "information including prices, times, layovers, and amenities. "
                "Consider user preferences for airlines, times, cabin class, and budget. "
                "Present options clearly with pros and cons to help users make informed decisions."
            ),
            tools=[
                # We would typically define tool functions here for:
                # - Searching flight APIs
                # - Checking airline policies
                # - Monitoring flight prices
                # - Fetching airport information
            ]
        )
        super().__init__(config or default_config, FlightSearchContext)
        self.logger = AgentLogger(self.name)
    
    async def run(
        self, 
        input_data: Union[str, List[Dict[str, Any]]], 
        context: Optional[FlightSearchContext] = None
    ) -> Dict[str, Any]:
        """
        Run the flight search agent with the provided input and context.
        
        Args:
            input_data: User input or conversation history
            context: Optional flight search context
            
        Returns:
            Updated context and flight search results
        """
        self.logger.info(f"Running flight search agent with input: {input_data if isinstance(input_data, str) else '...'}")
        
        # Initialize context if not provided
        if context is None:
            context = FlightSearchContext()
        
        try:
            result = await self.process(input_data, context)
            return {
                "context": context,
                "result": result,
            }
        except Exception as e:
            error_msg = f"Error in flight search agent: {str(e)}"
            self.logger.error(error_msg)
            raise AgentExecutionError(error_msg, self.name, original_error=e)
    
    @handle_errors(error_cls=AgentExecutionError)
    async def process(
        self, 
        input_data: Union[str, List[Dict[str, Any]]], 
        context: FlightSearchContext
    ) -> Dict[str, Any]:
        """
        Process the flight search request.
        
        Args:
            input_data: User input or conversation history
            context: Flight search context
            
        Returns:
            Flight search results
        """
        self.logger.info(f"Processing flight search for: {context.origin} to {context.destination}")
        
        # Prepare messages for the model
        messages = self._prepare_messages(input_data)
        
        # Extract search parameters if not already set
        if not context.origin or not context.destination or not context.departure_date:
            await self._extract_search_params(input_data, context)
        
        # Perform the flight search
        search_results = await self._search_flights(context)
        
        # Process and rank flight options
        ranked_options = await self._rank_flight_options(search_results, context)
        
        # Store the top options in the context
        context.flight_options = ranked_options
        
        # Generate a summary of the flight options
        summary = await self._generate_options_summary(ranked_options, context)
        
        return {
            "flight_options": [self._format_flight_option(option) for option in ranked_options],
            "summary": summary
        }
    
    async def _extract_search_params(
        self, 
        input_data: Union[str, List[Dict[str, Any]]], 
        context: FlightSearchContext
    ) -> None:
        """
        Extract flight search parameters from user input.
        
        Args:
            input_data: User input or conversation history
            context: Flight search context
        """
        self.logger.info("Extracting flight search parameters")
        
        # Prepare a specific prompt for parameter extraction
        extraction_prompt = (
            "Extract flight search parameters from the user's input. Include origin, destination, "
            "dates, number of travelers, cabin class, and any airline preferences. If information "
            "is missing, keep the current values. Format the output as a structured JSON object."
        )
        
        user_input = input_data if isinstance(input_data, str) else self._get_latest_user_input(input_data)
        
        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": user_input},
        ]
        
        # Add current parameters as context if they exist
        if context and any(vars(context).values()):
            messages.append({
                "role": "system",
                "content": f"Current parameters: {safe_serialize(context)}"
            })
        
        response = await self._call_model(messages)
        
        # In a real implementation, we would parse the JSON response 
        # and update the context with the extracted parameters
        
        # For now, we'll set some example values for demonstration
        if not context.origin:
            context.origin = "NYC"
        if not context.destination:
            context.destination = "LAX"
        if not context.departure_date:
            # Set departure date to tomorrow
            tomorrow = datetime.now() + timedelta(days=1)
            context.departure_date = tomorrow.strftime("%Y-%m-%d")
        if not context.return_date:
            # Set return date to one week from now
            next_week = datetime.now() + timedelta(days=8)
            context.return_date = next_week.strftime("%Y-%m-%d")
    
    async def _search_flights(self, context: FlightSearchContext) -> List[Dict[str, Any]]:
        """
        Search for flights based on the context parameters.
        
        Args:
            context: Flight search context
            
        Returns:
            List of flight search results
        """
        self.logger.info(f"Searching flights from {context.origin} to {context.destination}")
        
        # In a real implementation, this would call flight search APIs
        # For demonstration, we'll create some mock flight options
        
        mock_flights = [
            {
                "id": "F1",
                "airline": "Gamma Airways",
                "price": 350.0,
                "currency": context.currency,
                "cabin_class": context.cabin_class.value,
                "departure_time": "08:00",
                "arrival_time": "11:30",
                "duration_minutes": 210,
                "layovers": [],
                "baggage": "1 checked bag included",
                "refundable": True,
                "eco_friendly": True
            },
            {
                "id": "F2",
                "airline": "Beta Airlines",
                "price": 280.0,
                "currency": context.currency,
                "cabin_class": context.cabin_class.value,
                "departure_time": "14:15",
                "arrival_time": "19:45",
                "duration_minutes": 330,
                "layovers": ["ORD"],
                "baggage": "Carry-on only",
                "refundable": False,
                "eco_friendly": False
            },
            {
                "id": "F3",
                "airline": "Alpha Airlines",
                "price": 420.0,
                "currency": context.currency,
                "cabin_class": context.cabin_class.value,
                "departure_time": "10:30",
                "arrival_time": "13:45",
                "duration_minutes": 195,
                "layovers": [],
                "baggage": "2 checked bags included",
                "refundable": True,
                "eco_friendly": True
            }
        ]
        
        # Store the raw search results in the context
        context.search_results_raw = {"flights": mock_flights}
        
        return mock_flights
    
    async def _rank_flight_options(
        self, 
        search_results: List[Dict[str, Any]], 
        context: FlightSearchContext
    ) -> List[FlightOption]:
        """
        Rank and convert flight options based on user preferences.
        
        Args:
            search_results: Raw flight search results
            context: Flight search context
            
        Returns:
            List of ranked FlightOption objects
        """
        self.logger.info(f"Ranking {len(search_results)} flight options")
        
        flight_options = []
        
        # Convert raw flight data to FlightOption objects
        for result in search_results:
            # Create FlightLeg objects
            legs = [
                FlightLeg(
                    airline=result["airline"],
                    flight_number=f"{result['airline'][0:2]}123",  # Mock flight number
                    departure_airport=context.origin,
                    departure_time=f"{context.departure_date}T{result['departure_time']}",
                    arrival_airport=context.destination,
                    arrival_time=f"{context.departure_date}T{result['arrival_time']}",
                    duration_minutes=result["duration_minutes"],
                )
            ]
            
            # Add connecting legs if there are layovers
            layovers = result.get("layovers", [])
            
            # Create FlightOption
            option = FlightOption(
                id=result["id"],
                price=result["price"],
                currency=result["currency"],
                cabin_class=CabinClass(result["cabin_class"]),
                legs=legs,
                layover_count=len(layovers),
                total_duration_minutes=result["duration_minutes"],
                baggage_allowance=result.get("baggage"),
                refundable=result.get("refundable", False),
                changeable=result.get("changeable", False),
                eco_friendly=result.get("eco_friendly", False),
            )
            
            flight_options.append(option)
        
        # Apply any ranking or filtering based on user preferences
        # For this demo, we'll sort by price (lowest first)
        flight_options.sort(key=lambda x: x.price)
        
        return flight_options
    
    async def _generate_options_summary(
        self, 
        options: List[FlightOption], 
        context: FlightSearchContext
    ) -> str:
        """
        Generate a human-readable summary of flight options.
        
        Args:
            options: List of flight options
            context: Flight search context
            
        Returns:
            Summary text
        """
        self.logger.info("Generating flight options summary")
        
        if not options:
            return "No flight options found matching your criteria."
        
        # Prepare a specific prompt for generating a summary
        summary_prompt = (
            f"Summarize the following {len(options)} flight options from {context.origin} "
            f"to {context.destination} on {context.departure_date}. Highlight the best value, "
            f"fastest option, and any notable features or drawbacks. Be concise but informative."
        )
        
        # Prepare flight options in a format the model can understand
        options_text = "\n\n".join([
            f"Option {i+1}: {option.airline} - {option.formatted_price}\n"
            f"Departure: {option.departure_time} - Arrival: {option.arrival_time}\n"
            f"Duration: {option.formatted_duration} - Layovers: {option.layover_count}\n"
            f"Baggage: {option.baggage_allowance or 'Not specified'}\n"
            f"Refundable: {option.refundable} - Eco-friendly: {option.eco_friendly}"
            for i, option in enumerate(options[:5])  # Limit to top 5 options for brevity
        ])
        
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": summary_prompt},
            {"role": "system", "content": options_text},
        ]
        
        response = await self._call_model(messages)
        
        # Return the generated summary
        return response.get("content", "Flight options summary not available.")
    
    def _format_flight_option(self, option: FlightOption) -> Dict[str, Any]:
        """
        Format a flight option for display.
        
        Args:
            option: FlightOption object
            
        Returns:
            Formatted flight option dictionary
        """
        legs_formatted = []
        for leg in option.legs:
            legs_formatted.append({
                "airline": leg.airline,
                "flight_number": leg.flight_number,
                "departure": {
                    "airport": leg.departure_airport,
                    "time": leg.departure_time
                },
                "arrival": {
                    "airport": leg.arrival_airport,
                    "time": leg.arrival_time
                },
                "duration": f"{leg.duration_minutes // 60}h {leg.duration_minutes % 60}m"
            })
        
        return {
            "id": option.id,
            "price": {
                "amount": option.price,
                "currency": option.currency,
                "formatted": option.formatted_price
            },
            "cabin_class": option.cabin_class.value,
            "legs": legs_formatted,
            "layovers": option.layover_count,
            "duration": option.formatted_duration,
            "baggage": option.baggage_allowance,
            "refundable": option.refundable,
            "eco_friendly": option.eco_friendly,
            "amenities": option.amenities
        }
    
    def _get_latest_user_input(self, messages: List[Dict[str, Any]]) -> str:
        """
        Extract the latest user input from a list of messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Latest user input text
        """
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""
    
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