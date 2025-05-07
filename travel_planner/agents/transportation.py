"""
Transportation Agent for the travel planner system.

This module implements the specialized agent responsible for planning
local transportation, researching rental options, and optimizing
travel routes for the travel itinerary.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from travel_planner.agents.base import AgentConfig, AgentContext, BaseAgent
from travel_planner.utils.error_handling import with_retry
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


class TransportationType(str, Enum):
    """Types of transportation."""

    RENTAL_CAR = "rental_car"
    PUBLIC_TRANSIT = "public_transit"
    TAXI = "taxi"
    RIDESHARE = "rideshare"
    SHUTTLE = "shuttle"
    FERRY = "ferry"
    TRAIN = "train"
    BUS = "bus"
    WALKING = "walking"
    BICYCLE = "bicycle"


@dataclass
class TransportationOption:
    """A single transportation option."""

    id: str
    type: TransportationType
    name: str
    price: float
    currency: str
    duration_minutes: int
    start_location: str
    end_location: str
    provider: str = ""
    details: str = ""
    booking_url: str = ""
    schedule: dict[str, Any] | None = None
    amenities: list[str] = field(default_factory=list)
    accessibility: list[str] = field(default_factory=list)
    eco_friendly: bool = False

    @property
    def formatted_price(self) -> str:
        """Get the formatted price with currency symbol."""
        if self.currency == "USD":
            return f"${self.price:.2f}"
        elif self.currency == "EUR":
            return f"€{self.price:.2f}"
        else:
            return f"{self.price:.2f} {self.currency}"

    @property
    def formatted_duration(self) -> str:
        """Get the formatted duration as hours and minutes."""
        hours, minutes = divmod(self.duration_minutes, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"


@dataclass
class TransportationContext(AgentContext):
    """Context for the transportation agent."""

    destination: str = ""
    accommodation_location: str = ""
    start_date: str | None = None
    end_date: str | None = None
    traveler_count: int = 1
    preferred_transportation_types: list[TransportationType] = field(
        default_factory=list
    )
    has_children: bool = False
    has_accessibility_needs: bool = False
    max_budget: float | None = None
    transportation_options: dict[str, list[TransportationOption]] = field(
        default_factory=dict
    )
    selected_options: dict[str, TransportationOption] = field(default_factory=dict)
    search_params: dict[str, Any] = field(default_factory=dict)
    points_of_interest: list[str] = field(default_factory=list)


class TransportationAgent(BaseAgent[TransportationContext]):
    """
    Specialized agent for transportation planning.

    This agent is responsible for:
    1. Planning airport transfers and local transportation
    2. Researching rental car options
    3. Identifying public transit and transportation options
    4. Optimizing transportation for convenience and cost
    5. Considering special needs and requirements
    """

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the transportation agent.

        Args:
            config: Configuration for the agent (optional)
        """
        default_config = AgentConfig(
            name="transportation_agent",
            instructions="""
            You are a specialized transportation planner agent.
            Your goal is to research, compare, and recommend the best transportation
            options for travelers at their destination. Consider factors like
            convenience, cost, travel time, and special requirements.
            """,
            model="gpt-4o",
            tools=[],  # No tools initially, they would be added in a real implementation
        )

        config = config or default_config
        super().__init__(config, TransportationContext)

        # Add tools for specific transportation search functionality
        # These would typically be implemented as part of the full system
        # self.add_tool(search_rental_cars)
        # self.add_tool(find_public_transit)
        # self.add_tool(calculate_transfer_options)
        # self.add_tool(optimize_route)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: TransportationContext | None = None,
    ) -> Any:
        """
        Run the transportation agent with the provided input and context.

        Args:
            input_data: User input or conversation history
            context: Optional transportation context

        Returns:
            Updated context and transportation planning results
        """
        try:
            # Initialize context if not provided
            if not context:
                context = TransportationContext()

            # Process the input
            result = await self.process(input_data, context)
            return result
        except Exception as e:
            error_msg = f"Error in transportation agent: {e!s}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def process(
        self,
        input_data: str | list[dict[str, Any]],
        context: TransportationContext,
    ) -> dict[str, Any]:
        """
        Process the transportation planning request.

        Args:
            input_data: User input or conversation history
            context: Transportation context

        Returns:
            Transportation planning results
        """
        # Extract transportation requirements if not already set
        if not context.search_params:
            await self._extract_transportation_requirements(input_data, context)

        # Plan airport transfers
        airport_transfers = await self._plan_airport_transfers(context)
        context.transportation_options["airport_transfers"] = airport_transfers

        # Research local transportation options
        local_transportation = await self._research_local_transportation(context)
        context.transportation_options["local"] = local_transportation

        # If points of interest are provided, plan specific routes
        if context.points_of_interest:
            poi_transportation = await self._plan_poi_transportation(context)
            context.transportation_options["points_of_interest"] = poi_transportation

        # Generate a transportation plan summary
        plan_summary = await self._generate_transportation_plan(context)

        return {
            "context": context,
            "transportation_options": context.transportation_options,
            "selected_options": context.selected_options,
            "plan_summary": plan_summary,
        }

    async def _extract_transportation_requirements(
        self,
        input_data: str | list[dict[str, Any]],
        context: TransportationContext,
    ) -> None:
        """
        Extract transportation requirements from user input.

        Args:
            input_data: User input or conversation history
            context: Transportation context
        """
        # Prepare a specific prompt for requirement extraction
        extraction_prompt = (
            "Please extract transportation planning requirements from the following user input. "
            "Include destination, accommodation location, preferred transportation types, "
            "traveler count, whether there are children or accessibility needs, "
            "and any budget constraints. Format your response as a JSON object.\n\n"
            "User input: {input}"
        )

        user_input = (
            input_data
            if isinstance(input_data, str)
            else self._get_latest_user_input(input_data)
        )

        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": extraction_prompt.format(input=user_input)},
        ]

        # Add current parameters as context if they exist
        if context.search_params:
            messages.append(
                {
                    "role": "system",
                    "content": f"Current parameters: {context.search_params}",
                }
            )

        # In a real implementation, we would call the model and parse the JSON response
        # response = await self._call_model(messages)
        # and update the context with the extracted requirements

        # For now, we'll set some example values for demonstration
        context.destination = "Paris, France"
        context.accommodation_location = "15 Rue de Rivoli, 75001 Paris, France"
        context.start_date = "2025-06-15"
        context.end_date = "2025-06-22"
        context.traveler_count = 2
        context.preferred_transportation_types = [
            TransportationType.PUBLIC_TRANSIT,
            TransportationType.WALKING,
            TransportationType.TAXI,
        ]
        context.has_children = False
        context.has_accessibility_needs = False
        context.max_budget = 200.0
        context.points_of_interest = [
            "Eiffel Tower",
            "Louvre Museum",
            "Notre-Dame Cathedral",
            "Montmartre",
        ]

        # Store the parameters in the search_params dictionary
        context.search_params = {
            "destination": context.destination,
            "accommodation_location": context.accommodation_location,
            "start_date": context.start_date,
            "end_date": context.end_date,
            "traveler_count": context.traveler_count,
            "preferred_transportation_types": [
                t.value for t in context.preferred_transportation_types
            ],
            "has_children": context.has_children,
            "has_accessibility_needs": context.has_accessibility_needs,
            "max_budget": context.max_budget,
            "points_of_interest": context.points_of_interest,
        }

    async def _plan_airport_transfers(
        self, context: TransportationContext
    ) -> list[TransportationOption]:
        """
        Plan airport transfers based on the context parameters.

        Args:
            context: Transportation context

        Returns:
            List of airport transfer options
        """
        # In a real implementation, this would call transportation APIs
        # For demonstration, we'll create some mock transfer options

        mock_transfers = [
            TransportationOption(
                id="transfer1",
                type=TransportationType.SHUTTLE,
                name="Airport Shuttle Service",
                price=25.0,
                currency="EUR",
                duration_minutes=45,
                start_location="Charles de Gaulle Airport",
                end_location=context.accommodation_location,
                provider="Paris Airport Shuttle",
                details="Shared shuttle service, departs every 30 minutes",
                amenities=["wifi", "air_conditioning"],
            ),
            TransportationOption(
                id="transfer2",
                type=TransportationType.TAXI,
                name="Airport Taxi",
                price=65.0,
                currency="EUR",
                duration_minutes=35,
                start_location="Charles de Gaulle Airport",
                end_location=context.accommodation_location,
                provider="Paris Taxi Association",
                details="Direct taxi service, no waiting",
            ),
            TransportationOption(
                id="transfer3",
                type=TransportationType.TRAIN,
                name="RER B + Metro",
                price=10.5,
                currency="EUR",
                duration_minutes=60,
                start_location="Charles de Gaulle Airport",
                end_location="Near " + context.accommodation_location,
                provider="RATP",
                details="RER B to Châtelet-Les Halles, then Metro line 1",
                eco_friendly=True,
            ),
        ]

        # Sort options by preference (using preferred transportation types)
        ordered_transfers = sorted(
            mock_transfers,
            key=lambda x: (
                # Preferred type gets lowest value (highest priority)
                -1 if x.type in context.preferred_transportation_types else 0,
                # Then sort by price
                x.price,
            ),
        )

        return ordered_transfers

    async def _research_local_transportation(
        self, context: TransportationContext
    ) -> list[TransportationOption]:
        """
        Research local transportation options based on the context parameters.

        Args:
            context: Transportation context

        Returns:
            List of local transportation options
        """
        # In a real implementation, this would research local transportation
        # For demonstration, we'll create some mock local options

        mock_local_options = [
            TransportationOption(
                id="local1",
                type=TransportationType.PUBLIC_TRANSIT,
                name="Metro Pass (5 days)",
                price=38.35,
                currency="EUR",
                duration_minutes=0,  # Not applicable for passes
                start_location="Paris",
                end_location="Paris",
                provider="RATP",
                details="Unlimited metro, bus, and RER within Paris",
                eco_friendly=True,
            ),
            TransportationOption(
                id="local2",
                type=TransportationType.BICYCLE,
                name="Vélib' Bike Share (5 days)",
                price=25.0,
                currency="EUR",
                duration_minutes=0,  # Not applicable for rentals
                start_location="Paris",
                end_location="Paris",
                provider="Vélib' Métropole",
                details="Access to 20,000 bikes across 1,400 stations in Paris",
                eco_friendly=True,
            ),
            TransportationOption(
                id="local3",
                type=TransportationType.RENTAL_CAR,
                name="Economy Car Rental (7 days)",
                price=315.0,
                currency="EUR",
                duration_minutes=0,  # Not applicable for rentals
                start_location="Paris",
                end_location="Paris",
                provider="Europcar",
                details="Renault Clio or similar, unlimited mileage",
                amenities=["air_conditioning", "gps"],
            ),
        ]

        # Filter options based on user preferences
        filtered_options = [
            option
            for option in mock_local_options
            if option.type in context.preferred_transportation_types
            or not context.preferred_transportation_types
        ]

        # If no options match the preferred types, return all options
        if not filtered_options:
            return mock_local_options

        return filtered_options

    async def _plan_poi_transportation(
        self, context: TransportationContext
    ) -> dict[str, list[TransportationOption]]:
        """
        Plan transportation to points of interest.

        Args:
            context: Transportation context

        Returns:
            Dictionary mapping points of interest to transportation options
        """
        # In a real implementation, this would plan specific routes to POIs
        # For demonstration, we'll create some mock options for each POI

        poi_transportation = {}

        for poi in context.points_of_interest:
            # Create mock options for this POI
            options = []

            # Public transit option
            if (
                TransportationType.PUBLIC_TRANSIT
                in context.preferred_transportation_types
                or not context.preferred_transportation_types
            ):
                options.append(
                    TransportationOption(
                        id=f"{poi.lower().replace(' ', '_')}_transit",
                        type=TransportationType.PUBLIC_TRANSIT,
                        name=f"Metro to {poi}",
                        price=1.90,
                        currency="EUR",
                        duration_minutes=25,
                        start_location=context.accommodation_location,
                        end_location=poi,
                        provider="RATP",
                        details="Take Metro line and walk",
                        eco_friendly=True,
                    )
                )

            # Taxi option
            if (
                TransportationType.TAXI in context.preferred_transportation_types
                or not context.preferred_transportation_types
            ):
                options.append(
                    TransportationOption(
                        id=f"{poi.lower().replace(' ', '_')}_taxi",
                        type=TransportationType.TAXI,
                        name=f"Taxi to {poi}",
                        price=15.0,
                        currency="EUR",
                        duration_minutes=15,
                        start_location=context.accommodation_location,
                        end_location=poi,
                        provider="Paris Taxi",
                        details="Direct taxi service",
                    )
                )

            # Walking option for nearby POIs
            if (
                TransportationType.WALKING in context.preferred_transportation_types
                or not context.preferred_transportation_types
            ):
                options.append(
                    TransportationOption(
                        id=f"{poi.lower().replace(' ', '_')}_walking",
                        type=TransportationType.WALKING,
                        name=f"Walk to {poi}",
                        price=0.0,
                        currency="EUR",
                        duration_minutes=45,
                        start_location=context.accommodation_location,
                        end_location=poi,
                        details="Scenic walking route",
                        eco_friendly=True,
                    )
                )

            poi_transportation[poi] = options

        return poi_transportation

    async def _generate_transportation_plan(
        self, context: TransportationContext
    ) -> str:
        """
        Generate a comprehensive transportation plan.

        Args:
            context: Transportation context

        Returns:
            Transportation plan text
        """
        # Prepare a specific prompt for generating a plan
        plan_prompt = (
            "Create a detailed transportation plan for a trip to {destination} "
            "staying at {accommodation}. The trip is from {start_date} to {end_date} "
            "for {traveler_count} traveler(s).\n\n"
            "Please include:\n"
            "1. Airport transfer recommendations\n"
            "2. Best local transportation options\n"
            "3. Specific routes to these points of interest: {pois}\n"
            "4. Cost breakdown and budget analysis\n"
            "5. Tips for navigating local transportation\n\n"
            "Transportation preferences: {preferences}\n"
            "Max budget: {budget} EUR\n"
            "{accessibility}"
        )

        accessibility_note = ""
        if context.has_children:
            accessibility_note += "Note: Traveling with children. "
        if context.has_accessibility_needs:
            accessibility_note += "Note: Has accessibility requirements. "

        messages = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": plan_prompt.format(
                    destination=context.destination,
                    accommodation=context.accommodation_location,
                    start_date=context.start_date,
                    end_date=context.end_date,
                    traveler_count=context.traveler_count,
                    pois=", ".join(context.points_of_interest),
                    preferences=", ".join(
                        [t.value for t in context.preferred_transportation_types]
                    ),
                    budget=context.max_budget,
                    accessibility=accessibility_note,
                ),
            },
        ]

        response = await self._call_model(messages)

        # Return the generated plan
        return response.get("content", "")

    def _get_latest_user_input(self, messages: list[dict[str, Any]]) -> str:
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
    async def _call_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call the OpenAI API with the given messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Model response
        """
        # Log inputs for debugging
        logger.debug(f"Calling model with messages: {messages}")

        # Call OpenAI API
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Log the response
        logger.debug(f"Model response: {response}")

        # Extract the content from the response
        content = response.choices[0].message.content
        return {"content": content}
