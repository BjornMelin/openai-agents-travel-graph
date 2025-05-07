"""
Activity Planning Agent for the travel planner system.

This module implements the specialized agent responsible for researching,
scheduling, and recommending activities and attractions for the travel itinerary.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any

from travel_planner.agents.base import AgentConfig, AgentContext, BaseAgent
from travel_planner.utils.error_handling import with_retry
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)

# Constants
MAX_ACTIVITIES_PER_DAY = 3


class ActivityType(str, Enum):
    """Types of activities."""

    ATTRACTION = "attraction"
    TOUR = "tour"
    MUSEUM = "museum"
    OUTDOOR = "outdoor"
    FOOD = "food_and_drink"
    ENTERTAINMENT = "entertainment"
    SHOPPING = "shopping"
    CULTURAL = "cultural"
    ADVENTURE = "adventure"
    WELLNESS = "wellness"


class WeatherCondition(str, Enum):
    """Types of weather conditions."""

    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    SNOWY = "snowy"
    WINDY = "windy"
    HOT = "hot"
    COLD = "cold"


@dataclass
class Activity:
    """A single activity or attraction."""

    id: str
    name: str
    type: ActivityType
    location: str
    description: str
    price: float
    currency: str
    duration_minutes: int
    opening_hours: dict[str, dict[str, time]] = field(default_factory=dict)
    booking_required: bool = False
    booking_url: str = ""
    weather_dependent: bool = False
    suitable_weather: list[WeatherCondition] = field(default_factory=list)
    rating: float | None = None
    reviews_count: int = 0
    images: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    accessibility_features: list[str] = field(default_factory=list)

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
class ScheduledActivity:
    """An activity scheduled for a specific date and time."""

    activity: Activity
    date: str
    start_time: str
    end_time: str
    notes: str = ""


@dataclass
class DailyItinerary:
    """Itinerary for a single day."""

    date: str
    activities: list[ScheduledActivity] = field(default_factory=list)
    weather_forecast: dict[str, Any] | None = None
    notes: str = ""
    total_cost: float = 0.0
    currency: str = "EUR"


@dataclass
class ActivityPlanningContext(AgentContext):
    """Context for the activity planning agent."""

    destination: str = ""
    start_date: str | None = None
    end_date: str | None = None
    traveler_count: int = 1
    interests: list[str] = field(default_factory=list)
    has_children: bool = False
    has_accessibility_needs: bool = False
    budget_per_day: float | None = None
    accommodation_location: str = ""
    available_activities: list[Activity] = field(default_factory=list)
    daily_itineraries: dict[str, DailyItinerary] = field(default_factory=dict)
    weather_forecasts: dict[str, dict[str, Any]] = field(default_factory=dict)
    search_params: dict[str, Any] = field(default_factory=dict)
    excluded_activity_types: list[ActivityType] = field(default_factory=list)


class ActivityPlanningAgent(BaseAgent[ActivityPlanningContext]):
    """
    Specialized agent for activity planning.

    This agent is responsible for:
    1. Researching activities and attractions at the destination
    2. Creating daily itineraries with scheduled activities
    3. Considering factors like opening hours, travel time, and weather
    4. Balancing must-see attractions with personalized experiences
    5. Ensuring activities are within budget constraints
    """

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the activity planning agent.

        Args:
            config: Configuration for the agent (optional)
        """
        default_config = AgentConfig(
            name="activity_planning_agent",
            instructions="""
            You are a specialized activity planning agent.
            Your goal is to research, recommend, and schedule activities and attractions
            that match the traveler's interests and preferences. Create logical daily
            itineraries that account for location, opening hours, travel time, 
            and budget.
            """,
            model="gpt-4o",
            tools=[],  # No tools initially, they would be added in a real implementation
        )

        config = config or default_config
        super().__init__(config, ActivityPlanningContext)

        # Add tools for specific activity planning functionality
        # These would typically be implemented as part of the full system
        # self.add_tool(search_activities)
        # self.add_tool(check_opening_hours)
        # self.add_tool(get_weather_forecast)
        # self.add_tool(calculate_travel_time)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: ActivityPlanningContext | None = None,
    ) -> Any:
        """
        Run the activity planning agent with the provided input and context.

        Args:
            input_data: User input or conversation history
            context: Optional activity planning context

        Returns:
            Updated context and activity planning results
        """
        try:
            # Initialize context if not provided
            if not context:
                context = ActivityPlanningContext()

            # Process the input
            result = await self.process(input_data, context)
            return result
        except Exception as e:
            error_msg = f"Error in activity planning agent: {e!s}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def process(
        self, input_data: str | list[dict[str, Any]], context: ActivityPlanningContext
    ) -> dict[str, Any]:
        """
        Process the activity planning request.

        Args:
            input_data: User input or conversation history
            context: Activity planning context

        Returns:
            Activity planning results
        """
        # Extract activity preferences if not already set
        if not context.search_params:
            await self._extract_activity_preferences(input_data, context)

        # Get available activities at the destination
        if not context.available_activities:
            context.available_activities = await self._research_activities(context)

        # Get weather forecasts for the trip dates
        if not context.weather_forecasts and context.start_date and context.end_date:
            context.weather_forecasts = await self._get_weather_forecasts(context)

        # Create daily itineraries
        daily_itineraries = await self._create_daily_itineraries(context)
        context.daily_itineraries = daily_itineraries

        # Generate an itinerary summary
        itinerary_summary = await self._generate_itinerary_summary(context)

        return {
            "context": context,
            "daily_itineraries": daily_itineraries,
            "summary": itinerary_summary,
        }

    async def _extract_activity_preferences(
        self, input_data: str | list[dict[str, Any]], context: ActivityPlanningContext
    ) -> None:
        """
        Extract activity preferences from user input.

        Args:
            input_data: User input or conversation history
            context: Activity planning context
        """
        # Prepare a specific prompt for preference extraction
        extraction_prompt = (
            "Please extract activity preferences from the following user input. "
            "Include destination, trip dates, traveler details, interests, "
            "budget constraints, and any excluded activity types. "
            "Format your response as a JSON object.\n\n"
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

        # But for now we'll skip the actual API call
        # and update the context with the extracted preferences

        # For now, we'll set some example values for demonstration
        context.destination = "Paris, France"
        context.start_date = "2025-06-15"
        context.end_date = "2025-06-22"
        context.traveler_count = 2
        context.interests = ["art", "history", "food", "architecture", "local culture"]
        context.has_children = False
        context.has_accessibility_needs = False
        context.budget_per_day = 100.0
        context.accommodation_location = "15 Rue de Rivoli, 75001 Paris, France"
        context.excluded_activity_types = [ActivityType.ADVENTURE]

        # Store the parameters in the search_params dictionary
        context.search_params = {
            "destination": context.destination,
            "start_date": context.start_date,
            "end_date": context.end_date,
            "traveler_count": context.traveler_count,
            "interests": context.interests,
            "has_children": context.has_children,
            "has_accessibility_needs": context.has_accessibility_needs,
            "budget_per_day": context.budget_per_day,
            "accommodation_location": context.accommodation_location,
            "excluded_activity_types": [
                t.value for t in context.excluded_activity_types
            ],
        }

    async def _research_activities(
        self, context: ActivityPlanningContext
    ) -> list[Activity]:
        """
        Research activities at the destination based on context preferences.

        Args:
            context: Activity planning context

        Returns:
            List of activities
        """
        # In a real implementation, this would call activity search APIs
        # For demonstration, we'll create some mock activities

        mock_activities = [
            Activity(
                id="act1",
                name="Louvre Museum",
                type=ActivityType.MUSEUM,
                location="Rue de Rivoli, 75001 Paris, France",
                description="World's largest art museum and historic monument "
                "in Paris.",
                price=17.0,
                currency="EUR",
                duration_minutes=180,
                opening_hours={
                    "Monday": {"open": time(9, 0), "close": time(18, 0)},
                    "Tuesday": {"open": time(9, 0), "close": time(18, 0)},
                    "Wednesday": {"open": time(9, 0), "close": time(18, 0)},
                    "Thursday": {"open": time(9, 0), "close": time(18, 0)},
                    "Friday": {"open": time(9, 0), "close": time(21, 0)},
                    "Saturday": {"open": time(9, 0), "close": time(18, 0)},
                    "Sunday": {"open": time(9, 0), "close": time(18, 0)},
                },
                booking_required=True,
                booking_url="https://www.louvre.fr/en/visit",
                weather_dependent=False,
                rating=4.8,
                reviews_count=140000,
                tags=["art", "history", "culture"],
                accessibility_features=["wheelchair_accessible", "elevators"],
            ),
            Activity(
                id="act2",
                name="Eiffel Tower",
                type=ActivityType.ATTRACTION,
                location="Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
                description="Iconic wrought-iron lattice tower on the Champ de Mars "
                "in Paris.",
                price=26.8,
                currency="EUR",
                duration_minutes=120,
                opening_hours={
                    "Monday": {"open": time(9, 30), "close": time(23, 45)},
                    "Tuesday": {"open": time(9, 30), "close": time(23, 45)},
                    "Wednesday": {"open": time(9, 30), "close": time(23, 45)},
                    "Thursday": {"open": time(9, 30), "close": time(23, 45)},
                    "Friday": {"open": time(9, 30), "close": time(23, 45)},
                    "Saturday": {"open": time(9, 30), "close": time(23, 45)},
                    "Sunday": {"open": time(9, 30), "close": time(23, 45)},
                },
                booking_required=True,
                booking_url="https://www.toureiffel.paris/en",
                weather_dependent=True,
                suitable_weather=[WeatherCondition.SUNNY, WeatherCondition.CLOUDY],
                rating=4.6,
                reviews_count=220000,
                tags=["landmark", "views", "architecture"],
                accessibility_features=["elevator"],
            ),
            Activity(
                id="act3",
                name="Seine River Cruise",
                type=ActivityType.TOUR,
                location="Pont de l'Alma, 75008 Paris, France",
                description="Sightseeing cruise along the Seine River with views "
                "of Paris landmarks.",
                price=15.0,
                currency="EUR",
                duration_minutes=60,
                opening_hours={
                    "Monday": {"open": time(10, 0), "close": time(22, 0)},
                    "Tuesday": {"open": time(10, 0), "close": time(22, 0)},
                    "Wednesday": {"open": time(10, 0), "close": time(22, 0)},
                    "Thursday": {"open": time(10, 0), "close": time(22, 0)},
                    "Friday": {"open": time(10, 0), "close": time(22, 0)},
                    "Saturday": {"open": time(10, 0), "close": time(22, 0)},
                    "Sunday": {"open": time(10, 0), "close": time(22, 0)},
                },
                weather_dependent=True,
                suitable_weather=[WeatherCondition.SUNNY, WeatherCondition.CLOUDY],
                rating=4.5,
                reviews_count=75000,
                tags=["sightseeing", "views", "relaxing"],
                accessibility_features=["wheelchair_accessible"],
            ),
            Activity(
                id="act4",
                name="Montmartre Walking Tour",
                type=ActivityType.TOUR,
                location="Place du Tertre, 75018 Paris, France",
                description="Guided walking tour of the charming Montmartre district.",
                price=25.0,
                currency="EUR",
                duration_minutes=150,
                opening_hours={
                    "Monday": {"open": time(10, 0), "close": time(17, 0)},
                    "Tuesday": {"open": time(10, 0), "close": time(17, 0)},
                    "Wednesday": {"open": time(10, 0), "close": time(17, 0)},
                    "Thursday": {"open": time(10, 0), "close": time(17, 0)},
                    "Friday": {"open": time(10, 0), "close": time(17, 0)},
                    "Saturday": {"open": time(10, 0), "close": time(17, 0)},
                    "Sunday": {"open": time(10, 0), "close": time(17, 0)},
                },
                booking_required=True,
                weather_dependent=True,
                suitable_weather=[WeatherCondition.SUNNY, WeatherCondition.CLOUDY],
                rating=4.7,
                reviews_count=5000,
                tags=["walking", "art", "history", "local culture"],
            ),
            Activity(
                id="act5",
                name="Parisian Food Tour",
                type=ActivityType.FOOD,
                location="Saint-Germain-des-Prés, 75006 Paris, France",
                description="Culinary walking tour featuring tastings at "
                "various shops and cafes.",
                price=95.0,
                currency="EUR",
                duration_minutes=210,
                opening_hours={
                    "Monday": {"open": time(10, 30), "close": time(14, 30)},
                    "Tuesday": {"open": time(10, 30), "close": time(14, 30)},
                    "Wednesday": {"open": time(10, 30), "close": time(14, 30)},
                    "Thursday": {"open": time(10, 30), "close": time(14, 30)},
                    "Friday": {"open": time(10, 30), "close": time(14, 30)},
                    "Saturday": {"open": time(10, 30), "close": time(14, 30)},
                    "Sunday": {"open": time(10, 30), "close": time(14, 30)},
                },
                booking_required=True,
                booking_url="https://www.parisfoodtours.com",
                rating=4.9,
                reviews_count=3200,
                tags=["food", "walking", "local culture"],
            ),
            Activity(
                id="act6",
                name="Musée d'Orsay",
                type=ActivityType.MUSEUM,
                location="1 Rue de la Légion d'Honneur, 75007 Paris, France",
                description="Museum housed in the former Orsay railway station, "
                "featuring art from 1848 to 1914.",
                price=16.0,
                currency="EUR",
                duration_minutes=150,
                opening_hours={
                    "Monday": {"open": time(0, 0), "close": time(0, 0)},  # Closed
                    "Tuesday": {"open": time(9, 30), "close": time(18, 0)},
                    "Wednesday": {"open": time(9, 30), "close": time(18, 0)},
                    "Thursday": {"open": time(9, 30), "close": time(21, 45)},
                    "Friday": {"open": time(9, 30), "close": time(18, 0)},
                    "Saturday": {"open": time(9, 30), "close": time(18, 0)},
                    "Sunday": {"open": time(9, 30), "close": time(18, 0)},
                },
                booking_required=True,
                booking_url="https://www.musee-orsay.fr/en",
                weather_dependent=False,
                rating=4.7,
                reviews_count=65000,
                tags=["art", "history", "culture"],
                accessibility_features=["wheelchair_accessible", "elevators"],
            ),
            Activity(
                id="act7",
                name="Luxembourg Gardens",
                type=ActivityType.OUTDOOR,
                location="15 Rue de Vaugirard, 75006 Paris, France",
                description="Beautiful public park in the 6th arrondissement of Paris.",
                price=0.0,
                currency="EUR",
                duration_minutes=90,
                opening_hours={
                    "Monday": {"open": time(7, 30), "close": time(21, 30)},
                    "Tuesday": {"open": time(7, 30), "close": time(21, 30)},
                    "Wednesday": {"open": time(7, 30), "close": time(21, 30)},
                    "Thursday": {"open": time(7, 30), "close": time(21, 30)},
                    "Friday": {"open": time(7, 30), "close": time(21, 30)},
                    "Saturday": {"open": time(7, 30), "close": time(21, 30)},
                    "Sunday": {"open": time(7, 30), "close": time(21, 30)},
                },
                weather_dependent=True,
                suitable_weather=[WeatherCondition.SUNNY, WeatherCondition.CLOUDY],
                rating=4.8,
                reviews_count=35000,
                tags=["outdoors", "relaxing", "scenic"],
            ),
        ]

        # Filter out excluded activity types and filter by interests
        filtered_activities = []
        for activity in mock_activities:
            if activity.type not in context.excluded_activity_types:
                if not context.interests or any(
                    tag in context.interests for tag in activity.tags
                ):
                    filtered_activities.append(activity)

        # If no activities match the interests, return all non-excluded activities
        if not filtered_activities:
            filtered_activities = [
                act
                for act in mock_activities
                if act.type not in context.excluded_activity_types
            ]

        return filtered_activities

    async def _get_weather_forecasts(
        self, context: ActivityPlanningContext
    ) -> dict[str, dict[str, Any]]:
        """
        Get weather forecasts for the destination during the trip dates.

        Args:
            context: Activity planning context

        Returns:
            Dictionary mapping dates to weather forecasts
        """
        # In a real implementation, this would call a weather API
        # For demonstration, we'll create some mock forecasts

        forecasts = {}

        if not context.start_date or not context.end_date:
            return forecasts

        start = datetime.strptime(context.start_date, "%Y-%m-%d")
        end = datetime.strptime(context.end_date, "%Y-%m-%d")
        current_date = start

        # Generate forecasts for each day
        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")

            # Create a mock forecast (in a real implementation, this would come from an API)
            # We'll alternate between sunny and cloudy for simplicity
            is_sunny = (current_date - start).days % 2 == 0

            forecasts[date_str] = {
                "condition": WeatherCondition.SUNNY
                if is_sunny
                else WeatherCondition.CLOUDY,
                "temperature_celsius": 22 if is_sunny else 19,
                "temperature_fahrenheit": 72 if is_sunny else 66,
                "precipitation_chance": 10 if is_sunny else 40,
                "wind_speed_kmh": 10 if is_sunny else 15,
            }

            current_date += timedelta(days=1)

        return forecasts

    async def _create_daily_itineraries(
        self, context: ActivityPlanningContext
    ) -> dict[str, DailyItinerary]:
        """
        Create daily itineraries for the trip dates.

        Args:
            context: Activity planning context

        Returns:
            Dictionary mapping dates to daily itineraries
        """
        if (
            not context.start_date
            or not context.end_date
            or not context.available_activities
        ):
            return {}

        itineraries = {}

        start = datetime.strptime(context.start_date, "%Y-%m-%d")
        end = datetime.strptime(context.end_date, "%Y-%m-%d")
        current_date = start

        # Create an itinerary for each day
        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            day_of_week = current_date.strftime("%A")

            # Get weather forecast if available
            weather = context.weather_forecasts.get(date_str)

            # Create a new daily itinerary
            itinerary = DailyItinerary(
                date=date_str, weather_forecast=weather, currency="EUR"
            )

            # Filter activities based on day of week (opening hours) and weather
            suitable_activities = []
            for activity in context.available_activities:
                # Check if the activity is open on this day
                opening_hours = activity.opening_hours.get(day_of_week, {})
                if opening_hours and (
                    opening_hours.get("open") != time(0, 0)
                    or opening_hours.get("close") != time(0, 0)
                ):
                    # Check if the activity is suitable for the weather
                    if not activity.weather_dependent or not weather:
                        suitable_activities.append(activity)
                    elif (
                        weather
                        and weather.get("condition") in activity.suitable_weather
                    ):
                        suitable_activities.append(activity)

            # For a real implementation, we would create a logical daily schedule based on:
            # - Location proximity (to minimize travel time)
            # - Opening hours
            # - Activity durations
            # - Budget constraints

            # For this demo, we'll create a simple schedule with morning, afternoon, and evening activities

            # Morning activity (9:00 - 12:00)
            morning_activities = [
                a
                for a in suitable_activities
                if a.type
                in [ActivityType.MUSEUM, ActivityType.ATTRACTION, ActivityType.TOUR]
            ]
            if (
                morning_activities
                and len(itinerary.activities) < MAX_ACTIVITIES_PER_DAY
            ):
                selected = morning_activities[
                    0
                ]  # In a real implementation, we would make a smarter selection
                suitable_activities.remove(selected)

                itinerary.activities.append(
                    ScheduledActivity(
                        activity=selected,
                        date=date_str,
                        start_time="09:00",
                        end_time="12:00",
                        notes="Visit in the morning to avoid crowds",
                    )
                )

                itinerary.total_cost += selected.price

            # Lunch break

            # Afternoon activity (14:00 - 17:00)
            afternoon_activities = [
                a
                for a in suitable_activities
                if a.type
                in [ActivityType.OUTDOOR, ActivityType.CULTURAL, ActivityType.SHOPPING]
            ]
            if (
                afternoon_activities
                and len(itinerary.activities) < MAX_ACTIVITIES_PER_DAY
            ):
                selected = afternoon_activities[
                    0
                ]  # In a real implementation, we would make a smarter selection
                suitable_activities.remove(selected)

                itinerary.activities.append(
                    ScheduledActivity(
                        activity=selected,
                        date=date_str,
                        start_time="14:00",
                        end_time="17:00",
                        notes="Afternoon exploration",
                    )
                )

                itinerary.total_cost += selected.price

            # Evening activity (19:00 - 21:00)
            evening_activities = [
                a
                for a in suitable_activities
                if a.type in [ActivityType.FOOD, ActivityType.ENTERTAINMENT]
            ]
            if (
                evening_activities
                and len(itinerary.activities) < MAX_ACTIVITIES_PER_DAY
            ):
                # In a real implementation, we would make a smarter selection
                selected = evening_activities[0]

                itinerary.activities.append(
                    ScheduledActivity(
                        activity=selected,
                        date=date_str,
                        start_time="19:00",
                        end_time="21:00",
                        notes="Evening entertainment",
                    )
                )

                itinerary.total_cost += selected.price

            # Check budget constraints
            if context.budget_per_day and itinerary.total_cost > context.budget_per_day:
                # In a real implementation, we would adjust the itinerary to fit the budget
                itinerary.notes += (
                    f"\nNote: This day's activities exceed your "
                    f"daily budget of {context.budget_per_day} EUR."
                )

            # Add weather note
            if weather:
                itinerary.notes += (
                    f"\nWeather forecast: {weather.get('condition')}, "
                    f"{weather.get('temperature_celsius')}°C "
                    f"({weather.get('temperature_fahrenheit')}°F)"
                )

            # Add the itinerary to the dictionary
            itineraries[date_str] = itinerary

            # Move to the next day
            current_date += timedelta(days=1)

        return itineraries

    async def _generate_itinerary_summary(
        self, context: ActivityPlanningContext
    ) -> str:
        """
        Generate a comprehensive summary of the trip itinerary.

        Args:
            context: Activity planning context

        Returns:
            Itinerary summary text
        """
        # Prepare a specific prompt for generating a summary
        summary_prompt = (
            "Create a detailed summary of this trip itinerary to {destination} "
            "from {start_date} to {end_date} for {traveler_count} traveler(s).\n\n"
            "Please include:\n"
            "1. An overview of the activities scheduled\n"
            "2. Highlights not to be missed\n"
            "3. Budget breakdown and total cost\n"
            "4. Weather considerations\n"
            "5. Practical tips for the activities\n\n"
            "{itinerary_details}"
        )

        # Prepare itinerary details for the prompt
        itinerary_details = []
        total_cost = 0.0

        for date_str, itinerary in context.daily_itineraries.items():
            day_details = [f"Date: {date_str}"]

            if itinerary.weather_forecast:
                weather = itinerary.weather_forecast
                day_details.append(
                    f"Weather: {weather.get('condition')}, "
                    f"{weather.get('temperature_celsius')}°C"
                )

            for i, scheduled in enumerate(itinerary.activities, 1):
                activity = scheduled.activity
                day_details.append(
                    f"{i}. {activity.name} "
                    f"({scheduled.start_time} - {scheduled.end_time})\n"
                    f"   Type: {activity.type.value}, "
                    f"Price: {activity.formatted_price}\n"
                    f"   {activity.description[:100]}..."
                )

            day_details.append(f"Daily Cost: {itinerary.total_cost:.2f} EUR\n")
            total_cost += itinerary.total_cost

            itinerary_details.append("\n".join(day_details))

        messages = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": summary_prompt.format(
                    destination=context.destination,
                    start_date=context.start_date,
                    end_date=context.end_date,
                    traveler_count=context.traveler_count,
                    itinerary_details="\n\n".join(itinerary_details),
                ),
            },
        ]

        response = await self._call_model(messages)

        # Return the generated summary
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
