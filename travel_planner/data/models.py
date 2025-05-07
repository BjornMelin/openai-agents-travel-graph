"""
Data models for the travel planner system.

This module defines the core data structures used throughout the travel planning
process, including queries, user preferences, and the final travel plan.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TravelMode(str, Enum):
    """Travel modes for transportation."""

    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


class AccommodationType(str, Enum):
    """Types of accommodation."""

    HOTEL = "hotel"
    HOSTEL = "hostel"
    APARTMENT = "apartment"
    RESORT = "resort"
    VILLA = "villa"
    AIRBNB = "airbnb"
    BOUTIQUE = "boutique_hotel"
    GUESTHOUSE = "guesthouse"


class TransportationType(str, Enum):
    """Types of local transportation."""

    TAXI = "taxi"
    RIDESHARE = "rideshare"
    RENTAL_CAR = "rental_car"
    PUBLIC_TRANSIT = "public_transit"
    WALKING = "walking"
    BICYCLE = "bicycle"
    FERRY = "ferry"
    TRAIN = "train"
    BUS = "bus"
    SHUTTLE = "shuttle"


class ActivityType(str, Enum):
    """Types of travel activities."""

    SIGHTSEEING = "sightseeing"
    CULTURAL = "cultural"
    ADVENTURE = "adventure"
    RELAXATION = "relaxation"
    CULINARY = "culinary"
    SHOPPING = "shopping"


class AccommodationSearchParams(BaseModel):
    """Parameters for accommodation search."""

    destination: str
    check_in_date: date
    check_out_date: date
    adults: int = 2
    children: int = 0
    rooms: int = 1
    accommodation_type: AccommodationType | None = None
    amenities: list[str] | None = None
    max_price: float | None = None
    min_rating: float | None = None
    max_results: int = 5
    sort_by: str = "popularity"


class FlightSearchParams(BaseModel):
    """Parameters for flight search."""

    origin: str
    destination: str
    departure_date: date
    return_date: date | None = None
    adults: int = 1
    children: int = 0
    travel_class: TravelMode = TravelMode.ECONOMY
    max_results: int = 5
    sort_by: str = "price"


class NodeFunctionParams(BaseModel):
    """Parameters for creating node functions."""

    agent_class: Any  # Type[BaseAgent], using Any to avoid circular imports
    task_name: str
    complete_stage: Any  # WorkflowStage
    result_field: str
    plan_field: str
    message_template: str


class AgentTaskParams(BaseModel):
    """Parameters for executing agent tasks."""

    state: Any  # TravelPlanningState
    agent: Any  # BaseAgent
    task_name: str
    complete_stage: Any  # WorkflowStage
    result_formatter: Any  # Callable[[dict[str, Any]], str]
    result_processor: Any = (
        None  # Callable[[TravelPlanningState, dict[str, Any]], None] | None
    )
    ENTERTAINMENT = "entertainment"
    NATURE = "nature"
    EDUCATIONAL = "educational"
    NIGHTLIFE = "nightlife"


class BudgetCategory(str, Enum):
    """Budget categories for expense tracking."""

    FLIGHTS = "flights"
    ACCOMMODATION = "accommodation"
    LOCAL_TRANSPORTATION = "local_transportation"
    ACTIVITIES = "activities"
    FOOD = "food"
    SHOPPING = "shopping"
    MISCELLANEOUS = "miscellaneous"


class TravelDestination(BaseModel):
    """Destination information."""

    name: str
    country: str
    region: str | None = None
    description: str | None = None
    timezone: str | None = None
    currency: str | None = None
    language: str | None = None
    attractions: list[str] = Field(default_factory=list)
    safety_info: str | None = None
    weather: dict[str, Any] | None = None
    best_times_to_visit: list[str] | None = None


class TravelQuery(BaseModel):
    """
    User's travel query with basic requirements.

    This captures the initial user request and the essential details needed to start
    the travel planning process.
    """

    raw_query: str
    destination: str | None = None
    origin: str | None = None
    departure_date: date | None = None
    return_date: date | None = None
    travelers: int = 1
    budget_range: dict[str, float] | None = None
    purpose: str | None = None
    requirements: dict[str, Any] | None = None

    @classmethod
    @field_validator("budget_range")
    def validate_budget_range(cls, v):
        """Validate that budget range has min and max values."""
        if v is not None:
            if "min" not in v or "max" not in v:
                raise ValueError("Budget range must include 'min' and 'max' values")
            if v["min"] > v["max"]:
                raise ValueError("Minimum budget cannot be greater than maximum budget")
        return v


class UserPreferences(BaseModel):
    """
    Detailed user preferences for travel planning.

    This contains the user's specific preferences for various aspects of their trip,
    which guides the specialized agents in making appropriate recommendations.
    """

    # Flight preferences
    preferred_airlines: list[str] = Field(default_factory=list)
    travel_class: TravelMode | None = None
    direct_flights_only: bool = False
    max_layover_time: int | None = None  # in minutes
    preferred_departure_times: dict[str, str] | None = None

    # Accommodation preferences
    accommodation_types: list[AccommodationType] = Field(default_factory=list)
    hotel_rating: int | None = None  # 1-5 stars
    amenities: list[str] = Field(default_factory=list)
    neighborhood_preferences: list[str] = Field(default_factory=list)

    # Transportation preferences
    transportation_modes: list[TransportationType] = Field(default_factory=list)
    public_transport_preference: bool = True
    car_rental_preference: bool = False

    # Activity preferences
    activity_types: list[ActivityType] = Field(default_factory=list)
    pace_preference: str | None = None  # relaxed, moderate, busy
    cultural_interests: list[str] = Field(default_factory=list)
    cuisine_preferences: list[str] = Field(default_factory=list)
    special_interests: list[str] = Field(default_factory=list)

    # Accessibility needs
    accessibility_requirements: list[str] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)

    # Budget allocation preferences (percentages)
    budget_allocation: dict[BudgetCategory, float] | None = None


class Flight(BaseModel):
    """Flight option details."""

    airline: str
    flight_number: str
    departure_airport: str
    arrival_airport: str
    departure_time: datetime
    arrival_time: datetime
    price: float
    currency: str = "USD"
    travel_class: TravelMode
    layovers: list[dict[str, Any]] = Field(default_factory=list)
    duration_minutes: int
    booking_link: str | None = None
    refundable: bool = False
    baggage_allowance: dict[str, Any] | None = None


class Accommodation(BaseModel):
    """Accommodation option details."""

    name: str
    type: AccommodationType
    location: str
    address: str
    rating: float | None = None  # e.g., 4.5 out of 5
    price_per_night: float
    currency: str = "USD"
    total_price: float
    check_in_time: str
    check_out_time: str
    amenities: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    booking_link: str | None = None
    cancellation_policy: str | None = None
    highlights: list[str] = Field(default_factory=list)


class TransportationOption(BaseModel):
    """Local transportation option details."""

    type: TransportationType
    description: str
    cost: float | None = None
    currency: str = "USD"
    route: str | None = None
    duration_minutes: int | None = None
    frequency: str | None = None
    booking_required: bool = False
    booking_link: str | None = None


class Activity(BaseModel):
    """Activity or attraction details."""

    name: str
    type: ActivityType
    description: str
    location: str
    duration_minutes: int
    cost: float | None = None
    currency: str = "USD"
    booking_required: bool = False
    booking_link: str | None = None
    recommended_time: str | None = None
    highlights: list[str] = Field(default_factory=list)


class DailyItinerary(BaseModel):
    """Daily itinerary with activities and schedule."""

    date: date
    day_number: int
    activities: list[Activity] = Field(default_factory=list)
    transportation: list[TransportationOption] = Field(default_factory=list)
    meals: list[dict[str, Any]] = Field(default_factory=list)
    notes: str | None = None
    weather_forecast: dict[str, Any] | None = None


class BudgetItem(BaseModel):
    """Budget item for tracking expenses."""

    category: BudgetCategory
    description: str
    amount: float
    currency: str = "USD"
    is_estimated: bool = True
    notes: str | None = None


class BudgetSummary(BaseModel):
    """Budget summary with category breakdowns."""

    total_budget: float
    currency: str = "USD"
    spent: float = 0
    remaining: float = 0
    breakdown: dict[BudgetCategory, float] = Field(default_factory=dict)
    items: list[BudgetItem] = Field(default_factory=list)
    notes: str | None = None
    saving_recommendations: list[str] = Field(default_factory=list)


class TravelPlan(BaseModel):
    """
    Comprehensive travel plan with all details.

    This is the main output of the travel planning process, containing all the
    information needed for the trip.
    """

    destination: dict[str, Any] | None = None
    flights: list[Flight] = Field(default_factory=list)
    accommodation: list[Accommodation] = Field(default_factory=list)
    transportation: dict[str, TransportationOption] = Field(default_factory=dict)
    activities: dict[str, DailyItinerary] = Field(default_factory=dict)
    budget: BudgetSummary | None = None
    overview: str | None = None
    recommendations: list[str] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
