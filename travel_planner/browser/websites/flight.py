"""
Flight search automation for travel planning.

This module implements browser automation for flight search websites using Stagehand,
providing a consistent interface for retrieving flight options from different providers.
"""

import asyncio
import os
from datetime import date, datetime
from typing import Any, ClassVar

from stagehand import PromptTemplate

from travel_planner.browser.automation import BrowserAutomation
from travel_planner.data.models import Flight, FlightSearchParams, TravelMode
from travel_planner.utils.error_handling import with_retry
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


class FlightSearchAutomation(BrowserAutomation):
    """Browser automation for flight search websites."""

    SEARCH_SITES: ClassVar[dict[str, dict[str, str | list[str]]]] = {
        "google_flights": {
            "url": "https://www.google.com/travel/flights",
            "supported_features": ["one_way", "round_trip", "multi_city"],
        },
        "skyscanner": {
            "url": "https://www.skyscanner.com",
            "supported_features": ["one_way", "round_trip", "multi_city"],
        },
        "kayak": {
            "url": "https://www.kayak.com",
            "supported_features": ["one_way", "round_trip", "multi_city"],
        },
    }

    def __init__(
        self,
        provider: str = "google_flights",
        headless: bool = True,
        slow_mo: int = 50,
        timeout: int = 60000,
        use_cache: bool = True,
    ):
        """
        Initialize flight search automation.

        Args:
            provider: Flight search provider (google_flights, skyscanner, kayak)
            headless: Whether to run the browser in headless mode
            slow_mo: Slow down operations by this many milliseconds
            timeout: Default timeout for operations in milliseconds
            use_cache: Whether to use action caching
        """
        super().__init__(
            headless=headless, slow_mo=slow_mo, timeout=timeout, use_cache=use_cache
        )

        if provider not in self.SEARCH_SITES:
            raise ValueError(
                f"Unsupported provider: {provider}. Available providers: {list(self.SEARCH_SITES.keys())}"
            )

        self.provider = provider
        self.base_url = self.SEARCH_SITES[provider]["url"]

    @with_retry(retries=2, delay=1, backoff=2)
    async def search_flights(self, params: FlightSearchParams) -> list[Flight]:
        """
        Search for flights with the specified parameters.

        Args:
            origin: Origin airport code or city name
            destination: Destination airport code or city name
            departure_date: Departure date
            return_date: Return date (for round trip)
            adults: Number of adult passengers
            children: Number of child passengers
            travel_class: Travel class
            max_results: Maximum number of results to return
            sort_by: Sort criteria (price, duration, departure, arrival)

        Returns:
            List of Flight objects with search results
        """
        await self.setup()

        try:
            # Navigate to the flight search provider
            await self.navigate(self.base_url)

            # Wait for the page to load
            await asyncio.sleep(2)

            # Fill out the search form
            await self._fill_search_form(params)

            # Submit the search and wait for results
            await self._submit_search()

            # Extract flight results
            flights = await self._extract_flight_results(
                params.max_results, params.sort_by
            )

            return flights
        except Exception as e:
            logger.error(f"Error during flight search: {e!s}")
            # Take a screenshot of the failure for debugging
            error_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "error_screenshots"
            )
            os.makedirs(error_dir, exist_ok=True)
            screenshot_path = os.path.join(
                error_dir,
                f"flight_search_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            await self.screenshot(screenshot_path)
            logger.info(f"Error screenshot saved to {screenshot_path}")
            raise
        finally:
            await self.teardown()

    async def _fill_search_form(self, params: FlightSearchParams):
        """
        Fill out the flight search form.

        This uses AI-assisted form filling for flexibility across different sites.

        Args:
            origin: Origin airport code or city name
            destination: Destination airport code or city name
            departure_date: Departure date
            return_date: Return date (for round trip)
            adults: Number of adult passengers
            children: Number of child passengers
            travel_class: Travel class
        """
        trip_type = "round trip" if params.return_date else "one way"

        # Use AI-based form filling for flexibility across different sites
        prompt = PromptTemplate("""
        Fill out the flight search form with the following details:
        
        - Trip Type: {{trip_type}}
        - Origin: {{origin}}
        - Destination: {{destination}}
        - Departure Date: {{departure_date}}
        {% if return_date %}
        - Return Date: {{return_date}}
        {% endif %}
        - Number of Adults: {{adults}}
        - Number of Children: {{children}}
        - Cabin Class: {{travel_class}}
        
        Please select appropriate fields, enter the information, and prepare the search.
        DO NOT click the search button yet.
        """)

        context = {
            "trip_type": trip_type,
            "origin": params.origin,
            "destination": params.destination,
            "departure_date": params.departure_date.strftime("%Y-%m-%d"),
            "return_date": params.return_date.strftime("%Y-%m-%d")
            if params.return_date
            else None,
            "adults": params.adults,
            "children": params.children,
            "travel_class": params.travel_class.value.replace("_", " ").title(),
        }

        logger.info(
            f"Filling flight search form for {params.origin} to {params.destination}"
        )
        await self.execute_ai_action(prompt, context)

        # Additional provider-specific handling
        if self.provider == "google_flights":
            # Google Flights might need special handling for dates
            await self._handle_google_flights_specifics(
                params.departure_date, params.return_date
            )
        elif self.provider == "skyscanner":
            # Skyscanner might need special handling
            await self._handle_skyscanner_specifics()

    async def _handle_google_flights_specifics(
        self, departure_date: date, return_date: date | None = None
    ):
        """
        Handle Google Flights specific form interactions.

        Args:
            departure_date: Departure date
            return_date: Return date (for round trip)
        """
        # Google Flights may have specific date picker behavior
        # This method handles any special cases
        pass

    async def _handle_skyscanner_specifics(self):
        """Handle Skyscanner specific form interactions."""
        # Skyscanner may have cookie consent or other specific elements
        try:
            # Check for cookie consent dialog
            consent_button = await self.page.query_selector(
                "button[id*='cookie-consent'], button[id*='accept-cookies']"
            )
            if consent_button:
                await consent_button.click()
        except Exception as e:
            logger.warning(f"Error handling Skyscanner specifics: {e!s}")

    async def _submit_search(self):
        """Submit the search form and wait for results to load."""
        # Use AI to find and click the search button
        prompt = PromptTemplate("""
        Find and click the search button to start the flight search.
        After clicking, wait for the search results to load completely.
        """)

        logger.info("Submitting flight search")
        await self.execute_ai_action(prompt, {})

        # Wait for results to load
        await self._wait_for_search_results()

    async def _wait_for_search_results(self, max_wait_time: int = 30):
        """
        Wait for search results to load.

        Args:
            max_wait_time: Maximum time to wait in seconds
        """
        logger.info("Waiting for flight search results to load")

        # Use AI to determine when results are fully loaded
        prompt = PromptTemplate("""
        Wait for the flight search results to load completely.
        Check for:
        - Search progress indicators or spinners to disappear
        - Flight results to appear
        - Sort options and filters to become interactive
        
        Once the page is fully loaded with results, proceed.
        """)

        # Set a reasonable timeout for results to load
        try:
            await asyncio.wait_for(
                self.execute_ai_action(prompt, {}), timeout=max_wait_time
            )
        except TimeoutError:
            logger.warning(f"Timeout waiting for results after {max_wait_time}s")
            # Take a screenshot for debugging
            screenshot_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "debug_screenshots",
                f"timeout_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            await self.screenshot(screenshot_path)

    async def _extract_flight_results(
        self, max_results: int = 5, sort_by: str = "price"
    ) -> list[Flight]:
        """
        Extract flight results from the page.

        Args:
            max_results: Maximum number of results to extract
            sort_by: Sort criteria (price, duration, departure, arrival)

        Returns:
            List of Flight objects
        """
        logger.info(
            f"Extracting up to {max_results} flight results sorted by {sort_by}"
        )

        # First, use AI to sort results if needed
        if sort_by != "default":
            sort_prompt = PromptTemplate("""
            Sort the flight results by {sort_by}.
            Find and click the appropriate sort option or filter.
            """)

            await self.execute_ai_action(sort_prompt, {"sort_by": sort_by})
            await asyncio.sleep(2)  # Wait for sorting to take effect

        # Use AI to extract flight information
        extract_prompt = PromptTemplate("""
        Extract information for the top {{max_results}} flights from the search results.
        For each flight, extract:
        - Airline name
        - Flight number(s)
        - Departure and arrival airports
        - Departure and arrival times
        - Flight duration
        - Number of stops/layovers
        - Price
        - Cabin class
        - Any additional fees or restrictions
        
        Return the information in a structured format.
        """)

        extraction_result = await self.extract_data(
            extraction_prompt=extract_prompt.format(max_results=max_results), context={}
        )

        # Convert the extraction result to Flight objects
        flights = self._parse_extraction_to_flights(extraction_result)

        logger.info(f"Extracted {len(flights)} flights")
        return flights

    def _parse_extraction_to_flights(
        self, extraction_result: dict[str, Any]
    ) -> list[Flight]:
        """
        Parse the extraction result into Flight objects.

        Args:
            extraction_result: Result from AI extraction

        Returns:
            List of Flight objects
        """
        flights = []

        # Extract the flight data from the extraction result
        flight_data = extraction_result.get("flights", [])
        if not flight_data and isinstance(extraction_result, list):
            flight_data = extraction_result

        for data in flight_data[:5]:  # Limit to top 5 results
            try:
                # Parse departure and arrival times
                dep_time = data.get("departure_time")
                arr_time = data.get("arrival_time")

                if isinstance(dep_time, str):
                    dep_time = datetime.fromisoformat(dep_time.replace("Z", "+00:00"))
                if isinstance(arr_time, str):
                    arr_time = datetime.fromisoformat(arr_time.replace("Z", "+00:00"))

                # Parse duration
                duration_mins = data.get("duration_minutes")
                if not duration_mins and "duration" in data:
                    duration_str = data["duration"]
                    if isinstance(duration_str, str):
                        # Parse duration string like "2h 15m"
                        hours = 0
                        minutes = 0
                        if "h" in duration_str:
                            hours_part = duration_str.split("h")[0].strip()
                            if hours_part.isdigit():
                                hours = int(hours_part)
                        if "m" in duration_str:
                            minutes_part = (
                                duration_str.split("h")[-1].split("m")[0].strip()
                            )
                            if minutes_part.isdigit():
                                minutes = int(minutes_part)
                        duration_mins = hours * 60 + minutes

                # Create the Flight object
                flight = Flight(
                    airline=data.get("airline", "Unknown"),
                    flight_number=data.get("flight_number", "Unknown"),
                    departure_airport=data.get("departure_airport", ""),
                    arrival_airport=data.get("arrival_airport", ""),
                    departure_time=dep_time,
                    arrival_time=arr_time,
                    price=float(data.get("price", 0)),
                    currency=data.get("currency", "USD"),
                    travel_class=TravelMode(data.get("travel_class", "economy")),
                    layovers=data.get("layovers", []),
                    duration_minutes=duration_mins or 0,
                    booking_link=data.get("booking_link"),
                    refundable=data.get("refundable", False),
                    baggage_allowance=data.get("baggage_allowance"),
                )
                flights.append(flight)
            except Exception as e:
                logger.error(f"Error parsing flight data: {e!s}")
                logger.error(f"Problem data: {data}")
                # Continue with other flights even if one fails

        return flights

    @staticmethod
    def format_date_for_provider(date_obj: date, provider: str) -> str:
        """
        Format date appropriately for different providers.

        Args:
            date_obj: Date to format
            provider: Flight search provider

        Returns:
            Formatted date string
        """
        if provider == "google_flights":
            return date_obj.strftime("%Y-%m-%d")
        elif provider == "skyscanner":
            return date_obj.strftime("%d/%m/%Y")
        elif provider == "kayak":
            return date_obj.strftime("%m/%d/%Y")
        else:
            return date_obj.strftime("%Y-%m-%d")
