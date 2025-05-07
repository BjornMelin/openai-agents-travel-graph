"""
Accommodation search automation for travel planning.

This module implements browser automation for accommodation booking websites
using Stagehand, providing consistent interface for retrieving accommodation
options from different providers.
"""

import asyncio
import os
from datetime import date, datetime
from typing import Any, ClassVar

from stagehand import PromptTemplate

from travel_planner.browser.automation import BrowserAutomation
from travel_planner.browser.error_recovery import with_recovery
from travel_planner.data.models import (
    Accommodation,
    AccommodationSearchParams,
    AccommodationType,
)
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


class AccommodationSearchAutomation(BrowserAutomation):
    """Browser automation for accommodation search websites."""

    SEARCH_SITES: ClassVar[dict[str, dict[str, str | list[str]]]] = {
        "booking": {
            "url": "https://www.booking.com",
            "supported_features": ["hotels", "apartments", "hostels", "resorts"],
        },
        "airbnb": {
            "url": "https://www.airbnb.com",
            "supported_features": ["apartments", "homes", "villas"],
        },
        "hotels": {
            "url": "https://www.hotels.com",
            "supported_features": ["hotels", "resorts", "apartments"],
        },
    }

    def __init__(
        self,
        provider: str = "booking",
        headless: bool = True,
        slow_mo: int = 50,
        timeout: int = 60000,
        use_cache: bool = True,
    ):
        """
        Initialize accommodation search automation.

        Args:
            provider: Accommodation provider (booking, airbnb, hotels)
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

    @with_recovery(retries=2, delay=1, backoff=2)
    async def search_accommodations(
        self, params: AccommodationSearchParams
    ) -> list[Accommodation]:
        """
        Search for accommodations with the specified parameters.

        Args:
            destination: Destination city or area
            check_in_date: Check-in date
            check_out_date: Check-out date
            adults: Number of adults
            children: Number of children
            rooms: Number of rooms
            accommodation_type: Type of accommodation to filter by
            amenities: List of required amenities
            max_price: Maximum price per night
            min_rating: Minimum rating (e.g., 4.0)
            max_results: Maximum number of results to return
            sort_by: Sort criteria (popularity, price, rating, distance)

        Returns:
            List of Accommodation objects with search results
        """
        await self.setup()

        try:
            # Navigate to the accommodation provider
            await self.navigate(self.base_url)

            # Wait for the page to load
            await asyncio.sleep(2)

            # Handle any consent dialogs or pop-ups
            await self._handle_initial_popups()

            # Fill out the search form
            await self._fill_search_form(params)

            # Submit the search and wait for results
            await self._submit_search()

            # Apply additional filters if provided
            if (
                params.accommodation_type
                or params.amenities
                or params.max_price
                or params.min_rating
            ):
                await self._apply_filters(
                    accommodation_type=params.accommodation_type,
                    amenities=params.amenities,
                    max_price=params.max_price,
                    min_rating=params.min_rating,
                )

            # Sort results if needed
            if params.sort_by != "popularity":
                await self._sort_results(params.sort_by)

            # Extract accommodation results
            accommodations = await self._extract_accommodation_results(
                params.max_results
            )

            return accommodations
        except Exception as e:
            logger.error(f"Error during accommodation search: {e!s}")
            # Take a screenshot of the failure for debugging
            error_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "error_screenshots"
            )
            os.makedirs(error_dir, exist_ok=True)
            screenshot_path = os.path.join(
                error_dir,
                f"accommodation_search_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            await self.screenshot(screenshot_path)
            logger.info(f"Error screenshot saved to {screenshot_path}")
            raise
        finally:
            await self.teardown()

    async def _handle_initial_popups(self):
        """Handle consent dialogs, pop-ups, and other initial page elements."""
        try:
            if self.provider == "booking":
                # Check for cookie consent dialog on Booking.com
                cookie_button = await self.page.query_selector(
                    "button[id*='cookie-consent'], button[aria-label*='cookie'], button[data-testid='accept-cookies']"
                )
                if cookie_button:
                    await cookie_button.click()
                    await asyncio.sleep(1)
            elif self.provider == "airbnb":
                # Handle cookie consent on Airbnb
                cookie_button = await self.page.query_selector(
                    "button[data-testid='accept-cookies'], button[data-testid='accept-btn']"
                )
                if cookie_button:
                    await cookie_button.click()
                    await asyncio.sleep(1)

            # Use AI to close any unexpected pop-ups
            prompt = PromptTemplate("""
            Look for and close any pop-ups, modal dialogs, or overlays that might interfere 
            with the search process. This might include:
            - Cookie consent dialogs
            - Newsletter subscription prompts
            - App download suggestions
            - Special offers or promotions
            
            If you find any, close them by clicking the appropriate X button or "Close" link.
            """)

            await self.execute_ai_action(prompt, {})
        except Exception as e:
            logger.warning(f"Error handling popups: {e!s}")
            # Continue despite errors with popups

    async def _fill_search_form(self, params: AccommodationSearchParams):
        """
        Fill out the accommodation search form.

        This uses AI-assisted form filling for flexibility across different sites.

        Args:
            destination: Destination city or area
            check_in_date: Check-in date
            check_out_date: Check-out date
            adults: Number of adults
            children: Number of children
            rooms: Number of rooms
        """
        # Use AI-based form filling for flexibility across different sites
        prompt = PromptTemplate("""
        Fill out the accommodation search form with the following details:
        
        - Destination: {{destination}}
        - Check-in Date: {{check_in_date}}
        - Check-out Date: {{check_out_date}}
        - Number of Adults: {{adults}}
        - Number of Children: {{children}}
        - Number of Rooms: {{rooms}}
        
        Please find and fill out the appropriate form fields, but DO NOT submit the search yet.
        Just complete the form so it's ready for submission.
        """)

        context = {
            "destination": params.destination,
            "check_in_date": params.check_in_date.strftime("%Y-%m-%d"),
            "check_out_date": params.check_out_date.strftime("%Y-%m-%d"),
            "adults": params.adults,
            "children": params.children,
            "rooms": params.rooms,
        }

        logger.info(f"Filling accommodation search form for {params.destination}")
        await self.execute_ai_action(prompt, context)

    async def _submit_search(self):
        """Submit the search form and wait for results to load."""
        # Use AI to find and click the search button
        prompt = PromptTemplate("""
        Find and click the search button to start the accommodation search.
        After clicking, wait for the search results to load completely.
        """)

        logger.info("Submitting accommodation search")
        await self.execute_ai_action(prompt, {})

        # Wait for results to load
        await self._wait_for_search_results()

    async def _wait_for_search_results(self, max_wait_time: int = 30):
        """
        Wait for search results to load.

        Args:
            max_wait_time: Maximum time to wait in seconds
        """
        logger.info("Waiting for accommodation search results to load")

        # Use AI to determine when results are fully loaded
        prompt = PromptTemplate("""
        Wait for the accommodation search results to load completely.
        Check for:
        - Search progress indicators or spinners to disappear
        - Accommodation listing cards or results to appear
        - Filter options to become interactive
        
        Once the page is fully loaded with results, proceed.
        """)

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

    async def _apply_filters(
        self,
        accommodation_type: AccommodationType | None = None,
        amenities: list[str] | None = None,
        max_price: float | None = None,
        min_rating: float | None = None,
    ):
        """
        Apply filters to the search results.

        Args:
            accommodation_type: Type of accommodation to filter by
            amenities: List of required amenities
            max_price: Maximum price per night
            min_rating: Minimum rating
        """
        logger.info("Applying accommodation filters")

        # Build filter instructions based on provided parameters
        filter_instructions = []

        if accommodation_type:
            filter_instructions.append(
                f"- Accommodation Type: {accommodation_type.value}"
            )

        if amenities and len(amenities) > 0:
            amenities_str = ", ".join(amenities)
            filter_instructions.append(f"- Required Amenities: {amenities_str}")

        if max_price:
            filter_instructions.append(f"- Maximum Price: {max_price} per night")

        if min_rating:
            filter_instructions.append(f"- Minimum Rating: {min_rating} stars/points")

        if not filter_instructions:
            # No filters to apply
            return

        # Use AI to apply the filters
        prompt = PromptTemplate("""
        Apply the following filters to the accommodation search results:
        
        {{filters}}
        
        Find and interact with the appropriate filter controls on the page.
        After applying each filter, wait for the results to update before proceeding.
        """)

        context = {"filters": "\n".join(filter_instructions)}

        await self.execute_ai_action(prompt, context)

        # Wait for filters to be applied and results updated
        await asyncio.sleep(3)

    async def _sort_results(self, sort_by: str):
        """
        Sort the search results.

        Args:
            sort_by: Sort criteria (popularity, price, rating, distance)
        """
        logger.info(f"Sorting accommodation results by {sort_by}")

        # Create a mapping from our sort criteria to site-specific language
        sort_criteria_mapping = {
            "price": ["price", "lowest price", "cheapest", "price (low to high)"],
            "rating": ["rating", "review", "score", "guest", "top reviewed"],
            "distance": ["distance", "location", "closest", "proximity"],
            # popularity is usually the default
        }

        # Find appropriate terms for the selected sort criteria
        sort_terms = sort_criteria_mapping.get(sort_by.lower(), [sort_by])
        sort_terms_str = ", ".join(sort_terms)

        # Use AI to sort the results
        prompt = PromptTemplate("""
        Sort the accommodation search results by {{sort_criteria}}.
        
        Look for sort options or dropdown menus with terms like: {{sort_terms}}
        Select the appropriate option to sort the results.
        
        After selecting the sort option, wait for the results to update before proceeding.
        """)

        context = {"sort_criteria": sort_by, "sort_terms": sort_terms_str}

        await self.execute_ai_action(prompt, context)

        # Wait for sorting to be applied and results updated
        await asyncio.sleep(3)

    async def _extract_accommodation_results(
        self, max_results: int = 5
    ) -> list[Accommodation]:
        """
        Extract accommodation results from the page.

        Args:
            max_results: Maximum number of results to extract

        Returns:
            List of Accommodation objects
        """
        logger.info(f"Extracting up to {max_results} accommodation results")

        # Use AI to extract accommodation information
        extract_prompt = PromptTemplate("""
        Extract information for the top {{max_results}} accommodations from the search results.
        For each accommodation, extract:
        - Name
        - Type (hotel, apartment, hostel, resort, etc.)
        - Location/address
        - Price per night
        - Total price for the stay
        - Rating score
        - Check-in and check-out times (if available)
        - Available amenities
        - Links to images
        - Booking link or URL
        - Cancellation policy (if available)
        - Any key highlights or special features
        
        Return the information in a structured format.
        """)

        extraction_result = await self.extract_data(
            extraction_prompt=extract_prompt.format(max_results=max_results), context={}
        )

        # Convert the extraction result to Accommodation objects
        accommodations = self._parse_extraction_to_accommodations(extraction_result)

        logger.info(f"Extracted {len(accommodations)} accommodations")
        return accommodations

    def _parse_extraction_to_accommodations(
        self, extraction_result: dict[str, Any]
    ) -> list[Accommodation]:
        """
        Parse the extraction result into Accommodation objects.

        Args:
            extraction_result: Result from AI extraction

        Returns:
            List of Accommodation objects
        """
        accommodations = []

        # Extract the accommodation data from the extraction result
        accommodation_data = self._extract_accommodation_data(extraction_result)

        for data in accommodation_data[:5]:  # Limit to top 5 results
            try:
                # Map accommodation type string to enum
                accom_type = self._determine_accommodation_type(
                    data.get("type", "").lower()
                )

                # Parse prices
                price_per_night = self._parse_price(data.get("price_per_night", 0))
                total_price = self._parse_price(data.get("total_price", 0))

                # Create the Accommodation object
                accommodation = Accommodation(
                    name=data.get("name", "Unknown"),
                    type=accom_type,
                    location=data.get("location", ""),
                    address=data.get("address", ""),
                    rating=data.get("rating"),
                    price_per_night=price_per_night,
                    currency=data.get("currency", "USD"),
                    total_price=total_price,
                    check_in_time=data.get("check_in_time", ""),
                    check_out_time=data.get("check_out_time", ""),
                    amenities=data.get("amenities", []),
                    images=data.get("images", []),
                    booking_link=data.get("booking_link"),
                    cancellation_policy=data.get("cancellation_policy"),
                    highlights=data.get("highlights", []),
                )
                accommodations.append(accommodation)
            except Exception as e:
                logger.error(f"Error parsing accommodation data: {e!s}")
                logger.error(f"Problem data: {data}")
                # Continue with other accommodations even if one fails

        return accommodations

    def _extract_accommodation_data(
        self, extraction_result: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract accommodation data from extraction result."""
        accommodation_data = extraction_result.get("accommodations", [])
        if not accommodation_data and isinstance(extraction_result, list):
            accommodation_data = extraction_result
        return accommodation_data

    def _determine_accommodation_type(self, accom_type_str: str) -> AccommodationType:
        """Map accommodation type string to enum value."""
        # Define mapping from keywords to accommodation types
        type_mapping = {
            "hotel": AccommodationType.HOTEL,
            "apartment": AccommodationType.APARTMENT,
            "hostel": AccommodationType.HOSTEL,
            "resort": AccommodationType.RESORT,
            "villa": AccommodationType.VILLA,
            "airbnb": AccommodationType.AIRBNB,
            "boutique": AccommodationType.BOUTIQUE,
            "guesthouse": AccommodationType.GUESTHOUSE,
        }

        # Check for keywords in the type string
        for keyword, accom_type in type_mapping.items():
            if keyword in accom_type_str:
                return accom_type

        # Default to hotel if no match
        return AccommodationType.HOTEL

    def _parse_price(self, price_value: Any) -> float:
        """Parse price value to float."""
        if not isinstance(price_value, str):
            return float(price_value) if price_value else 0

        # Remove currency symbols and commas, then convert to float
        price_str = price_value.replace(",", "")
        price_str = "".join(c for c in price_str if c.isdigit() or c == ".")
        return float(price_str) if price_str else 0

    @staticmethod
    def format_date_for_provider(date_obj: date, provider: str) -> str:
        """
        Format date appropriately for different providers.

        Args:
            date_obj: Date to format
            provider: Accommodation provider

        Returns:
            Formatted date string
        """
        if provider == "booking":
            return date_obj.strftime("%Y-%m-%d")
        elif provider == "airbnb":
            return date_obj.strftime("%m/%d/%Y")
        elif provider == "hotels":
            return date_obj.strftime("%m/%d/%Y")
        else:
            return date_obj.strftime("%Y-%m-%d")
