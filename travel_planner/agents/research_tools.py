"""
Research tools for the travel planning system.

This module implements integration with Tavily and Firecrawl
for comprehensive destination research in the travel planning process.
"""

import os
from typing import Any

from travel_planner.utils.logging import get_logger
from travel_planner.utils.rate_limiting import APIClient, rate_limited

logger = get_logger(__name__)


class TavilyResearch(APIClient):
    """Integration with Tavily AI search for travel research."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize Tavily research.

        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY environment variable)
        """
        api_key = api_key or os.environ.get("TAVILY_API_KEY")

        if not api_key:
            raise ValueError(
                "Tavily API key must be provided either as an argument or "
                "as TAVILY_API_KEY environment variable"
            )

        super().__init__(
            service_name="tavily", base_url="https://api.tavily.com/v1", api_key=api_key
        )

    async def search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int = 10,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Perform a Tavily search.

        Args:
            query: Search query
            search_depth: Search depth ('basic' or 'advanced')
            max_results: Maximum number of results
            include_domains: Domains to include
            exclude_domains: Domains to exclude

        Returns:
            Search results
        """
        logger.info(f"Performing Tavily search: {query}")

        request = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
        }

        if include_domains:
            request["include_domains"] = include_domains

        if exclude_domains:
            request["exclude_domains"] = exclude_domains

        # Use the MCP tool to perform the search
        return await self._call_tavily_mcp(request)

    async def search_travel_destination(
        self,
        destination: str,
        include_images: bool = True,
        include_weather: bool = True,
        include_travel_advisories: bool = True,
    ) -> dict[str, Any]:
        """
        Specialized search for travel destinations.

        Args:
            destination: Destination to research
            include_images: Whether to include images
            include_weather: Whether to include weather information
            include_travel_advisories: Whether to include travel advisories

        Returns:
            Destination research results
        """
        logger.info(f"Researching travel destination: {destination}")

        # Build a comprehensive query for the destination
        # We'll run multiple specialized searches and combine the results
        results = {}

        # General information query
        general_query = f"travel guide {destination} tourist attractions things to do"
        general_results = await self.search(
            general_query,
            search_depth="advanced",
            max_results=5,
            include_domains=[
                "lonelyplanet.com",
                "wikitravel.org",
                "tripadvisor.com",
                "travel.usnews.com",
            ],
        )
        results["general_info"] = general_results

        # Weather information if requested
        if include_weather:
            weather_query = f"weather climate best time to visit {destination}"
            weather_results = await self.search(
                weather_query,
                search_depth="basic",
                max_results=3,
                include_domains=["weather.com", "accuweather.com", "weatherspark.com"],
            )
            results["weather"] = weather_results

        # Travel advisories if requested
        if include_travel_advisories:
            advisory_query = f"travel advisory safety {destination}"
            advisory_results = await self.search(
                advisory_query,
                search_depth="basic",
                max_results=3,
                include_domains=["travel.state.gov", "gov.uk/foreign-travel-advice"],
            )
            results["travel_advisories"] = advisory_results

        # Extract and process the combined results
        return self._process_destination_results(destination, results)

    def _process_destination_results(
        self, destination: str, raw_results: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process raw search results into structured destination information.

        Args:
            destination: Destination name
            raw_results: Raw search results

        Returns:
            Processed destination information
        """
        processed = {
            "name": destination,
            "attractions": [],
            "weather": {},
            "best_times_to_visit": [],
            "safety_info": "",
            "local_transportation": [],
            "cultural_info": [],
        }

        # Process different types of information
        self._process_general_info(raw_results.get("general_info", {}), processed)
        self._process_weather_info(raw_results.get("weather", {}), processed)
        self._process_travel_advisories(
            raw_results.get("travel_advisories", {}), processed
        )

        return processed

    def _process_general_info(
        self, general_info: dict[str, Any], processed: dict[str, Any]
    ) -> None:
        """Process general destination information."""
        for result in general_info.get("results", []):
            if "content" not in result:
                continue

            content = result["content"]
            self._extract_attractions(content, processed)
            self._extract_cultural_info(content, processed)
            self._extract_transportation_info(content, processed)

    def _extract_attractions(self, content: str, processed: dict[str, Any]) -> None:
        """Extract attractions from content."""
        if not ("attractions" in content.lower() or "things to do" in content.lower()):
            return

        attraction_markers = ["attraction", "visit", "sight", "landmark"]
        for line in content.split("\n"):
            if any(marker in line.lower() for marker in attraction_markers):
                processed["attractions"].append(line.strip())

    def _extract_cultural_info(self, content: str, processed: dict[str, Any]) -> None:
        """Extract cultural information from content."""
        cultural_markers = ["culture", "tradition", "custom", "etiquette"]
        if any(marker in content.lower() for marker in cultural_markers):
            processed["cultural_info"].append(content)

    def _extract_transportation_info(
        self, content: str, processed: dict[str, Any]
    ) -> None:
        """Extract transportation information from content."""
        transport_markers = ["transport", "getting around", "metro", "bus", "taxi"]
        if any(marker in content.lower() for marker in transport_markers):
            processed["local_transportation"].append(content)

    def _process_weather_info(
        self, weather_info: dict[str, Any], processed: dict[str, Any]
    ) -> None:
        """Process weather information."""
        for result in weather_info.get("results", []):
            if "content" not in result:
                continue

            content = result["content"]
            self._extract_best_times(content, processed)
            self._extract_seasonal_weather(content, processed)

    def _extract_best_times(self, content: str, processed: dict[str, Any]) -> None:
        """Extract best times to visit."""
        if "best time" not in content.lower():
            return

        for line in content.split("\n"):
            if "best time" in line.lower():
                processed["best_times_to_visit"].append(line.strip())

    def _extract_seasonal_weather(
        self, content: str, processed: dict[str, Any]
    ) -> None:
        """Extract seasonal weather information."""
        seasons = ["summer", "winter", "spring", "fall", "autumn"]
        if any(season in content.lower() for season in seasons):
            processed["weather"]["seasonal"] = content

    def _process_travel_advisories(
        self, advisory_info: dict[str, Any], processed: dict[str, Any]
    ) -> None:
        """Process travel advisory information."""
        advisory_texts = []
        for result in advisory_info.get("results", []):
            if "content" in result:
                advisory_texts.append(result["content"])

        if advisory_texts:
            processed["safety_info"] = "\n".join(advisory_texts)

    @rate_limited("tavily")
    async def _call_tavily_mcp(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Call Tavily using rate limiting and exponential backoff.

        Args:
            request: Request parameters

        Returns:
            Search results
        """
        logger.info(f"Calling Tavily API with query: {request.get('query', 'unknown')}")

        try:
            # In a real implementation, we would use the request method from the APIClient
            # return await self.request(
            #     method="POST",
            #     endpoint="search",
            #     json_data=request
            # )

            # For now, return mock results
            mock_results = {
                "results": [
                    {
                        "title": "Sample Search Result 1",
                        "url": "https://example.com/result1",
                        "content": "This is a sample search result about travel destinations.",
                    },
                    {
                        "title": "Sample Search Result 2",
                        "url": "https://example.com/result2",
                        "content": "Another sample search result with information about attractions.",
                    },
                ],
                "query": request["query"],
                "search_depth": request["search_depth"],
            }

            return mock_results

        except Exception as e:
            logger.error(f"Error calling Tavily API: {e}")
            # Return minimal results as fallback
            return {
                "results": [],
                "query": request.get("query", "unknown"),
                "error": str(e),
            }


class FirecrawlResearch(APIClient):
    """Integration with Firecrawl for in-depth web research."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize Firecrawl research.

        Args:
            api_key: Firecrawl API key (defaults to FIRECRAWL_API_KEY environment variable)
        """
        api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")

        if not api_key:
            raise ValueError(
                "Firecrawl API key must be provided either as an argument or "
                "as FIRECRAWL_API_KEY environment variable"
            )

        super().__init__(
            service_name="firecrawl",
            base_url="https://api.firecrawl.dev/v1",
            api_key=api_key,
        )

    async def deep_research(
        self, query: str, max_urls: int = 10, max_depth: int = 3, time_limit: int = 120
    ) -> dict[str, Any]:
        """
        Perform deep research using Firecrawl.

        Args:
            query: Research query
            max_urls: Maximum number of URLs to crawl
            max_depth: Maximum crawl depth
            time_limit: Time limit in seconds

        Returns:
            Research results
        """
        logger.info(f"Performing Firecrawl deep research: {query}")

        request = {
            "query": query,
            "maxUrls": max_urls,
            "maxDepth": max_depth,
            "timeLimit": time_limit,
        }

        # Use the MCP tool to perform the research
        return await self._call_firecrawl_mcp("firecrawl_deep_research", request)

    async def crawl_travel_site(
        self,
        url: str,
        max_depth: int = 2,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Crawl a travel website for information.

        Args:
            url: Starting URL
            max_depth: Maximum crawl depth
            include_paths: URL paths to include
            exclude_paths: URL paths to exclude

        Returns:
            Crawl results
        """
        logger.info(f"Crawling travel site: {url}")

        request = {"url": url, "maxDepth": max_depth}

        if include_paths:
            request["includePaths"] = include_paths

        if exclude_paths:
            request["excludePaths"] = exclude_paths

        # Use the MCP tool to perform the crawl
        return await self._call_firecrawl_mcp("firecrawl_crawl", request)

    async def extract_travel_info(
        self, urls: list[str], extraction_type: str = "destination"
    ) -> dict[str, Any]:
        """
        Extract structured travel information from URLs.

        Args:
            urls: URLs to extract information from
            extraction_type: Type of information to extract

        Returns:
            Extracted information
        """
        logger.info(f"Extracting travel information from {len(urls)} URLs")

        # Define schema based on extraction type
        schema = None
        prompt = None

        if extraction_type == "destination":
            schema = {
                "name": "string",
                "country": "string",
                "description": "string",
                "attractions": "array",
                "best_time_to_visit": "string",
                "safety_level": "string",
                "budget_category": "string",
                "transportation_options": "array",
            }
            prompt = "Extract key travel destination information from this content."
        elif extraction_type == "accommodation":
            schema = {
                "type": "string",
                "price_range": "string",
                "amenities": "array",
                "location_quality": "string",
                "recommended_for": "string",
            }
            prompt = "Extract accommodation information from this content."
        elif extraction_type == "activity":
            schema = {
                "name": "string",
                "category": "string",
                "description": "string",
                "duration": "string",
                "price_range": "string",
                "best_for": "string",
                "booking_required": "boolean",
            }
            prompt = "Extract activity information from this content."

        request = {"urls": urls, "schema": schema, "prompt": prompt}

        # Use the MCP tool to perform the extraction
        return await self._call_firecrawl_mcp("firecrawl_extract", request)

    async def search_travel_content(self, query: str, limit: int = 5) -> dict[str, Any]:
        """
        Search for travel content.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            Search results
        """
        logger.info(f"Searching for travel content: {query}")

        request = {
            "query": query,
            "limit": limit,
            "scrapeOptions": {"formats": ["markdown"]},
        }

        # Use the MCP tool to perform the search
        return await self._call_firecrawl_mcp("firecrawl_search", request)

    @rate_limited("firecrawl")
    async def _call_firecrawl_mcp(
        self, function_name: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Call Firecrawl using rate limiting and exponential backoff.

        Args:
            function_name: Firecrawl function name
            request: Request parameters

        Returns:
            Function results
        """
        logger.info(f"Calling Firecrawl API function {function_name}")

        try:
            # In a real implementation, we would use the request method from the APIClient
            # endpoint = f"{function_name.replace('firecrawl_', '')}"
            # return await self.request(
            #     method="POST",
            #     endpoint=endpoint,
            #     json_data=request
            # )

            # Return mock results for demonstration
            if function_name == "firecrawl_deep_research":
                return {
                    "summary": "Sample deep research results about travel destinations.",
                    "keyInsights": [
                        "Key insight 1 about the destination.",
                        "Key insight 2 about travel tips.",
                        "Key insight 3 about local attractions.",
                    ],
                    "sources": [
                        {
                            "url": "https://example.com/travel1",
                            "title": "Travel Example 1",
                        },
                        {
                            "url": "https://example.com/travel2",
                            "title": "Travel Example 2",
                        },
                    ],
                }
            elif function_name == "firecrawl_crawl":
                return {
                    "crawlId": "sample-crawl-id",
                    "status": "completed",
                    "urlsCrawled": 5,
                    "results": [
                        {
                            "url": "https://example.com/travel/page1",
                            "content": "Sample content 1",
                        },
                        {
                            "url": "https://example.com/travel/page2",
                            "content": "Sample content 2",
                        },
                    ],
                }
            elif function_name == "firecrawl_extract":
                return {
                    "extractions": [
                        {
                            "url": "https://example.com/destination1",
                            "data": {
                                "name": "Sample Destination",
                                "country": "Sample Country",
                                "description": "A beautiful destination with many attractions.",
                                "attractions": [
                                    "Attraction 1",
                                    "Attraction 2",
                                    "Attraction 3",
                                ],
                                "best_time_to_visit": "Spring and Fall",
                                "safety_level": "High",
                                "budget_category": "Mid-range",
                                "transportation_options": ["Bus", "Taxi", "Metro"],
                            },
                        }
                    ]
                }
            elif function_name == "firecrawl_search":
                return {
                    "results": [
                        {
                            "title": "Sample Travel Search Result 1",
                            "url": "https://example.com/travel-result1",
                            "content": "This is a sample travel search result.",
                        },
                        {
                            "title": "Sample Travel Search Result 2",
                            "url": "https://example.com/travel-result2",
                            "content": "Another sample travel search result.",
                        },
                    ]
                }
            else:
                return {"error": f"Unknown function: {function_name}"}

        except Exception as e:
            logger.error(f"Error calling Firecrawl API function {function_name}: {e}")
            # Return minimal results as fallback
            return {"error": str(e), "function": function_name}


class DestinationResearchTools:
    """Combined research tools for destination research."""

    def __init__(
        self, tavily_api_key: str | None = None, firecrawl_api_key: str | None = None
    ):
        """
        Initialize destination research tools.

        Args:
            tavily_api_key: Tavily API key
            firecrawl_api_key: Firecrawl API key
        """
        self.tavily = TavilyResearch(api_key=tavily_api_key)
        self.firecrawl = FirecrawlResearch(api_key=firecrawl_api_key)

    async def research_destination(
        self, destination: str, detailed: bool = False
    ) -> dict[str, Any]:
        """
        Perform comprehensive destination research.

        Args:
            destination: Destination to research
            detailed: Whether to perform detailed research

        Returns:
            Research results
        """
        logger.info(f"Researching destination: {destination} (detailed={detailed})")

        # Start with basic Tavily research
        tavily_results = await self.tavily.search_travel_destination(
            destination,
            include_images=True,
            include_weather=True,
            include_travel_advisories=True,
        )

        # For detailed research, add Firecrawl deep research
        if detailed:
            # Perform deep research on the destination
            deep_research_query = f"travel guide {destination} tourist attractions things to do local customs transportation"
            firecrawl_results = await self.firecrawl.deep_research(
                deep_research_query, max_urls=15, max_depth=3, time_limit=180
            )

            # Extract travel website information
            travel_sites = [
                f"https://www.lonelyplanet.com/search?q={destination}",
                f"https://www.tripadvisor.com/Search?q={destination}",
                f"https://wikitravel.org/en/{destination.replace(' ', '_')}",
            ]

            extraction_results = await self.firecrawl.extract_travel_info(
                travel_sites, extraction_type="destination"
            )

            # Combine all results
            combined_results = self._combine_research_results(
                tavily_results, firecrawl_results, extraction_results
            )

            return combined_results
        else:
            # For basic research, just return Tavily results
            return tavily_results

    async def get_destination_activities(
        self, destination: str, activity_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Get activities for a destination.

        Args:
            destination: Destination to get activities for
            activity_types: Types of activities to include

        Returns:
            List of activities
        """
        logger.info(f"Getting activities for {destination}")

        # Build query based on activity types
        if activity_types:
            activity_query = (
                f"top {', '.join(activity_types)} activities in {destination}"
            )
        else:
            activity_query = f"top things to do activities in {destination}"

        # Search for activities
        search_results = await self.firecrawl.search_travel_content(
            activity_query, limit=10
        )

        # Extract activity information
        if "results" in search_results and len(search_results["results"]) > 0:
            activity_urls = [result["url"] for result in search_results["results"]]
            extraction_results = await self.firecrawl.extract_travel_info(
                activity_urls[:5],  # Limit to top 5 results
                extraction_type="activity",
            )

            if "extractions" in extraction_results:
                return [
                    extraction["data"]
                    for extraction in extraction_results["extractions"]
                ]

        # Fallback to simple extraction from search results
        activities = []
        for result in search_results.get("results", []):
            if "content" in result:
                activities.append(
                    {
                        "name": result.get("title", "Unknown Activity"),
                        "description": result.get("content", "")[
                            :200
                        ],  # Truncate for brevity
                        "url": result.get("url", ""),
                    }
                )

        return activities

    async def get_accommodation_info(
        self, destination: str, accommodation_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Get accommodation information for a destination.

        Args:
            destination: Destination to get accommodation for
            accommodation_type: Type of accommodation

        Returns:
            Accommodation information
        """
        logger.info(f"Getting accommodation info for {destination}")

        # Build query based on accommodation type
        if accommodation_type:
            accommodation_query = f"best {accommodation_type} in {destination}"
        else:
            accommodation_query = f"best places to stay in {destination}"

        # Search for accommodation information
        search_results = await self.tavily.search(
            accommodation_query, search_depth="basic", max_results=5
        )

        # Process results
        accommodations = []
        for result in search_results.get("results", []):
            if "content" in result:
                accommodations.append(
                    {
                        "name": result.get("title", "Unknown Accommodation"),
                        "description": result.get("content", "")[
                            :200
                        ],  # Truncate for brevity
                        "url": result.get("url", ""),
                    }
                )

        return accommodations

    def _combine_research_results(
        self,
        tavily_results: dict[str, Any],
        firecrawl_results: dict[str, Any],
        extraction_results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Combine research results from multiple sources.

        Args:
            tavily_results: Results from Tavily
            firecrawl_results: Results from Firecrawl deep research
            extraction_results: Results from Firecrawl extraction

        Returns:
            Combined research results
        """
        combined = tavily_results.copy()

        # Add key insights from deep research
        if "keyInsights" in firecrawl_results:
            combined["key_insights"] = firecrawl_results["keyInsights"]

        # Add sources from deep research
        if "sources" in firecrawl_results:
            combined["sources"] = firecrawl_results["sources"]

        # Add extracted information
        if "extractions" in extraction_results:
            for extraction in extraction_results["extractions"]:
                if "data" in extraction:
                    data = extraction["data"]

                    # Merge attractions
                    if "attractions" in data and isinstance(data["attractions"], list):
                        combined["attractions"].extend(data["attractions"])
                        # Remove duplicates
                        combined["attractions"] = list(set(combined["attractions"]))

                    # Add other fields if not already present
                    for key, value in data.items():
                        if key not in combined or not combined[key]:
                            combined[key] = value

        return combined
