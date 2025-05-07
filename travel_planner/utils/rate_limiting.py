"""
Rate limiting and API request management for external services.

This module provides utilities for rate limiting, request throttling,
exponential backoff, and API quota management to ensure proper use of
external services while respecting their rate limits and quotas.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, TypeVar, cast

import aiohttp
from aiolimiter import AsyncLimiter
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from travel_planner.utils.error_handling import APIError

# Type variables for function decorator typing
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

# HTTP Status Codes
HTTP_STATUS_OK = 200
HTTP_STATUS_REDIRECT = 300
HTTP_STATUS_TOO_MANY_REQUESTS = 429


@dataclass
class RateLimitConfig:
    """Configuration for a service's rate limits."""

    service_name: str
    requests_per_minute: int  # Rate limit in requests per minute
    requests_per_day: int  # Daily quota
    max_retries: int = 3  # Maximum number of retries for failed requests
    min_wait_seconds: float = 1.0  # Minimum wait time for backoff
    max_wait_seconds: float = 30.0  # Maximum wait time for backoff
    retry_status_codes: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )
    cooldown_after_quota: int = 60  # Seconds to wait after hitting quota


@dataclass
class QuotaUsage:
    """Tracks API quota usage for a service."""

    daily_count: int = 0
    last_reset: datetime = field(default_factory=datetime.now)

    def increment(self) -> None:
        """Increment the daily usage counter, resetting if necessary."""
        current_time = datetime.now()

        # Reset daily counter if it's a new day
        if current_time.date() > self.last_reset.date():
            self.daily_count = 0
            self.last_reset = current_time

        self.daily_count += 1

    def get_remaining(self, daily_quota: int) -> int:
        """Get remaining requests for the day."""
        # Reset if needed
        current_time = datetime.now()
        if current_time.date() > self.last_reset.date():
            self.daily_count = 0
            self.last_reset = current_time

        return max(0, daily_quota - self.daily_count)

    def is_quota_exceeded(self, daily_quota: int) -> bool:
        """Check if daily quota is exceeded."""
        return self.get_remaining(daily_quota) <= 0


class ServiceRateLimiter:
    """
    Rate limiter for a specific service.

    Manages both short-term rate limits (requests per minute) and
    long-term quotas (requests per day) for a service.
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize the rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.quota_usage = QuotaUsage()

        # Create AsyncLimiter with requests_per_minute / 60 = requests per second
        requests_per_second = max(
            0.02, config.requests_per_minute / 60
        )  # minimum of 1 request per 50 seconds
        self.limiter = AsyncLimiter(requests_per_second, 1)

        # Track request timestamps for windowed rate limiting
        self.request_timestamps: list[float] = []

        logger.info(
            f"Initialized rate limiter for {config.service_name} "
            f"({config.requests_per_minute}/min, {config.requests_per_day}/day)"
        )

    async def acquire(self) -> bool:
        """
        Acquire permission to make a request.

        Returns:
            True if request is allowed, False otherwise
        """
        # Check if daily quota is exceeded
        if self.quota_usage.is_quota_exceeded(self.config.requests_per_day):
            logger.warning(
                f"Daily quota exceeded for {self.config.service_name} "
                f"({self.config.requests_per_day} requests/day)"
            )
            return False

        # Clean up old request timestamps
        current_time = time.time()
        minute_ago = current_time - 60
        self.request_timestamps = [
            ts for ts in self.request_timestamps if ts > minute_ago
        ]

        # Check if minute limit is reached
        if len(self.request_timestamps) >= self.config.requests_per_minute:
            logger.warning(
                f"Rate limit reached for {self.config.service_name} "
                f"({self.config.requests_per_minute} requests/minute)"
            )
            return False

        # Acquire token from AsyncLimiter
        async with self.limiter:
            # Record this request
            self.request_timestamps.append(current_time)
            self.quota_usage.increment()
            return True

    def get_backoff_time(self) -> float:
        """
        Calculate backoff time when rate limited.

        Returns:
            Backoff time in seconds
        """
        # If we're near the quota, use longer backoff
        remaining_quota = self.quota_usage.get_remaining(self.config.requests_per_day)
        quota_factor = max(
            1, (self.config.requests_per_day * 0.1) / max(1, remaining_quota)
        )

        # If we're at the minute limit, calculate time until oldest request expires
        if len(self.request_timestamps) >= self.config.requests_per_minute:
            current_time = time.time()
            oldest = min(self.request_timestamps)
            time_until_slot_available = max(0, oldest + 60 - current_time)
            return time_until_slot_available * quota_factor

        # Otherwise use a small backoff
        return self.config.min_wait_seconds * quota_factor

    def should_retry_exception(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.

        Args:
            exception: The exception to check

        Returns:
            True if should retry, False otherwise
        """
        # Always retry on connection errors
        if isinstance(
            exception, aiohttp.ClientConnectorError | aiohttp.ServerDisconnectedError
        ):
            return True

        # Retry on API errors with specific status codes
        if (
            isinstance(exception, APIError)
            and exception.status_code in self.config.retry_status_codes
        ):
            return True

        return False

    def get_quota_stats(self) -> dict[str, Any]:
        """
        Get quota usage statistics.

        Returns:
            Dictionary with quota statistics
        """
        return {
            "service": self.config.service_name,
            "daily_quota": self.config.requests_per_day,
            "used_today": self.quota_usage.daily_count,
            "remaining": self.quota_usage.get_remaining(self.config.requests_per_day),
            "minute_limit": self.config.requests_per_minute,
            "current_minute_usage": len(self.request_timestamps),
        }


class RateLimitManager:
    """
    Manager for rate limiters across multiple services.

    Provides a centralized interface for managing rate limits and quotas
    for multiple external services.
    """

    def __init__(self):
        """Initialize the rate limit manager."""
        self.limiters: dict[str, ServiceRateLimiter] = {}
        self.default_config = RateLimitConfig(
            service_name="default",
            requests_per_minute=30,
            requests_per_day=1000,
            max_retries=3,
            min_wait_seconds=1.0,
            max_wait_seconds=30.0,
        )

    def register_service(self, config: RateLimitConfig) -> ServiceRateLimiter:
        """
        Register a service with the rate limit manager.

        Args:
            config: Rate limit configuration for the service

        Returns:
            ServiceRateLimiter for the registered service
        """
        limiter = ServiceRateLimiter(config)
        self.limiters[config.service_name] = limiter
        return limiter

    def get_limiter(self, service_name: str) -> ServiceRateLimiter:
        """
        Get the rate limiter for a service.

        Args:
            service_name: Name of the service

        Returns:
            ServiceRateLimiter for the service, or a default one if not registered
        """
        if service_name not in self.limiters:
            logger.warning(
                f"No rate limiter configured for {service_name}, "
                f"using default configuration."
            )
            # Create a new config with this service name but default values
            config = RateLimitConfig(
                service_name=service_name,
                requests_per_minute=self.default_config.requests_per_minute,
                requests_per_day=self.default_config.requests_per_day,
                max_retries=self.default_config.max_retries,
                min_wait_seconds=self.default_config.min_wait_seconds,
                max_wait_seconds=self.default_config.max_wait_seconds,
            )
            self.register_service(config)

        return self.limiters[service_name]

    def get_all_quota_stats(self) -> list[dict[str, Any]]:
        """
        Get quota statistics for all services.

        Returns:
            List of dictionaries with quota statistics for each service
        """
        return [limiter.get_quota_stats() for limiter in self.limiters.values()]

    async def wait_if_needed(self, service_name: str) -> None:
        """
        Wait if rate limits for a service are close to being exceeded.

        Args:
            service_name: Name of the service
        """
        limiter = self.get_limiter(service_name)

        # Check if we should wait
        if not await limiter.acquire():
            backoff_time = limiter.get_backoff_time()
            logger.warning(
                f"Rate limit reached for {service_name}, "
                f"waiting {backoff_time:.2f} seconds"
            )
            await asyncio.sleep(backoff_time)


# Create a global instance for the application to use
rate_limit_manager = RateLimitManager()


def configure_rate_limits(service_configs: list[RateLimitConfig]) -> None:
    """
    Configure rate limits for multiple services.

    Args:
        service_configs: List of rate limit configurations for services
    """
    for config in service_configs:
        rate_limit_manager.register_service(config)


def before_sleep_callback(retry_state: RetryCallState) -> None:
    """
    Callback executed before sleeping between retries.

    Args:
        retry_state: Current retry state
    """
    exception = retry_state.outcome.exception()
    if exception:
        logger.warning(
            f"Request failed (attempt {retry_state.attempt_number}/"
            f"{retry_state.retry_object.stop.max_attempt_number}), "
            f"retrying in {retry_state.next_action.sleep:.2f} seconds: {exception!s}"
        )


async def with_rate_limit(
    service_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """
    Execute a function with rate limiting.

    Args:
        service_name: Name of the service being called
        func: Function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function
    """
    limiter = rate_limit_manager.get_limiter(service_name)

    # Wait if we're currently rate limited
    if not await limiter.acquire():
        backoff_time = limiter.get_backoff_time()
        logger.warning(
            f"Rate limit reached for {service_name}, waiting {backoff_time:.2f} seconds"
        )
        await asyncio.sleep(backoff_time)

    # Execute with retry logic
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(limiter.config.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=limiter.config.min_wait_seconds,
                max=limiter.config.max_wait_seconds,
            ),
            reraise=True,
            before_sleep=before_sleep_callback,
        ):
            with attempt:
                # Try executing the function
                return await func(*args, **kwargs)
    except Exception as e:
        logger.error(
            f"Request to {service_name} failed after "
            f"{limiter.config.max_retries} attempts: {e!s}"
        )
        raise


def rate_limited(service_name: str) -> Callable[[F], F]:
    """
    Decorator to apply rate limiting to a function.

    Args:
        service_name: Name of the service being called

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await with_rate_limit(service_name, func, *args, **kwargs)

        return cast(F, wrapper)

    return decorator


class APIClient:
    """
    Base client for API requests with rate limiting and retries.

    Provides a foundation for service-specific API clients with built-in
    rate limiting, exponential backoff, and error handling.
    """

    def __init__(self, service_name: str, base_url: str, api_key: str | None = None):
        """
        Initialize the API client.

        Args:
            service_name: Name of the service
            base_url: Base URL for API requests
            api_key: API key for authentication (optional)
        """
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.limiter = rate_limit_manager.get_limiter(service_name)

    async def request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Make an API request with rate limiting and retries.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters (optional)
            json_data: JSON data for request body (optional)
            headers: Additional HTTP headers (optional)

        Returns:
            Parsed JSON response

        Raises:
            APIError: If the request fails after retries
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Prepare headers
        request_headers = {}
        if headers:
            request_headers.update(headers)
        if self.api_key:
            request_headers["Authorization"] = f"Bearer {self.api_key}"

        # Define the actual request function
        async def do_request():
            async with aiohttp.ClientSession() as session:
                request_method = getattr(session, method.lower())

                async with request_method(
                    url, params=params, json=json_data, headers=request_headers
                ) as response:
                    status_code = response.status
                    response_text = await response.text()

                    # Handle rate limiting responses
                    if status_code == HTTP_STATUS_TOO_MANY_REQUESTS:
                        # Extract retry-after header if available
                        retry_after = response.headers.get("Retry-After")
                        wait_time = (
                            int(retry_after)
                            if retry_after and retry_after.isdigit()
                            else 60
                        )

                        logger.warning(
                            f"Rate limited by {self.service_name} API, "
                            f"waiting {wait_time} seconds (Retry-After: {retry_after})"
                        )

                        raise APIError(
                            "Rate limit exceeded",
                            self.service_name,
                            status_code=status_code,
                        )

                    # Handle other error responses
                    if not (HTTP_STATUS_OK <= status_code < HTTP_STATUS_REDIRECT):
                        raise APIError(
                            f"API request failed: {response_text}",
                            self.service_name,
                            status_code=status_code,
                        )

                    # Parse and return successful response
                    try:
                        return await response.json()
                    except aiohttp.ContentTypeError:
                        # Not JSON, return text as is
                        return {"text": response_text}

        # Execute with rate limiting
        return await with_rate_limit(self.service_name, do_request)

    @dataclass
    class RequestConfig:
        """Configuration for a generic HTTP request."""

        url: str
        method: str = "GET"
        params: dict[str, Any] | None = None
        json_data: dict[str, Any] | None = None
        headers: dict[str, str] | None = None
        service_name: str | None = None

    @rate_limited("UNKNOWN")  # Will be replaced with actual service name
    async def generic_request(
        self,
        config: RequestConfig,
    ) -> dict[str, Any]:
        """
        Make a generic HTTP request with rate limiting and retries.

        Args:
            config: Configuration for the HTTP request including URL, method,
                   parameters, JSON data, headers, and optional service name

        Returns:
            Parsed JSON response

        Raises:
            APIError: If the request fails after retries
        """
        # Replace the decorator's service name with the actual service
        actual_service = config.service_name or self.service_name
        self.generic_request.__defaults__ = (actual_service,)  # type: ignore

        # Prepare headers
        request_headers = {}
        if config.headers:
            request_headers.update(config.headers)
        if self.api_key and "Authorization" not in request_headers:
            request_headers["Authorization"] = f"Bearer {self.api_key}"

        async with aiohttp.ClientSession() as session:
            request_method = getattr(session, config.method.lower())

            async with request_method(
                config.url,
                params=config.params,
                json=config.json_data,
                headers=request_headers,
            ) as response:
                status_code = response.status
                response_text = await response.text()

                # Handle error responses
                if not (HTTP_STATUS_OK <= status_code < HTTP_STATUS_REDIRECT):
                    raise APIError(
                        f"API request failed: {response_text}",
                        actual_service,
                        status_code=status_code,
                    )

                # Parse and return successful response
                try:
                    return await response.json()
                except aiohttp.ContentTypeError:
                    # Not JSON, return text as is
                    return {"text": response_text}


# Default rate limit configurations for common external services
DEFAULT_RATE_LIMITS = [
    # OpenAI API has a default tier of 3 requests per minute and 200 per day
    RateLimitConfig(
        service_name="openai",
        requests_per_minute=3,
        requests_per_day=200,
        max_retries=5,
        min_wait_seconds=1.0,
        max_wait_seconds=60.0,
    ),
    # Tavily default is 60 requests per minute and 1000 per day
    RateLimitConfig(
        service_name="tavily",
        requests_per_minute=60,
        requests_per_day=1000,
        max_retries=3,
        min_wait_seconds=1.0,
        max_wait_seconds=30.0,
    ),
    # Firecrawl default is 10 requests per minute and a few hundred per day
    RateLimitConfig(
        service_name="firecrawl",
        requests_per_minute=10,
        requests_per_day=300,
        max_retries=3,
        min_wait_seconds=1.0,
        max_wait_seconds=30.0,
    ),
    # Supabase uses Postgres connection pools, so limit concurrent requests
    RateLimitConfig(
        service_name="supabase",
        requests_per_minute=100,
        requests_per_day=5000,
        max_retries=3,
        min_wait_seconds=0.5,
        max_wait_seconds=15.0,
    ),
]


def initialize_rate_limiting():
    """
    Initialize rate limiting with default configurations.

    Call this at application startup to configure rate limiters
    for common external services.
    """
    configure_rate_limits(DEFAULT_RATE_LIMITS)
    logger.info(f"Initialized rate limiting for {len(DEFAULT_RATE_LIMITS)} services")


# Helper function to update rate limits from configuration
def update_rate_limits_from_config(config_dict: dict[str, dict[str, Any]]) -> None:
    """
    Update rate limits from a configuration dictionary.

    Args:
        config_dict: Dictionary mapping service names to rate limit configurations
    """
    for service_name, service_config in config_dict.items():
        config = RateLimitConfig(
            service_name=service_name,
            requests_per_minute=service_config.get("requests_per_minute", 30),
            requests_per_day=service_config.get("requests_per_day", 1000),
            max_retries=service_config.get("max_retries", 3),
            min_wait_seconds=service_config.get("min_wait_seconds", 1.0),
            max_wait_seconds=service_config.get("max_wait_seconds", 30.0),
            retry_status_codes=service_config.get(
                "retry_status_codes", [429, 500, 502, 503, 504]
            ),
            cooldown_after_quota=service_config.get("cooldown_after_quota", 60),
        )
        rate_limit_manager.register_service(config)
        logger.info(f"Updated rate limits for {service_name} from configuration")
