"""
Error handling utilities for the Travel Planner system.

This module provides decorators, helper functions, and custom exception
classes to handle errors consistently across the application.
"""

import functools
import traceback
from collections.abc import Callable
from typing import Any, TypeVar, cast

from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Type variables for function decorator typing
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class TravelPlannerError(Exception):
    """Base exception class for all Travel Planner errors."""

    def __init__(self, message: str, original_error: Exception | None = None):
        """
        Initialize a TravelPlannerError.

        Args:
            message: Error message
            original_error: The original exception that caused this error (optional)
        """
        self.original_error = original_error
        if original_error:
            message = f"{message} - Original error: {original_error!s}"
        super().__init__(message)


class APIError(TravelPlannerError):
    """Error raised when an external API request fails."""

    def __init__(
        self,
        message: str,
        service_name: str,
        status_code: int | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize an APIError.

        Args:
            message: Error message
            service_name: Name of the API service
            status_code: HTTP status code (optional)
            original_error: The original exception that caused this error (optional)
        """
        self.service_name = service_name
        self.status_code = status_code
        status_str = f" (status: {status_code})" if status_code else ""
        full_message = f"Error in {service_name} API{status_str}: {message}"
        super().__init__(full_message, original_error)


class AgentExecutionError(TravelPlannerError):
    """Error raised when an agent fails to execute correctly."""

    def __init__(
        self, message: str, agent_name: str, original_error: Exception | None = None
    ):
        """
        Initialize an AgentExecutionError.

        Args:
            message: Error message
            agent_name: Name of the agent that failed
            original_error: The original exception that caused this error (optional)
        """
        self.agent_name = agent_name
        full_message = f"Error executing agent '{agent_name}': {message}"
        super().__init__(full_message, original_error)


class ValidationError(TravelPlannerError):
    """Error raised when validation of input or data fails."""

    pass


class ResourceNotFoundError(TravelPlannerError):
    """Error raised when a requested resource is not found."""

    pass


def handle_errors(
    default_value: T | None = None, error_cls: type[Exception] = TravelPlannerError
) -> Callable[[F], F]:
    """
    Decorator to catch and handle exceptions, logging them and
    optionally returning a default value.

    Args:
        default_value: Value to return if an exception occurs (optional)
        error_cls: Exception type to re-raise (default: TravelPlannerError)

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_name = func.__name__
                logger.error(f"Error in {func_name}: {e!s}")
                logger.debug(f"Traceback: {traceback.format_exc()}")

                if default_value is not None:
                    logger.info(f"Returning default value from {func_name}")
                    return default_value

                # Re-raise as specified error class
                raise error_cls(str(e), original_error=e) from e

        return cast(F, wrapper)

    return decorator


def with_retry(
    max_attempts: int = 3,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 10.0,
    retry_exceptions: tuple = (APIError,),
) -> Callable[[F], F]:
    """
    Decorator to retry a function with exponential backoff when specific
    exceptions occur.

    Args:
        max_attempts: Maximum number of attempts
        min_wait_seconds: Minimum wait time between retries
        max_wait_seconds: Maximum wait time between retries
        retry_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create retry logic using tenacity
            @retry(
                retry=retry_if_exception_type(retry_exceptions),
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(
                    multiplier=1, min=min_wait_seconds, max=max_wait_seconds
                ),
                reraise=True,
            )
            def retry_func() -> Any:
                return func(*args, **kwargs)

            try:
                return retry_func()
            except RetryError as e:
                # Extract the last exception
                last_attempt = e.last_attempt
                if (
                    hasattr(last_attempt, "exception")
                    and last_attempt.exception() is not None
                ):
                    original_error = last_attempt.exception()
                    func_name = func.__name__
                    logger.error(
                        f"All retry attempts failed for {func_name}: {original_error!s}"
                    )
                    raise TravelPlannerError(
                        f"Function {func_name} failed after {max_attempts} attempts. "
                        f"Original error: {original_error!s}",
                        original_error=original_error,
                    ) from e
                # If no specific original error was found in the last attempt,
                # re-raise the RetryError, chaining the original RetryError.
                raise TravelPlannerError(
                    f"Function {func_name} failed after {max_attempts} attempts "
                    f"due to RetryError"
                ) from e

        return cast(F, wrapper)

    return decorator


def safe_execute(
    func: Callable[..., T], *args: Any, default: T | None = None, **kwargs: Any
) -> T | None:
    """
    Execute a function safely, catching any exceptions and
    optionally returning a default value.

    Args:
        func: Function to execute
        *args: Positional arguments to pass to the function
        default: Default value to return if an exception occurs (optional)
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function or default value if an exception occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        func_name = getattr(func, "__name__", str(func))
        logger.error(f"Error executing {func_name}: {e!s}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return default
