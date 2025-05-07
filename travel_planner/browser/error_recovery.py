"""
Error recovery mechanisms for browser automation in the travel planner system.

This module implements error recovery strategies for browser automation,
making the travel planning process more robust against common web issues.
"""

import asyncio
import os
from datetime import datetime
from functools import wraps
from typing import TypeVar

from stagehand import Page
from stagehand.exceptions import (
    ElementNotFoundError,
    NavigationError,
    TimeoutError,
)

from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class BrowserRecoveryStrategy:
    """Base class for browser error recovery strategies."""

    def __init__(self, max_attempts: int = 3, delay: float = 1.0):
        """
        Initialize recovery strategy.

        Args:
            max_attempts: Maximum number of recovery attempts
            delay: Delay between attempts in seconds
        """
        self.max_attempts = max_attempts
        self.delay = delay

    async def can_handle(self, error: Exception, page: Page | None = None) -> bool:
        """
        Check if this strategy can handle the given error.

        Args:
            error: The error to handle
            page: The browser page where the error occurred

        Returns:
            True if this strategy can handle the error
        """
        raise NotImplementedError("Subclasses must implement can_handle")

    async def handle(self, error: Exception, page: Page | None = None) -> bool:
        """
        Handle the error and attempt recovery.

        Args:
            error: The error to handle
            page: The browser page where the error occurred

        Returns:
            True if recovery was successful
        """
        raise NotImplementedError("Subclasses must implement handle")


class NavigationRecoveryStrategy(BrowserRecoveryStrategy):
    """Strategy for recovering from navigation errors."""

    async def can_handle(self, error: Exception, page: Page | None = None) -> bool:
        """
        Check if this strategy can handle the given error.

        Args:
            error: The error to handle
            page: The browser page where the error occurred

        Returns:
            True if this is a navigation error
        """
        return isinstance(error, NavigationError)

    async def handle(self, error: Exception, page: Page | None = None) -> bool:
        """
        Handle navigation errors.

        Args:
            error: The error to handle
            page: The browser page where the error occurred

        Returns:
            True if recovery was successful
        """
        if not page:
            return False

        logger.info("Attempting to recover from navigation error")

        try:
            # Check if the page is still responsive
            await page.evaluate("() => document.title")

            # If we have a URL, try reloading the page
            current_url = await page.evaluate("() => window.location.href")
            if current_url:
                logger.info(f"Reloading page {current_url}")
                await page.reload(wait_until="networkidle2")
                return True
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e!s}")

        return False


class ElementNotFoundRecoveryStrategy(BrowserRecoveryStrategy):
    """Strategy for recovering from element not found errors."""

    async def can_handle(self, error: Exception, page: Page | None = None) -> bool:
        """
        Check if this strategy can handle the given error.

        Args:
            error: The error to handle
            page: The browser page where the error occurred

        Returns:
            True if this is an element not found error
        """
        return isinstance(error, ElementNotFoundError)

    async def handle(self, error: Exception, page: Page | None = None) -> bool:
        """
        Handle element not found errors.

        Args:
            error: The error to handle
            page: The browser page where the error occurred

        Returns:
            True if recovery was successful
        """
        if not page:
            return False

        logger.info("Attempting to recover from element not found error")

        try:
            # Element might not be loaded yet, wait a bit and try again
            await asyncio.sleep(2)

            # Try scrolling down slightly to ensure the element is in view
            await page.evaluate("window.scrollBy(0, 100)")

            # Element might be in a different frame, try checking frames
            frames = page.frames()
            if len(frames) > 1:
                logger.info(
                    f"Page has {len(frames)} frames, checking for elements in frames"
                )

            return True
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e!s}")

        return False


class TimeoutRecoveryStrategy(BrowserRecoveryStrategy):
    """Strategy for recovering from timeout errors."""

    async def can_handle(self, error: Exception, page: Page | None = None) -> bool:
        """
        Check if this strategy can handle the given error.

        Args:
            error: The error to handle
            page: The browser page where the error occurred

        Returns:
            True if this is a timeout error
        """
        return isinstance(error, TimeoutError) or (
            isinstance(error, asyncio.TimeoutError)
        )

    async def handle(self, error: Exception, page: Page | None = None) -> bool:
        """
        Handle timeout errors.

        Args:
            error: The error to handle
            page: The browser page where the error occurred

        Returns:
            True if recovery was successful
        """
        if not page:
            return False

        logger.info("Attempting to recover from timeout error")

        try:
            # Take a screenshot for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "error_screenshots"
            )
            os.makedirs(screenshot_dir, exist_ok=True)
            screenshot_path = os.path.join(screenshot_dir, f"timeout_{timestamp}.png")
            await page.screenshot(path=screenshot_path, full_page=True)
            logger.info(f"Timeout screenshot saved to {screenshot_path}")

            # Wait a bit longer and try again
            await asyncio.sleep(5)

            # Reload the page
            await page.reload(wait_until="networkidle2")

            # Wait for potentially slow elements
            await page.wait_for_function(
                """() => {
                    return document.readyState === 'complete' && 
                        !document.querySelector('.loading') &&
                        !document.querySelector('[data-loading="true"]');
                }""",
                timeout=10000,
            )

            return True
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e!s}")

        return False


class BrowserRecoveryManager:
    """Manager for browser error recovery strategies."""

    def __init__(self):
        """Initialize the recovery manager with default strategies."""
        self.strategies: list[BrowserRecoveryStrategy] = [
            NavigationRecoveryStrategy(),
            ElementNotFoundRecoveryStrategy(),
            TimeoutRecoveryStrategy(),
        ]

    def add_strategy(self, strategy: BrowserRecoveryStrategy):
        """
        Add a recovery strategy.

        Args:
            strategy: Strategy to add
        """
        self.strategies.append(strategy)

    async def handle_error(
        self, error: Exception, page: Page | None = None, max_attempts: int = 3
    ) -> bool:
        """
        Attempt to handle an error using available strategies.

        Args:
            error: The error to handle
            page: The browser page where the error occurred
            max_attempts: Maximum number of recovery attempts

        Returns:
            True if recovery was successful
        """
        logger.error(f"Browser error: {error!s}")

        for strategy in self.strategies:
            if await strategy.can_handle(error, page):
                for attempt in range(max_attempts):
                    logger.info(f"Recovery attempt {attempt + 1}/{max_attempts}")

                    if await strategy.handle(error, page):
                        logger.info("Recovery successful")
                        return True

                    await asyncio.sleep(1 * (attempt + 1))

        logger.error("All recovery attempts failed")
        return False


# Singleton recovery manager instance
recovery_manager = BrowserRecoveryManager()


def with_recovery(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator for browser automation functions with error recovery.

    Args:
        max_attempts: Maximum number of recovery attempts
        delay: Delay between attempts in seconds

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            page = None

            # Try to find the page object in args or kwargs
            for arg in args:
                if isinstance(arg, Page):
                    page = arg
                    break

            for _key, value in kwargs.items():
                if isinstance(value, Page):
                    page = value
                    break

            # If function is a method, try to get page from self
            if len(args) > 0 and hasattr(args[0], "page"):
                page = args[0].page

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.error(
                        f"Error in {func.__name__}, attempt {attempt + 1}/{max_attempts}: {e!s}"
                    )

                    # Attempt recovery
                    if await recovery_manager.handle_error(e, page):
                        logger.info(
                            f"Retrying {func.__name__} after successful recovery"
                        )
                    else:
                        logger.error(f"Recovery failed for {func.__name__}")
                        if attempt == max_attempts - 1:
                            # Last attempt failed
                            raise

                    # Wait before retrying
                    await asyncio.sleep(delay * (attempt + 1))

            # If we get here, all attempts failed
            if last_error:
                raise last_error

        return wrapper

    return decorator
