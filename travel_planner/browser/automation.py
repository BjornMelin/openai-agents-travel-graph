"""
Browser automation for the travel planner system.

This module implements browser automation using Stagehand, which combines
deterministic workflows with LLM-powered automation for robust web interactions.
"""

import asyncio
import os
from typing import Any

from stagehand import Browser, Page, PromptTemplate, create_llm
from stagehand.cache import FileCache
from stagehand.exceptions import BrowserError, ElementNotFoundError, NavigationError

from travel_planner.utils.error_handling import with_retry
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)

# Configure cache for browser actions to improve efficiency
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(cache_dir, exist_ok=True)
action_cache = FileCache(cache_dir)


class BrowserAutomation:
    """Base class for browser automation with Stagehand."""

    def __init__(
        self,
        headless: bool = True,
        slow_mo: int = 50,
        timeout: int = 30000,
        use_cache: bool = True,
    ):
        """
        Initialize browser automation.

        Args:
            headless: Whether to run the browser in headless mode
            slow_mo: Slow down operations by this many milliseconds
            timeout: Default timeout for operations in milliseconds
            use_cache: Whether to use action caching
        """
        self.headless = headless
        self.slow_mo = slow_mo
        self.timeout = timeout
        self.use_cache = use_cache
        self.browser: Browser | None = None
        self.page: Page | None = None

        # Set up LLM for Stagehand
        self.llm = create_llm()

    async def __aenter__(self):
        """Set up browser when used as a context manager."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up browser when exiting context manager."""
        await self.teardown()

    async def setup(self):
        """Set up the browser and page."""
        if self.browser is None:
            self.browser = await Browser.create(
                headless=self.headless, slowMo=self.slow_mo
            )

        if self.page is None:
            self.page = await self.browser.new_page()
            await self.page.set_default_timeout(self.timeout)

            # Set up caching if enabled
            if self.use_cache:
                self.page.use_cache(action_cache)

    async def teardown(self):
        """Clean up the browser and page."""
        if self.page:
            await self.page.close()
            self.page = None

        if self.browser:
            await self.browser.close()
            self.browser = None

    @with_retry(retries=3, delay=1, backoff=2)
    async def navigate(self, url: str) -> bool:
        """
        Navigate to a URL with retry logic.

        Args:
            url: URL to navigate to

        Returns:
            True if navigation was successful
        """
        if not self.page:
            await self.setup()

        try:
            logger.info(f"Navigating to {url}")
            await self.page.goto(url, wait_until="networkidle2")
            return True
        except NavigationError as e:
            logger.error(f"Navigation error: {e!s}")
            raise

    async def screenshot(self, path: str):
        """
        Take a screenshot of the current page.

        Args:
            path: Path to save the screenshot
        """
        if not self.page:
            raise BrowserError("Browser not initialized")

        await self.page.screenshot(path=path, full_page=True)

    async def execute_ai_action(
        self, prompt_template: str, context: dict[str, Any], retry_count: int = 3
    ) -> dict[str, Any] | None:
        """
        Execute an action using AI assistance.

        Args:
            prompt_template: Template for the prompt to send to the LLM
            context: Context variables for the template
            retry_count: Number of times to retry on failure

        Returns:
            Result from the AI action
        """
        if not self.page:
            await self.setup()

        prompt = PromptTemplate(prompt_template)

        for attempt in range(retry_count):
            try:
                result = await self.page.ai_action(prompt, context, llm=self.llm)
                return result
            except Exception as e:
                logger.error(
                    f"AI action error (attempt {attempt + 1}/{retry_count}): {e!s}"
                )
                if attempt == retry_count - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))

        return None

    async def extract_data(
        self,
        selector: str | None = None,
        extraction_prompt: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str | dict[str, Any] | list[dict[str, Any]]:
        """
        Extract data from the page using selectors or AI.

        Args:
            selector: CSS selector to extract data from
            extraction_prompt: Prompt for AI extraction
            context: Context for AI extraction

        Returns:
            Extracted data
        """
        if not self.page:
            await self.setup()

        if selector:
            # Use deterministic extraction with selector
            element = await self.page.query_selector(selector)
            if not element:
                raise ElementNotFoundError(f"Element not found: {selector}")
            return await element.text_content()
        elif extraction_prompt:
            # Use AI-based extraction
            context = context or {}
            prompt = PromptTemplate(extraction_prompt)
            result = await self.page.ai_extract(prompt, context, llm=self.llm)
            return result
        else:
            raise ValueError("Either selector or extraction_prompt must be provided")

    async def click(
        self,
        selector: str | None = None,
        text: str | None = None,
        ai_description: str | None = None,
    ):
        """
        Click an element using selector, text content, or AI assistance.

        Args:
            selector: CSS selector to click
            text: Text content to click
            ai_description: Description of the element to click using AI
        """
        if not self.page:
            await self.setup()

        try:
            if selector:
                # Use deterministic click with selector
                await self.page.click(selector)
            elif text:
                # Use text-based click
                await self.page.click_text(text)
            elif ai_description:
                # Use AI to find and click element
                prompt = PromptTemplate(
                    "Find and click the element described as: {{description}}"
                )
                await self.page.ai_action(
                    prompt, {"description": ai_description}, llm=self.llm
                )
            else:
                raise ValueError(
                    "One of selector, text, or ai_description must be provided"
                )
        except Exception as e:
            logger.error(f"Click error: {e!s}")
            raise

    async def fill_form(self, form_data: dict[str, str], use_ai: bool = False):
        """
        Fill a form with provided data.

        Args:
            form_data: Dictionary of field selectors/descriptions and values
            use_ai: Whether to use AI for filling the form
        """
        if not self.page:
            await self.setup()

        if use_ai:
            # Use AI to fill the form
            prompt = PromptTemplate("""
            Fill out the form with the following information:
            {{form_data}}
            
            Look for appropriate input fields and fill them with the corresponding values.
            """)
            await self.page.ai_action(
                prompt, {"form_data": str(form_data)}, llm=self.llm
            )
        else:
            # Use deterministic form filling
            for selector, value in form_data.items():
                try:
                    await self.page.fill(selector, value)
                except Exception as e:
                    logger.error(f"Error filling field {selector}: {e!s}")
                    raise

    async def wait_for_navigation(self, timeout: int | None = None):
        """
        Wait for navigation to complete.

        Args:
            timeout: Maximum time to wait in milliseconds
        """
        if not self.page:
            await self.setup()

        timeout = timeout or self.timeout
        await self.page.wait_for_navigation(timeout=timeout)

    async def wait_for_selector(self, selector: str, timeout: int | None = None):
        """
        Wait for an element to appear.

        Args:
            selector: CSS selector to wait for
            timeout: Maximum time to wait in milliseconds
        """
        if not self.page:
            await self.setup()

        timeout = timeout or self.timeout
        await self.page.wait_for_selector(selector, timeout=timeout)

    async def wait_for_function(self, js_function: str, timeout: int | None = None):
        """
        Wait for a JavaScript function to return true.

        Args:
            js_function: JavaScript function code
            timeout: Maximum time to wait in milliseconds
        """
        if not self.page:
            await self.setup()

        timeout = timeout or self.timeout
        await self.page.wait_for_function(js_function, timeout=timeout)
