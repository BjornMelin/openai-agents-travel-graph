"""
Pytest configuration for the Travel Planner system tests.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

# Register asyncio marker
pytest.importorskip("pytest_asyncio")
pytest.mark.asyncio = pytest.mark.asyncio

# Import project modules after configuring pytest
from travel_planner.agents.base import AgentConfig  # noqa: E402
from travel_planner.config import (  # noqa: E402
    APIConfig,
    SystemConfig,
    TravelPlannerConfig,
)
from travel_planner.utils import LogLevel, setup_logging  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Set up logging for tests."""
    setup_logging(LogLevel.DEBUG)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()

    # Mock the chat completions create method
    mock_client.chat.completions.create = AsyncMock()

    # Set up a basic response structure
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Test response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_response

    return mock_client


@pytest.fixture
def test_agent_config():
    """Test agent configuration."""
    return AgentConfig(
        name="Test Agent",
        instructions="You are a test agent",
        model="gpt-4o",
        temperature=0.5,
    )


@pytest.fixture
def test_config():
    """Test application configuration."""
    return TravelPlannerConfig(
        api=APIConfig(
            openai_api_key="test-key",
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            tavily_api_key="test-key",
            firecrawl_api_key="test-key",
        ),
        system=SystemConfig(
            log_level=LogLevel.DEBUG,
            environment="test",
            max_concurrency=2,
            default_budget=1000,
            default_currency="USD",
        ),
    )
