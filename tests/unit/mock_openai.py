"""
Mock implementations for OpenAI to support testing.

This module provides mock classes and functions to simulate OpenAI functionality
in tests without requiring the actual OpenAI package. It should be imported
before any modules that depend on OpenAI.
"""

import os
import sys
from unittest.mock import MagicMock


class MockAsyncOpenAI(MagicMock):
    """Mock for the AsyncOpenAI client."""

    def __init__(self, **kwargs):
        super().__init__()
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock()

        # Mock response structure
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "This is a mock response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # Set up async mock
        async_response = MagicMock()
        async_response.__aenter__ = MagicMock(return_value=mock_response)
        async_response.__aexit__ = MagicMock(return_value=None)

        self.chat.completions.create.return_value = async_response


class MockOpenAI(MagicMock):
    """Mock for the OpenAI client."""

    def __init__(self, api_key=None, **kwargs):
        """
        Initialize mock OpenAI client, ignoring API key requirements
        and all other parameters.
        """
        super().__init__()
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock()

        # Mock response structure
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "This is a mock response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        self.chat.completions.create.return_value = mock_response


# Replace OpenAI imports
def setup_mock_openai():
    """Set up the mock OpenAI modules and classes."""
    # Set up environments for mocking
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "sk-mock-api-key-for-testing"

    # Create a mock OpenAI module
    mock_openai = MagicMock()
    mock_openai.OpenAI = MockOpenAI
    mock_openai.AsyncOpenAI = MockAsyncOpenAI

    # Create additional exceptions
    class OpenAIError(Exception):
        """Mock for OpenAI's base error."""

        pass

    mock_openai.OpenAIError = OpenAIError

    # Register in sys.modules to override any imports
    sys.modules["openai"] = mock_openai

    return mock_openai
