"""
Unit tests for the base agent class.
"""

# Import mock OpenAI before anything else
from tests.unit.mock_openai import setup_mock_openai

# Set up mock OpenAI client
setup_mock_openai()

import pytest  # noqa: E402

from travel_planner.agents.base import (  # noqa: E402
    AgentConfig,
    BaseAgent,
    InvalidConfigurationException,
)

# Constants for test assertions
DEFAULT_TEMPERATURE = 0.7
SINGLE_MESSAGE_COUNT = 2
MULTI_MESSAGE_COUNT = 4


def test_agent_initialization():
    """Test that agent initializes with correct configuration."""
    config = AgentConfig(
        name="Test Agent",
        instructions="Test instructions",
    )
    agent = BaseAgent(config)

    assert agent.name == "Test Agent"
    assert agent.instructions == "Test instructions"
    assert agent.config.model == "gpt-4o"  # Default model
    assert agent.config.temperature == DEFAULT_TEMPERATURE  # Default temperature


def test_agent_initialization_invalid_config():
    """Test that agent initialization with invalid config raises exception."""
    # Missing name
    with pytest.raises(InvalidConfigurationException):
        config = AgentConfig(
            name="",
            instructions="Test instructions",
        )
        agent = BaseAgent(config)
        agent._validate_config()

    # Missing instructions
    with pytest.raises(InvalidConfigurationException):
        config = AgentConfig(
            name="Test Agent",
            instructions="",
        )
        agent = BaseAgent(config)
        agent._validate_config()


def test_prepare_messages_string_input():
    """Test that _prepare_messages correctly formats string input."""
    config = AgentConfig(
        name="Test Agent",
        instructions="Test instructions",
    )
    agent = BaseAgent(config)

    messages = agent._prepare_messages("Hello")

    assert len(messages) == SINGLE_MESSAGE_COUNT
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Test instructions"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"


def test_prepare_messages_list_input():
    """Test that _prepare_messages correctly formats list input."""
    config = AgentConfig(
        name="Test Agent",
        instructions="Test instructions",
    )
    agent = BaseAgent(config)

    input_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]

    messages = agent._prepare_messages(input_messages)

    assert len(messages) == MULTI_MESSAGE_COUNT  # Input messages + system message
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Test instructions"
    assert messages[1:] == input_messages


def test_prepare_messages_with_existing_system():
    """Test that _prepare_messages preserves existing system message."""
    config = AgentConfig(
        name="Test Agent",
        instructions="Test instructions",
    )
    agent = BaseAgent(config)

    input_messages = [
        {"role": "system", "content": "Existing instructions"},
        {"role": "user", "content": "Hello"},
    ]

    messages = agent._prepare_messages(input_messages)

    assert len(messages) == SINGLE_MESSAGE_COUNT
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Existing instructions"  # Preserved
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"


@pytest.mark.asyncio
async def test_run_method_not_implemented():
    """Test that run method raises NotImplementedError."""
    config = AgentConfig(
        name="Test Agent",
        instructions="Test instructions",
    )
    agent = BaseAgent(config)

    with pytest.raises(NotImplementedError):
        await agent.run("Hello")


@pytest.mark.asyncio
async def test_process_method_not_implemented():
    """Test that process method raises NotImplementedError."""
    config = AgentConfig(
        name="Test Agent",
        instructions="Test instructions",
    )
    agent = BaseAgent(config)

    with pytest.raises(NotImplementedError):
        await agent.process("Hello", None)
