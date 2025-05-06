"""
Logging framework for the Travel Planner system.

This module configures logging for the Travel Planner application,
providing a consistent logging interface across all modules.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any

from loguru import logger

from travel_planner.config import LogLevel


def get_logger(name: str):
    """
    Get a logger instance for the specified module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance with the module name attached
    """
    return logger.bind(name=name)


def setup_logging(
    log_level: LogLevel | str = LogLevel.INFO, log_file: str | None = None
):
    """
    Set up the logging configuration for the application.

    Args:
        log_level: The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
    """
    # Convert string to enum if necessary
    if isinstance(log_level, str):
        log_level = LogLevel(log_level.upper())

    # Remove default logger
    logger.remove()

    # Add console logger
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_level.value,
        colorize=True,
    )

    # Add file logger if specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        logger.add(
            log_file,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{name}:{function}:{line} - {message}"
            ),
            level=log_level.value,
            rotation="10 MB",
            compression="zip",
        )

    logger.info(f"Logging initialized with level {log_level.value}")


class AgentLogger:
    """
    Logger specialized for agent operations, providing context-aware logging
    with agent-specific information.
    """

    def __init__(self, agent_name: str, agent_id: str | None = None):
        """
        Initialize the agent logger.

        Args:
            agent_name: Name of the agent
            agent_id: Unique ID for the agent instance (optional)
        """
        self.agent_name = agent_name
        self.agent_id = (
            agent_id or f"{agent_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        self.logger = logger.bind(agent_name=agent_name, agent_id=agent_id)

    def debug(self, message: str, **kwargs):
        """Log a debug message with agent context."""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log an info message with agent context."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log a warning message with agent context."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log an error message with agent context."""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log a critical message with agent context."""
        self.logger.critical(message, **kwargs)

    def log_api_request(
        self, api_name: str, endpoint: str, params: dict[str, Any] | None = None
    ):
        """
        Log an API request.

        Args:
            api_name: Name of the API being called
            endpoint: API endpoint
            params: Request parameters (optional)
        """
        self.debug(
            f"API Request: {api_name} - {endpoint}",
            api_name=api_name,
            endpoint=endpoint,
            params=self._safe_json(params),
        )

    def log_api_response(
        self,
        api_name: str,
        endpoint: str,
        status_code: int,
        response_data: Any | None = None,
    ):
        """
        Log an API response.

        Args:
            api_name: Name of the API being called
            endpoint: API endpoint
            status_code: HTTP status code
            response_data: Response data (optional)
        """
        self.debug(
            f"API Response: {api_name} - {endpoint} - Status: {status_code}",
            api_name=api_name,
            endpoint=endpoint,
            status_code=status_code,
            response=self._safe_json(response_data),
        )

    def log_llm_input(
        self, model: str, messages: list[dict[str, Any]], temperature: float
    ):
        """
        Log input to a language model.

        Args:
            model: Name of the model
            messages: Input messages
            temperature: Temperature setting
        """
        self.debug(
            f"LLM Request: {model} - Temperature: {temperature}",
            model=model,
            temperature=temperature,
            messages=self._safe_json(messages),
        )

    def log_llm_output(self, model: str, response: Any):
        """
        Log output from a language model.

        Args:
            model: Name of the model
            response: Model response
        """
        self.debug(
            f"LLM Response: {model}",
            model=model,
            response=self._safe_json(response),
        )

    def log_agent_state(self, state: dict[str, Any]):
        """
        Log the current state of the agent.

        Args:
            state: Current agent state
        """
        self.debug(
            f"Agent State: {self.agent_name}",
            state=self._safe_json(state),
        )

    def _safe_json(self, obj: Any) -> str | None:
        """
        Safely convert an object to JSON, handling conversion errors.

        Args:
            obj: Object to convert to JSON

        Returns:
            JSON string or None if conversion fails
        """
        if obj is None:
            return None

        try:
            return json.dumps(obj, default=str)
        except Exception as e:
            self.warning(f"Failed to serialize object to JSON: {e!s}")
            return str(obj)
