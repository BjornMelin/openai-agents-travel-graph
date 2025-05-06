"""
Configuration management for the Travel Planner system.

This module handles loading and managing configuration for the entire
travel planning system, including environment variables, API keys,
and default settings for agents and services.
"""

import os
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, field_validator

# Load environment variables from .env file
load_dotenv()


class LogLevel(str, Enum):
    """Log levels supported by the system."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class BrowserConfig(BaseModel):
    """Configuration for browser automation."""

    headless: bool = Field(default=True, description="Run browser in headless mode")
    cache_ttl: int = Field(default=3600, description="Cache time to live in seconds")
    timeout: int = Field(default=30000, description="Default timeout in milliseconds")
    user_agent: str | None = Field(default=None, description="Custom user agent string")

    @classmethod
    def from_env(cls) -> "BrowserConfig":
        """Create a BrowserConfig from environment variables."""
        return cls(
            headless=os.getenv("HEADLESS", "true").lower() == "true",
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
            timeout=int(os.getenv("BROWSER_TIMEOUT", "30000")),
            user_agent=os.getenv("USER_AGENT"),
        )


class AgentModelConfig(BaseModel):
    """Configuration for an agent's LLM model."""

    name: str = Field(..., description="Model name to use")
    temperature: float = Field(default=0.7, description="Model temperature")
    max_tokens: int | None = Field(default=None, description="Max tokens to generate")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Validate temperature is within reasonable bounds."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Temperature must be between 0.0 and 1.0, got {value}")
        return value

    @classmethod
    def from_env(cls, prefix: str = "") -> "AgentModelConfig":
        """Create an AgentModelConfig from environment variables."""
        prefix = f"{prefix}_" if prefix else ""
        return cls(
            name=os.getenv(f"{prefix}MODEL", "gpt-4o"),
            temperature=float(os.getenv(f"{prefix}TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv(f"{prefix}MAX_TOKENS", "0")) or None,
        )


class APIConfig(BaseModel):
    """Configuration for external APIs."""

    openai_api_key: str = Field(..., description="OpenAI API key")
    supabase_url: str = Field(..., description="Supabase URL")
    supabase_key: str = Field(..., description="Supabase API key")
    tavily_api_key: str | None = Field(default=None, description="Tavily API key")
    firecrawl_api_key: str | None = Field(default=None, description="Firecrawl API key")

    class ValidationError(Exception):
        """Exception raised for API configuration validation errors."""

        def __init__(
            self, missing_keys: list[str], optional_missing: list[str] | None = None
        ):
            self.missing_keys = missing_keys
            self.optional_missing = optional_missing or []
            message = f"Missing required API keys: {', '.join(missing_keys)}"
            if optional_missing:
                message += f". Optional keys missing: {', '.join(optional_missing)}"
            super().__init__(message)

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create an APIConfig from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_key=os.getenv("SUPABASE_KEY", ""),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
        )

    def validate(self, raise_error: bool = False) -> bool:
        """
        Validate that required API keys are present.

        Args:
            raise_error: If True, raise ValidationError instead of returning False

        Returns:
            True if all required keys are present, False otherwise

        Raises:
            ValidationError: If raise_error is True and validation fails
        """
        missing_keys = []
        optional_missing = []

        # Required keys
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        if not self.supabase_url:
            missing_keys.append("SUPABASE_URL")
        if not self.supabase_key:
            missing_keys.append("SUPABASE_KEY")

        # Optional keys
        if not self.tavily_api_key:
            optional_missing.append("TAVILY_API_KEY")
        if not self.firecrawl_api_key:
            optional_missing.append("FIRECRAWL_API_KEY")

        if missing_keys:
            error_msg = f"Missing required API keys: {', '.join(missing_keys)}"
            if optional_missing:
                logger.warning(
                    f"Optional API keys missing: {', '.join(optional_missing)}. "
                    f"Some features may be limited."
                )

            logger.error(error_msg)

            if raise_error:
                raise self.ValidationError(missing_keys, optional_missing)
            return False

        if optional_missing:
            logger.warning(
                f"Optional API keys missing: {', '.join(optional_missing)}. "
                f"Some features may be limited."
            )

        return True


class SystemConfig(BaseModel):
    """System-wide configuration."""

    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)",
    )
    max_concurrency: int = Field(default=3, description="Maximum concurrent operations")
    default_budget: float = Field(default=2000, description="Default budget amount")
    default_currency: str = Field(default="USD", description="Default currency")

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Create a SystemConfig from environment variables."""
        return cls(
            log_level=LogLevel(os.getenv("LOG_LEVEL", "INFO")),
            environment=os.getenv("ENVIRONMENT", "development"),
            max_concurrency=int(os.getenv("MAX_CONCURRENCY", "3")),
            default_budget=float(os.getenv("DEFAULT_BUDGET", "2000")),
            default_currency=os.getenv("DEFAULT_CURRENCY", "USD"),
        )


@dataclass
class TravelPlannerConfig:
    """Main configuration class for the Travel Planner system."""

    api: APIConfig = field(default_factory=APIConfig.from_env)
    system: SystemConfig = field(default_factory=SystemConfig.from_env)
    browser: BrowserConfig = field(default_factory=BrowserConfig.from_env)
    agent_models: dict[str, AgentModelConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize agent models if not provided."""
        if not self.agent_models:
            self.agent_models = {
                "orchestrator": AgentModelConfig.from_env("ORCHESTRATOR"),
                "destination": AgentModelConfig.from_env("DESTINATION"),
                "flight": AgentModelConfig.from_env("FLIGHT"),
                "accommodation": AgentModelConfig.from_env("ACCOMMODATION"),
                "transportation": AgentModelConfig.from_env("TRANSPORTATION"),
                "activity": AgentModelConfig.from_env("ACTIVITY"),
                "budget": AgentModelConfig.from_env("BUDGET"),
            }

    class ConfigurationError(Exception):
        """Exception raised for configuration validation errors."""

        pass

    def validate(self, raise_error: bool = False) -> bool:
        """
        Validate the entire configuration.

        Args:
            raise_error: If True, raise ConfigurationError instead of returning False

        Returns:
            True if configuration is valid, False otherwise

        Raises:
            ConfigurationError: If raise_error is True and validation fails
        """
        try:
            # Validate API configuration
            self.api.validate(raise_error=True)

            # Validate browser configuration
            if self.browser.timeout <= 0:
                raise ValueError("Browser timeout must be positive")

            # Validate system configuration
            if self.system.max_concurrency <= 0:
                raise ValueError("Max concurrency must be positive")

            return True

        except Exception as e:
            if isinstance(e, self.api.ValidationError):
                # Already logged in APIConfig.validate
                pass
            else:
                logger.error(f"Configuration validation failed: {e!s}")

            if raise_error:
                raise self.ConfigurationError(
                    f"Configuration validation failed: {e!s}"
                ) from e

            return False

    def get_agent_model(self, agent_type: str) -> AgentModelConfig:
        """
        Get model configuration for a specific agent type.

        Args:
            agent_type: Type of agent to get model config for

        Returns:
            AgentModelConfig for the requested agent type, or a default if not found
        """
        return self.agent_models.get(
            agent_type,
            self.agent_models.get("default", AgentModelConfig(name="gpt-4o")),
        )


# Global configuration instance
config = TravelPlannerConfig()


def initialize_config(
    custom_config_path: str | None = None,
    validate: bool = True,
    raise_on_error: bool = False,
) -> TravelPlannerConfig:
    """
    Initialize and validate the configuration.

    Args:
        custom_config_path: Path to a custom .env file to load
        validate: Whether to validate the configuration
        raise_on_error: Whether to raise an exception on validation failure

    Returns:
        Initialized and validated configuration object

    Raises:
        TravelPlannerConfig.ConfigurationError: If validation fails and
            raise_on_error is True
        FileNotFoundError: If custom_config_path is provided but does not exist
    """
    # Load custom .env file if provided
    if custom_config_path:
        if not os.path.exists(custom_config_path):
            error_msg = f"Custom configuration file not found: {custom_config_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading custom configuration from {custom_config_path}")
        load_dotenv(custom_config_path, override=True)

        # Reload configuration into the existing global 'config' object
        # This ensures that other modules importing 'config' see the updates
        # without needing to re-import or handle a new object.
        config.api = APIConfig.from_env()
        config.system = SystemConfig.from_env()
        config.browser = BrowserConfig.from_env()
        # Reset agent_models so __post_init__ re-populates them based on new env vars
        config.agent_models = {}
        config.__post_init__()

    if validate:
        is_valid = config.validate(raise_error=raise_on_error)
        if not is_valid:
            logger.warning(
                "Configuration validation failed. The application may not function "
                "correctly. Please check your environment variables and ensure all "
                "required API keys are set."
            )

            # Print instructions for setting up environment variables
            logger.info(
                "Required environment variables: OPENAI_API_KEY, "
                "SUPABASE_URL, SUPABASE_KEY"
            )
            logger.info(
                "Optional environment variables: TAVILY_API_KEY, FIRECRAWL_API_KEY"
            )
            logger.info(
                "You can set these in a .env file in the project root, "
                "or as environment variables in your shell."
            )

    return config
