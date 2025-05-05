"""
Configuration management for the Travel Planner system.

This module handles loading and managing configuration for the entire
travel planning system, including environment variables, API keys,
and default settings for agents and services.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from loguru import logger


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
    user_agent: Optional[str] = Field(default=None, description="Custom user agent string")
    
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
    max_tokens: Optional[int] = Field(default=None, description="Max tokens to generate")
    
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
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key")
    firecrawl_api_key: Optional[str] = Field(default=None, description="Firecrawl API key")
    
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
    
    def validate(self) -> bool:
        """Validate that required API keys are present."""
        missing_keys = []
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        if not self.supabase_url:
            missing_keys.append("SUPABASE_URL")
        if not self.supabase_key:
            missing_keys.append("SUPABASE_KEY")
            
        if missing_keys:
            logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
            return False
        return True


class SystemConfig(BaseModel):
    """System-wide configuration."""
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
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
    agent_models: Dict[str, AgentModelConfig] = field(default_factory=dict)
    
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
    
    def validate(self) -> bool:
        """Validate the entire configuration."""
        return self.api.validate()
    
    def get_agent_model(self, agent_type: str) -> AgentModelConfig:
        """Get model configuration for a specific agent type."""
        return self.agent_models.get(agent_type, self.agent_models.get("default", AgentModelConfig(name="gpt-4o")))


# Global configuration instance
config = TravelPlannerConfig()


def initialize_config() -> TravelPlannerConfig:
    """Initialize and validate the configuration."""
    if not config.validate():
        logger.warning("Configuration validation failed, some features may not work correctly")
    return config