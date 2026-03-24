"""Application configuration using Pydantic Settings."""

import pathlib
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """API and CORS configuration."""

    API_PREFIX: str = "/api"
    ALLOWED_ORIGINS: dict[str, list[str]] = {
        "development": ["http://localhost:3002"],
        "production": [],
    }


class AgentSettings(BaseSettings):
    """Pydantic AI agent configuration."""

    AGENT_MODEL: str = "gpt-4o-mini"


class PostgresSettings(BaseSettings):
    """PostgreSQL connection configuration."""

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int


class Settings(
    ApiSettings,
    AgentSettings,
    PostgresSettings,
):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=str(pathlib.Path(__file__).parent.parent.parent.parent.parent / ".env"),
        env_ignore_empty=True,
        extra="ignore",
    )

    ENVIRONMENT: Literal["development", "production"] = "development"
    SAMPLES_DIR: str = str(pathlib.Path(__file__).parent.parent.parent.parent.parent / "data" / "samples")
    TRANSFORM_CACHE_DIR: str = str(pathlib.Path(__file__).parent.parent.parent.parent.parent / "data" / "transforms")
    SPLICE_DIR: str | None = None

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.model_validate({})
