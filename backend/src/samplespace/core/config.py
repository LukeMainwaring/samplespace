import pathlib
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    API_PREFIX: str = "/api"
    ALLOWED_ORIGINS: dict[str, list[str]] = {
        "development": ["http://localhost:3002"],
        "production": [],
    }


class AgentSettings(BaseSettings):
    AGENT_MODEL: str = "gpt-5.4-mini"


class PostgresSettings(BaseSettings):
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
    model_config = SettingsConfigDict(
        env_file=str(pathlib.Path(__file__).parent.parent.parent.parent.parent / ".env"),
        env_ignore_empty=True,
        extra="ignore",
    )

    ENVIRONMENT: Literal["development", "production"] = "development"
    SAMPLES_DIR: str = str(pathlib.Path(__file__).parent.parent.parent.parent.parent / "data" / "samples")
    TRANSFORM_CACHE_DIR: str = str(pathlib.Path(__file__).parent.parent.parent.parent.parent / "data" / "transforms")
    UPLOAD_DIR: str = str(pathlib.Path(__file__).parent.parent.parent.parent.parent / "data" / "uploads")
    UPLOAD_MAX_SIZE_MB: int = 50
    SAMPLE_LIBRARY_DIR: str | None = None

    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings.model_validate({})
