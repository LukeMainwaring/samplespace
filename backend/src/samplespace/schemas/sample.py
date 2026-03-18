"""Pydantic schemas for sample API contracts."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SampleSchema(BaseModel):
    """Sample response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    key: str | None = None
    bpm: float | None = None
    duration: float | None = None
    sample_type: str | None = Field(None, description="Category of the sample (e.g., kick, snare, pad, lead)")
    created_at: datetime


class SampleListResponse(BaseModel):
    """Paginated list of samples."""

    samples: list[SampleSchema]
    total: int


class SampleSearchRequest(BaseModel):
    """Search request with optional filters."""

    query: str | None = Field(None, description="Natural language search query for CLAP semantic search")
    key: str | None = Field(None, description="Filter by musical key (e.g., 'C major', 'A minor')")
    bpm_min: float | None = Field(None, description="Minimum BPM filter")
    bpm_max: float | None = Field(None, description="Maximum BPM filter")
    sample_type: str | None = Field(None, description="Filter by sample type")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")
