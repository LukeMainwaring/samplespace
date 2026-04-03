from datetime import datetime

from pydantic import BaseModel, Field

from samplespace.schemas.base import BaseSchema
from samplespace.schemas.sample_type import SampleType


class PaginationParams(BaseSchema):
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class SampleFilterParams(BaseSchema):
    sample_type: SampleType | None = None
    is_loop: bool | None = None


class ListSamplesParams(SampleFilterParams, PaginationParams):
    pass


class SampleSchema(BaseSchema):
    id: str
    filename: str
    source: str = "library"
    pack_name: str | None = None
    key: str | None = None
    bpm: int | None = None
    duration: float | None = None
    sample_type: SampleType | None = Field(None, description="Category of the sample (e.g., kick, snare, pad, synth)")
    is_loop: bool = False
    created_at: datetime


class SimilarSampleSchema(BaseSchema):
    sample: SampleSchema
    distance: float


class SampleListResponse(BaseModel):
    samples: list[SampleSchema]
    total: int


class SampleSearchRequest(BaseModel):
    query: str | None = Field(None, description="Natural language search query for CLAP semantic search")
    key: str | None = Field(None, description="Filter by musical key (e.g., 'C major', 'A minor')")
    bpm_min: int | None = Field(None, description="Minimum BPM filter")
    bpm_max: int | None = Field(None, description="Maximum BPM filter")
    sample_type: SampleType | None = Field(None, description="Filter by sample type")
    is_loop: bool | None = Field(None, description="Filter by loop (True) or one-shot (False)")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")
