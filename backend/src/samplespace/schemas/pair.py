from pydantic import Field

from samplespace.schemas.base import BaseSchema


class DimensionScore(BaseSchema):
    value: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0, description="Effective weight after rebalancing")
    explanation: str


class PairScore(BaseSchema):
    sample_a_id: str
    sample_b_id: str
    overall: float = Field(ge=0.0, le=1.0, description="Weighted composite score")
    key_score: DimensionScore | None = None
    bpm_score: DimensionScore | None = None
    type_score: DimensionScore | None = None
    spectral_score: DimensionScore | None = None
    summary: str = Field(description="Human-readable explanation")
