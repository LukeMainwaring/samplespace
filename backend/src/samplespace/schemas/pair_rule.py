"""Schemas for learned pair rules."""

from datetime import datetime

from pydantic import Field

from samplespace.schemas.base import BaseSchema


class PairRuleSchema(BaseSchema):
    """Response schema for a pair rule."""

    id: int
    version: int
    type_pair: str
    feature_name: str
    threshold: float
    direction: str
    confidence: float = Field(ge=0.0, le=1.0)
    sample_count: int
    is_active: bool
    created_at: datetime
