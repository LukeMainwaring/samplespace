from datetime import datetime

from pydantic import Field

from samplespace.schemas.base import BaseSchema


class PreferenceMeta(BaseSchema):
    version: int
    accuracy: float = Field(ge=0.0, le=1.0)
    verdict_count: int
    feature_importances: dict[str, float]
    trained_at: datetime


class PreferenceExplanation(BaseSchema):
    meta: PreferenceMeta
    summary: str
    top_features: list[tuple[str, float, str]]
