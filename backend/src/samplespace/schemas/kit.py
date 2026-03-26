"""Pydantic schemas for kit builder results."""

from pydantic import Field

from samplespace.schemas.base import BaseSchema
from samplespace.schemas.sample import SampleSchema


class KitSlot(BaseSchema):
    """A single slot in an assembled kit."""

    position: int = Field(ge=0, description="0-indexed slot position")
    requested_type: str = Field(description="Sample type requested for this slot (e.g., kick, snare)")
    sample: SampleSchema
    compatibility_score: float = Field(ge=0.0, le=1.0, description="Average pairwise score with other kit samples")


class PairwiseEntry(BaseSchema):
    """One pairwise compatibility score within a kit."""

    slot_a: int
    slot_b: int
    score: float = Field(ge=0.0, le=1.0)
    summary: str


class KitResult(BaseSchema):
    """Complete assembled kit with scoring details."""

    slots: list[KitSlot]
    overall_score: float = Field(ge=0.0, le=1.0, description="Mean of all pairwise scores")
    pairwise_scores: list[PairwiseEntry]
    vibe: str | None = None
    genre: str | None = None
    skipped_types: list[str] = Field(default_factory=list, description="Types with no candidates found")
