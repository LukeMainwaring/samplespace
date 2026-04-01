from datetime import datetime

from pydantic import Field

from samplespace.schemas.base import BaseSchema


class PairFeatures(BaseSchema):
    spectral_overlap: float = Field(ge=0.0, le=1.0, description="Frequency spectrum IoU")
    onset_alignment: float = Field(ge=0.0, le=1.0, description="Onset cross-correlation")
    timbral_contrast: float = Field(ge=0.0, le=1.0, description="MFCC cosine distance")
    harmonic_consonance: float = Field(ge=0.0, le=1.0, description="Chroma correlation")
    spectral_centroid_gap: float = Field(ge=0.0, le=1.0, description="Normalized centroid difference")
    rms_energy_ratio: float = Field(ge=0.0, le=1.0, description="Normalized log energy ratio")


class PairVerdictSchema(BaseSchema):
    id: int
    thread_id: str
    sample_a_id: str
    sample_b_id: str
    verdict: bool
    pair_score: float
    pair_features: PairFeatures | None = None
    created_at: datetime
