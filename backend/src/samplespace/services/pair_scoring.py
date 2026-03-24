"""Multi-dimensional pair compatibility scoring between two audio samples."""

import logging

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from samplespace.models.sample import Sample
from samplespace.schemas.pair import DimensionScore, PairScore
from samplespace.services import music_theory as music_theory_service

logger = logging.getLogger(__name__)

# Default weights for each scoring dimension (must sum to 1.0)
DEFAULT_WEIGHTS: dict[str, float] = {
    "key": 0.30,
    "bpm": 0.20,
    "type": 0.25,
    "spectral": 0.25,
}

# Type complementarity matrix — symmetric lookup via frozenset keys.
# Higher scores mean the types naturally complement each other.
TYPE_COMPLEMENTARITY: dict[frozenset[str], float] = {
    # High complementarity — different frequency ranges / roles
    frozenset({"kick", "hihat"}): 0.9,
    frozenset({"bass", "lead"}): 0.9,
    frozenset({"kick", "snare"}): 0.85,
    frozenset({"pad", "lead"}): 0.85,
    frozenset({"pad", "vocals"}): 0.85,
    # Medium complementarity
    frozenset({"bass", "pad"}): 0.8,
    frozenset({"snare", "hihat"}): 0.8,
    frozenset({"kick", "bass"}): 0.75,
    frozenset({"kick", "lead"}): 0.75,
    frozenset({"kick", "pad"}): 0.75,
    frozenset({"snare", "bass"}): 0.7,
    frozenset({"snare", "lead"}): 0.7,
    frozenset({"snare", "pad"}): 0.7,
    frozenset({"hihat", "bass"}): 0.7,
    frozenset({"hihat", "lead"}): 0.7,
    frozenset({"hihat", "pad"}): 0.7,
    # Low complementarity — same type
    frozenset({"kick"}): 0.2,
    frozenset({"snare"}): 0.2,
    frozenset({"hihat"}): 0.2,
    frozenset({"bass"}): 0.2,
    frozenset({"lead"}): 0.3,
    frozenset({"pad"}): 0.3,
    frozenset({"vocals"}): 0.3,
}
DEFAULT_TYPE_SCORE = 0.5

# Type pairs with complementarity >= this threshold are considered "complementary"
# for spectral score interpretation.
_COMPLEMENTARY_THRESHOLD = 0.6


async def score_pair(db: AsyncSession, sample_a_id: str, sample_b_id: str) -> PairScore:
    """Score compatibility between two samples across multiple dimensions.

    Loads both samples from the database, computes available scoring dimensions
    (key, BPM, type, spectral), rebalances weights for missing dimensions,
    and returns a composite score with per-dimension breakdowns.
    """
    sample_a = await Sample.get(db, sample_a_id)
    if sample_a is None:
        return _error_score(sample_a_id, sample_b_id, f"Sample {sample_a_id} not found")

    sample_b = await Sample.get(db, sample_b_id)
    if sample_b is None:
        return _error_score(sample_a_id, sample_b_id, f"Sample {sample_b_id} not found")

    # Collect available dimensions
    dimensions: dict[str, DimensionScore] = {}

    # Key score — only for loops with known keys
    if sample_a.is_loop and sample_b.is_loop and sample_a.key and sample_b.key:
        dimensions["key"] = _compute_key_score(sample_a.key, sample_b.key)

    # BPM score — only for loops with known BPMs
    if sample_a.is_loop and sample_b.is_loop and sample_a.bpm and sample_b.bpm:
        dimensions["bpm"] = _compute_bpm_score(sample_a.bpm, sample_b.bpm)

    # Type score
    if sample_a.sample_type and sample_b.sample_type:
        dimensions["type"] = _compute_type_score(sample_a.sample_type, sample_b.sample_type)

    # Spectral score (CNN embedding cosine distance)
    if sample_a.cnn_embedding and sample_b.cnn_embedding:
        types_are_complementary = _are_types_complementary(sample_a.sample_type, sample_b.sample_type)
        dimensions["spectral"] = _compute_spectral_score(
            sample_a.cnn_embedding, sample_b.cnn_embedding, types_are_complementary
        )

    # Rebalance weights and compute composite
    if not dimensions:
        return PairScore(
            sample_a_id=sample_a_id,
            sample_b_id=sample_b_id,
            overall=0.5,
            summary="Not enough metadata to score this pair — no key, BPM, type, or spectral data available.",
        )

    _rebalance_weights(dimensions)
    overall = sum(d.value * d.weight for d in dimensions.values())

    score = PairScore(
        sample_a_id=sample_a_id,
        sample_b_id=sample_b_id,
        overall=round(overall, 2),
        key_score=dimensions.get("key"),
        bpm_score=dimensions.get("bpm"),
        type_score=dimensions.get("type"),
        spectral_score=dimensions.get("spectral"),
        summary="",  # filled below
    )
    score.summary = _generate_summary(score, sample_a.filename, sample_b.filename)
    return score


def _compute_key_score(key_a: str, key_b: str) -> DimensionScore:
    """Compute key compatibility dimension."""
    value, explanation = music_theory_service.key_compatibility_score(key_a, key_b)
    return DimensionScore(value=value, weight=0.0, explanation=f"Key: {explanation}")


def _compute_bpm_score(bpm_a: int, bpm_b: int) -> DimensionScore:
    """Compute BPM compatibility with integer-multiple normalization."""
    norm_a = _normalize_bpm(bpm_a)
    norm_b = _normalize_bpm(bpm_b)

    if max(norm_a, norm_b) == 0:
        return DimensionScore(value=0.5, weight=0.0, explanation="BPM: could not compare")

    value = 1.0 - abs(norm_a - norm_b) / max(norm_a, norm_b)

    diff = abs(bpm_a - bpm_b)
    if diff == 0:
        explanation = f"identical BPM ({bpm_a})"
    elif bpm_a != norm_a or bpm_b != norm_b:
        explanation = f"{bpm_a} and {bpm_b} BPM (normalized to {norm_a}/{norm_b}, compatible as integer multiples)"
    elif diff <= 5:
        explanation = f"nearly identical BPM ({bpm_a} vs {bpm_b})"
    else:
        explanation = f"{bpm_a} vs {bpm_b} BPM (difference of {diff})"

    return DimensionScore(value=round(value, 2), weight=0.0, explanation=f"BPM: {explanation}")


def _compute_type_score(type_a: str, type_b: str) -> DimensionScore:
    """Compute type complementarity dimension."""
    pair_key = frozenset({type_a.lower(), type_b.lower()})
    value = TYPE_COMPLEMENTARITY.get(pair_key, DEFAULT_TYPE_SCORE)

    if value >= 0.8:
        label = "highly complementary"
    elif value >= 0.6:
        label = "complementary"
    elif value >= 0.4:
        label = "neutral"
    else:
        label = "same type — limited complementarity"

    explanation = f"{type_a} + {type_b}: {label}"
    return DimensionScore(value=value, weight=0.0, explanation=f"Type: {explanation}")


def _compute_spectral_score(
    emb_a: list[float],
    emb_b: list[float],
    types_are_complementary: bool,
) -> DimensionScore:
    """Compute CNN spectral distance dimension.

    For complementary types, spectral difference is desirable (score = distance).
    For same/similar types, spectral similarity is desirable (score = 1 - distance).
    """
    arr_a = np.array(emb_a, dtype=np.float32)
    arr_b = np.array(emb_b, dtype=np.float32)
    cosine_distance = float(1.0 - np.dot(arr_a, arr_b))
    cosine_distance = max(0.0, min(cosine_distance, 1.0))  # clamp

    if types_are_complementary:
        value = cosine_distance
        if value >= 0.5:
            label = "spectrally distinct — good for layering"
        else:
            label = "spectrally similar — may compete for frequency space"
    else:
        value = 1.0 - cosine_distance
        if value >= 0.5:
            label = "spectrally similar — cohesive sound"
        else:
            label = "spectrally different — less cohesive"

    return DimensionScore(value=round(value, 2), weight=0.0, explanation=f"Spectral: {label}")


def _normalize_bpm(bpm: int) -> int:
    """Normalize BPM to the 60-180 range by halving or doubling."""
    normalized = bpm
    while normalized > 180:
        normalized //= 2
    while normalized < 60:
        normalized *= 2
    return normalized


def _are_types_complementary(type_a: str | None, type_b: str | None) -> bool:
    """Check if two sample types are considered complementary."""
    if not type_a or not type_b:
        return True  # assume complementary when unknown
    pair_key = frozenset({type_a.lower(), type_b.lower()})
    score = TYPE_COMPLEMENTARITY.get(pair_key, DEFAULT_TYPE_SCORE)
    return score >= _COMPLEMENTARY_THRESHOLD


def _rebalance_weights(dimensions: dict[str, DimensionScore]) -> None:
    """Redistribute default weights proportionally across available dimensions.

    Mutates the DimensionScore objects in place to set their effective weights.
    """
    total_available_weight = sum(DEFAULT_WEIGHTS[k] for k in dimensions)
    for key, dim in dimensions.items():
        dim.weight = round(DEFAULT_WEIGHTS[key] / total_available_weight, 2)


def _generate_summary(score: PairScore, filename_a: str, filename_b: str) -> str:
    """Generate a human-readable summary of the pair score."""
    if score.overall >= 0.8:
        level = "Strong compatibility"
    elif score.overall >= 0.6:
        level = "Good compatibility"
    elif score.overall >= 0.4:
        level = "Moderate compatibility"
    else:
        level = "Low compatibility"

    parts = [f"{level} ({score.overall})"]

    dimension_explanations = []
    for dim in [score.key_score, score.bpm_score, score.type_score, score.spectral_score]:
        if dim is not None:
            dimension_explanations.append(dim.explanation.lower())

    if dimension_explanations:
        parts.append("; ".join(dimension_explanations))

    return f"{filename_a} × {filename_b}: {' — '.join(parts)}."


def _error_score(sample_a_id: str, sample_b_id: str, message: str) -> PairScore:
    """Return a PairScore indicating an error."""
    return PairScore(
        sample_a_id=sample_a_id,
        sample_b_id=sample_b_id,
        overall=0.0,
        summary=message,
    )
