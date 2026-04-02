import logging

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from samplespace.models.sample import Sample
from samplespace.schemas.pair import DimensionScore, PairScore
from samplespace.schemas.sample_type import SampleType
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
    frozenset({SampleType.KICK, SampleType.HIHAT}): 0.9,
    frozenset({SampleType.BASS, SampleType.SYNTH}): 0.9,
    frozenset({SampleType.KICK, SampleType.SNARE}): 0.85,
    frozenset({SampleType.PAD, SampleType.SYNTH}): 0.85,
    frozenset({SampleType.PAD, SampleType.VOCAL}): 0.85,
    # Medium complementarity
    frozenset({SampleType.BASS, SampleType.PAD}): 0.8,
    frozenset({SampleType.SNARE, SampleType.HIHAT}): 0.8,
    frozenset({SampleType.KICK, SampleType.BASS}): 0.75,
    frozenset({SampleType.KICK, SampleType.SYNTH}): 0.75,
    frozenset({SampleType.KICK, SampleType.PAD}): 0.75,
    frozenset({SampleType.SNARE, SampleType.BASS}): 0.7,
    frozenset({SampleType.SNARE, SampleType.SYNTH}): 0.7,
    frozenset({SampleType.SNARE, SampleType.PAD}): 0.7,
    frozenset({SampleType.HIHAT, SampleType.BASS}): 0.7,
    frozenset({SampleType.HIHAT, SampleType.SYNTH}): 0.7,
    frozenset({SampleType.HIHAT, SampleType.PAD}): 0.7,
    # Low complementarity — same type
    frozenset({SampleType.KICK}): 0.2,
    frozenset({SampleType.SNARE}): 0.2,
    frozenset({SampleType.HIHAT}): 0.2,
    frozenset({SampleType.BASS}): 0.2,
    frozenset({SampleType.SYNTH}): 0.3,
    frozenset({SampleType.PAD}): 0.3,
    frozenset({SampleType.VOCAL}): 0.3,
}
DEFAULT_TYPE_SCORE = 0.5

# Type pairs with complementarity >= this threshold are considered "complementary"
# for spectral score interpretation.
_COMPLEMENTARY_THRESHOLD = 0.6


async def score_pair(db: AsyncSession, sample_a_id: str, sample_b_id: str) -> PairScore:
    sample_a = await Sample.get(db, sample_a_id)
    if sample_a is None:
        return _error_score(sample_a_id, sample_b_id, f"Sample {sample_a_id} not found")

    sample_b = await Sample.get(db, sample_b_id)
    if sample_b is None:
        return _error_score(sample_a_id, sample_b_id, f"Sample {sample_b_id} not found")

    dimensions: dict[str, DimensionScore] = {}

    if sample_a.is_loop and sample_b.is_loop and sample_a.key and sample_b.key:
        dimensions["key"] = _compute_key_score(sample_a.key, sample_b.key)

    if sample_a.is_loop and sample_b.is_loop and sample_a.bpm and sample_b.bpm:
        dimensions["bpm"] = _compute_bpm_score(sample_a.bpm, sample_b.bpm)

    if sample_a.sample_type and sample_b.sample_type:
        dimensions["type"] = _compute_type_score(sample_a.sample_type, sample_b.sample_type)

    if sample_a.cnn_embedding is not None and sample_b.cnn_embedding is not None:
        types_are_complementary = _are_types_complementary(sample_a.sample_type, sample_b.sample_type)
        dimensions["spectral"] = _compute_spectral_score(
            sample_a.cnn_embedding, sample_b.cnn_embedding, types_are_complementary
        )

    if not dimensions:
        return PairScore(
            sample_a_id=sample_a_id,
            sample_b_id=sample_b_id,
            overall=0.5,
            summary="Not enough metadata to score this pair — no key, BPM, type, or spectral data available.",
        )

    _rebalance_weights(dimensions)
    overall = round(sum(d.value * d.weight for d in dimensions.values()), 2)

    summary = _generate_summary(overall, dimensions, sample_a.filename, sample_b.filename)

    return PairScore(
        sample_a_id=sample_a_id,
        sample_b_id=sample_b_id,
        overall=overall,
        key_score=dimensions.get("key"),
        bpm_score=dimensions.get("bpm"),
        type_score=dimensions.get("type"),
        spectral_score=dimensions.get("spectral"),
        summary=summary,
    )


def _compute_key_score(key_a: str, key_b: str) -> DimensionScore:
    value, explanation = music_theory_service.key_compatibility_score(key_a, key_b)
    return DimensionScore(value=value, weight=0.0, explanation=f"Key: {explanation}")


def _compute_bpm_score(bpm_a: int, bpm_b: int) -> DimensionScore:
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
    norm_a = float(np.linalg.norm(arr_a))
    norm_b = float(np.linalg.norm(arr_b))
    if norm_a == 0.0 or norm_b == 0.0:
        cosine_distance = 1.0
    else:
        cosine_distance = float(1.0 - np.dot(arr_a, arr_b) / (norm_a * norm_b))
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
    if bpm <= 0:
        return 0
    normalized = bpm
    while normalized > 180:
        normalized //= 2
    while normalized < 60:
        normalized *= 2
    return normalized


def _are_types_complementary(type_a: str | None, type_b: str | None) -> bool:
    if not type_a or not type_b:
        return True  # assume complementary when unknown
    pair_key = frozenset({type_a.lower(), type_b.lower()})
    score = TYPE_COMPLEMENTARITY.get(pair_key, DEFAULT_TYPE_SCORE)
    return score >= _COMPLEMENTARY_THRESHOLD


def _rebalance_weights(dimensions: dict[str, DimensionScore]) -> None:
    total_available_weight = sum(DEFAULT_WEIGHTS[k] for k in dimensions)
    for key, dim in dimensions.items():
        dim.weight = DEFAULT_WEIGHTS[key] / total_available_weight


def _generate_summary(
    overall: float,
    dimensions: dict[str, DimensionScore],
    filename_a: str,
    filename_b: str,
) -> str:
    if overall >= 0.8:
        level = "Strong compatibility"
    elif overall >= 0.6:
        level = "Good compatibility"
    elif overall >= 0.4:
        level = "Moderate compatibility"
    else:
        level = "Low compatibility"

    parts = [f"{level} ({overall})"]

    dimension_explanations = [dim.explanation.lower() for dim in dimensions.values()]
    if dimension_explanations:
        parts.append("; ".join(dimension_explanations))

    return f"{filename_a} × {filename_b}: {' — '.join(parts)}."


def _error_score(sample_a_id: str, sample_b_id: str, message: str) -> PairScore:
    return PairScore(
        sample_a_id=sample_a_id,
        sample_b_id=sample_b_id,
        overall=0.0,
        summary=message,
    )
