"""Shared candidate retrieval and reranking utilities.

Used by both the kit builder (multi-sample assembly) and pair evaluation
(single-pair candidate finding). Provides CLAP query building, context-aware
reranking, and BPM compatibility scoring.
"""

from samplespace.schemas.sample import SampleSchema
from samplespace.schemas.sample_type import UNPITCHED_TYPES
from samplespace.schemas.thread import SongContext
from samplespace.services.music_theory import normalize_bpm, semitone_key_score

# Re-ranking weight profiles: (clap, bpm, key)
_TONAL_WEIGHTS = (0.4, 0.25, 0.35)
_PERCUSSIVE_WEIGHTS = (0.5, 0.5, 0.0)

# Default number of candidates to keep after reranking
DEFAULT_RERANK_LIMIT = 10


def build_clap_query(
    sample_type: str,
    vibe: str | None = None,
    genre: str | None = None,
    song_context: SongContext | None = None,
) -> str:
    """Build a descriptive CLAP query incorporating type, vibe, genre, and song context."""
    parts: list[str] = []

    if genre:
        parts.append(genre)

    parts.append(f"{sample_type} sample")

    if song_context:
        if sample_type.lower() not in UNPITCHED_TYPES and song_context.key:
            parts.append(song_context.key)
        if song_context.bpm:
            parts.append(f"{song_context.bpm} BPM")

    if vibe:
        parts.append(vibe)

    return ", ".join(parts)


def rerank_candidates(
    candidates: list[SampleSchema],
    sample_type: str,
    song_context: SongContext | None,
    *,
    limit: int = DEFAULT_RERANK_LIMIT,
) -> list[SampleSchema]:
    """Re-rank CLAP results using song context BPM/key compatibility."""
    if not song_context or len(candidates) <= limit:
        return candidates[:limit]

    has_bpm = song_context.bpm is not None
    has_key = song_context.key is not None
    is_tonal = sample_type.lower() not in UNPITCHED_TYPES

    w_clap, w_bpm, w_key = _TONAL_WEIGHTS if is_tonal else _PERCUSSIVE_WEIGHTS

    # Redistribute unavailable dimension weights to CLAP
    if not has_bpm:
        w_clap += w_bpm
        w_bpm = 0.0
    if not has_key or not is_tonal:
        w_clap += w_key
        w_key = 0.0

    scored: list[tuple[float, SampleSchema]] = []
    n = len(candidates)

    for i, sample in enumerate(candidates):
        clap_score = 1.0 - (i / n)

        bpm_score = 0.0
        if has_bpm and sample.bpm:
            bpm_score = bpm_compatibility(song_context.bpm, sample.bpm)  # type: ignore[arg-type]

        key_score = 0.0
        if has_key and sample.key and is_tonal:
            key_score = semitone_key_score(song_context.key, sample.key)  # type: ignore[arg-type]

        composite = w_clap * clap_score + w_bpm * bpm_score + w_key * key_score
        scored.append((composite, sample))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:limit]]


def bpm_compatibility(bpm_a: int, bpm_b: int) -> float:
    """Compute BPM compatibility score (0-1) with octave normalization."""
    norm_a = normalize_bpm(bpm_a)
    norm_b = normalize_bpm(bpm_b)
    if max(norm_a, norm_b) == 0:
        return 0.5
    return 1.0 - abs(norm_a - norm_b) / max(norm_a, norm_b)
