import asyncio
import logging
from dataclasses import dataclass, field
from urllib.parse import quote

from pydantic_ai import RunContext
from sqlalchemy.ext.asyncio import AsyncSession

from samplespace.agents.deps import AgentDeps
from samplespace.schemas.sample import SampleSchema
from samplespace.services import audio_transform as audio_transform_service
from samplespace.services import music_theory as music_theory_service
from samplespace.services import sample as sample_service

logger = logging.getLogger(__name__)


@dataclass
class TransformResult:
    """Result of transforming a single sample."""

    sample: SampleSchema
    transformed: bool
    audio_url: str
    note: str
    n_steps: int = 0
    actual_target_key: str | None = None
    skipped_reasons: list[str] = field(default_factory=list)


async def transform_single_sample(
    db: AsyncSession,
    sample_id: str,
    target_key: str | None,
    target_bpm: int | None,
) -> TransformResult | str:
    """Core transform logic shared by match_to_context and transform_kit.

    Returns a TransformResult on success, or an error string if the sample
    can't be found.
    """
    raw_sample = await sample_service.get_sample_by_id(db, sample_id)
    if raw_sample is None:
        return f"Sample {sample_id} not found."

    sample = SampleSchema.model_validate(raw_sample)
    base_url = f"/api/samples/{sample_id}/audio"

    # One-shots cannot be meaningfully transformed
    if not sample.is_loop:
        return TransformResult(
            sample=sample,
            transformed=False,
            audio_url=base_url,
            note="one-shot, no transform needed",
        )

    # Determine what transforms apply
    will_pitch = target_key is not None and sample.key is not None
    will_stretch = target_bpm is not None and sample.bpm is not None
    skipped: list[str] = []

    if target_key and not sample.key:
        skipped.append("Key transformation skipped — sample has no detected key.")
    if target_bpm and not sample.bpm:
        skipped.append("BPM transformation skipped — sample has no detected BPM.")

    if not will_pitch and not will_stretch:
        return TransformResult(
            sample=sample,
            transformed=False,
            audio_url=base_url,
            note="missing metadata for transformation",
            skipped_reasons=skipped,
        )

    # Compute actual pitch target (handles cross-mode via relative keys)
    actual_target_key: str | None = None
    n_steps = 0
    if will_pitch and sample.key and target_key:
        actual_target_key = music_theory_service.compute_target_key(sample.key, target_key)
        if actual_target_key is None:
            skipped.append(f"Could not compute target key from {sample.key} → {target_key}.")
            will_pitch = False
        elif actual_target_key == sample.key:
            will_pitch = False
        else:
            n_steps = music_theory_service.semitone_delta(sample.key, actual_target_key) or 0

    # Check if already matching
    bpm_matches = not will_stretch or sample.bpm == target_bpm
    if not will_pitch and bpm_matches:
        return TransformResult(
            sample=sample,
            transformed=False,
            audio_url=base_url,
            note="already matches target",
            skipped_reasons=skipped,
        )

    # Find audio file on disk
    audio_path = sample_service.find_audio_file(raw_sample)
    if audio_path is None:
        return TransformResult(
            sample=sample,
            transformed=False,
            audio_url=base_url,
            note="audio file not found, skipped",
        )

    # Run transformation (CPU-bound, offload to thread)
    await asyncio.to_thread(
        audio_transform_service.transform_sample,
        audio_path,
        sample_id,
        source_key=sample.key if will_pitch else None,
        target_key=actual_target_key if will_pitch else None,
        source_bpm=sample.bpm if not bpm_matches else None,
        target_bpm=target_bpm if not bpm_matches else None,
    )

    # Build transformed audio URL
    query_parts: list[str] = []
    if will_pitch and actual_target_key:
        query_parts.append(f"key={quote(actual_target_key, safe='')}")
    if not bpm_matches and target_bpm:
        query_parts.append(f"bpm={target_bpm}")
    query = "&".join(query_parts)
    transformed_url = f"/api/samples/{sample_id}/audio/transformed?{query}"

    # Build note
    note_parts: list[str] = []
    if will_pitch and actual_target_key:
        note_parts.append(f"{sample.key} → {actual_target_key} ({n_steps:+d} semitones)")
    if not bpm_matches:
        note_parts.append(f"{sample.bpm} → {target_bpm} BPM")

    return TransformResult(
        sample=sample,
        transformed=True,
        audio_url=transformed_url,
        note=", ".join(note_parts),
        n_steps=n_steps,
        actual_target_key=actual_target_key if will_pitch else None,
        skipped_reasons=skipped,
    )


async def match_to_context(
    ctx: RunContext[AgentDeps],
    sample_id: str,
    target_key: str | None = None,
    target_bpm: int | None = None,
) -> str:
    """Pitch-shift and/or time-stretch a sample to match a target key and BPM.

    Use this when a sample is a good sonic fit but needs to be transposed or
    tempo-adjusted to match the song context. If target_key and target_bpm are
    not provided, falls back to the active song context values.

    Only works for loops — one-shots have no reference key/BPM.

    Args:
        sample_id: The ID of the sample to transform.
        target_key: Target key (e.g. "G minor"). Falls back to song context.
        target_bpm: Target BPM. Falls back to song context.
    """
    try:
        return await _match_to_context(ctx, sample_id, target_key, target_bpm)
    except Exception:
        logger.exception("Error transforming sample")
        return "An error occurred while transforming the sample."


async def _match_to_context(
    ctx: RunContext[AgentDeps],
    sample_id: str,
    target_key: str | None,
    target_bpm: int | None,
) -> str:
    # Resolve targets from song context if not explicitly provided
    song_ctx = ctx.deps.song_context
    if target_key is None and song_ctx:
        target_key = song_ctx.key
    if target_bpm is None and song_ctx:
        target_bpm = song_ctx.bpm

    if target_key is None and target_bpm is None:
        return (
            "No target key or BPM available. Set a song context first "
            '(e.g. "I\'m working in G minor at 120 BPM") or provide explicit targets.'
        )

    result = await transform_single_sample(ctx.deps.db, sample_id, target_key, target_bpm)

    if isinstance(result, str):
        return result

    if not result.sample.is_loop:
        return (
            f"**{result.sample.filename}** is a one-shot — one-shots don't have a reference "
            "key or BPM, so automatic transformation doesn't apply."
        )

    if not result.transformed and not result.skipped_reasons:
        return f"**{result.sample.filename}** already matches the target — no transformation needed."

    if not result.transformed:
        return f"**{result.sample.filename}** is missing the metadata needed for transformation. " + " ".join(
            result.skipped_reasons
        )

    return _format_match_result(result, target_key, target_bpm)


def _format_match_result(
    result: TransformResult,
    target_key: str | None,
    target_bpm: int | None,
) -> str:
    parts: list[str] = []

    # Description of what was done
    if result.note:
        parts.append(f"Transformed **{result.sample.filename}** {result.note}.")
    else:
        parts.append(f"Processed **{result.sample.filename}** (no audible change needed).")

    # Audio player code fence
    parts.append("")
    parts.append("```audio")
    parts.append(result.audio_url)
    parts.append("```")

    # Warnings and notes
    if abs(result.n_steps) > 5:
        parts.append("")
        parts.append(
            f"**Note:** This is a large pitch shift ({result.n_steps:+d} semitones) — listen carefully for artifacts."
        )

    if result.skipped_reasons:
        parts.append("")
        for note in result.skipped_reasons:
            parts.append(f"*{note}*")

    return "\n".join(parts)
