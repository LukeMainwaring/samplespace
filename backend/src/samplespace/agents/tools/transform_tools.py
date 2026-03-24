"""Audio transformation tools for the sample assistant agent."""

import asyncio
import logging
from urllib.parse import quote

from pydantic_ai import Agent, RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.services import audio_transform as audio_transform_service
from samplespace.services import music_theory as music_theory_service
from samplespace.services import sample as sample_service

logger = logging.getLogger(__name__)


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
    # Look up sample
    sample = await sample_service.get_sample_by_id(ctx.deps.db, sample_id)
    if sample is None:
        return f"Sample {sample_id} not found."

    # One-shots cannot be meaningfully transformed
    if not sample.is_loop:
        return (
            f"**{sample.filename}** is a one-shot — one-shots don't have a reference "
            "key or BPM, so automatic transformation doesn't apply."
        )

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

    # Check if sample has the needed metadata
    will_pitch_shift = target_key is not None and sample.key is not None
    will_time_stretch = target_bpm is not None and sample.bpm is not None
    skipped: list[str] = []

    if target_key and not sample.key:
        skipped.append("Key transformation skipped — sample has no detected key.")
    if target_bpm and not sample.bpm:
        skipped.append("BPM transformation skipped — sample has no detected BPM.")

    if not will_pitch_shift and not will_time_stretch:
        return f"**{sample.filename}** is missing the metadata needed for transformation. " + " ".join(skipped)

    # Compute actual pitch target (handles cross-mode via relative keys)
    actual_target_key = None
    n_steps = 0
    if will_pitch_shift:
        actual_target_key = music_theory_service.compute_target_key(sample.key, target_key)  # type: ignore[arg-type]
        if actual_target_key is None:
            skipped.append(f"Could not compute target key from {sample.key} → {target_key}.")
            will_pitch_shift = False
        elif actual_target_key == sample.key:
            will_pitch_shift = False
        else:
            n_steps = music_theory_service.semitone_delta(sample.key, actual_target_key) or 0  # type: ignore[arg-type]

    # Check if already matching
    bpm_matches = not will_time_stretch or (sample.bpm == target_bpm)
    key_matches = not will_pitch_shift
    if key_matches and bpm_matches and not skipped:
        return f"**{sample.filename}** already matches the target — no transformation needed."

    # Find audio file on disk
    audio_path = sample_service.find_audio_file(sample)
    if audio_path is None:
        return f"Audio file not found for **{sample.filename}**."

    # Run transformation (CPU-bound, offload to thread)
    cache_path = await asyncio.to_thread(
        audio_transform_service.transform_sample,
        audio_path,
        sample_id,
        source_key=sample.key if will_pitch_shift else None,
        target_key=actual_target_key if will_pitch_shift else None,
        source_bpm=sample.bpm if will_time_stretch else None,
        target_bpm=target_bpm if will_time_stretch else None,
    )

    # Build response
    return _format_result(
        sample.filename,
        sample.key,
        actual_target_key,
        n_steps,
        sample.bpm,
        target_bpm if will_time_stretch else None,
        sample_id,
        actual_target_key if will_pitch_shift else target_key,
        target_bpm,
        skipped,
        cache_path,
    )


def _format_result(
    filename: str,
    source_key: str | None,
    actual_target_key: str | None,
    n_steps: int,
    source_bpm: int | None,
    target_bpm: int | None,
    sample_id: str,
    url_key: str | None,
    url_bpm: int | None,
    skipped: list[str],
    cache_path: object,
) -> str:
    """Format the transformation result as markdown with an audio code fence."""
    parts: list[str] = []

    # Description of what was done
    transforms: list[str] = []
    if actual_target_key and n_steps != 0:
        transforms.append(f"from {source_key} to {actual_target_key} ({n_steps:+d} semitones)")
    if target_bpm and source_bpm and target_bpm != source_bpm:
        transforms.append(f"from {source_bpm} to {target_bpm} BPM")

    if transforms:
        parts.append(f"Transformed **{filename}** {', '.join(transforms)}.")
    else:
        parts.append(f"Processed **{filename}** (no audible change needed).")

    # Audio player code fence
    query_parts: list[str] = []
    if url_key:
        query_parts.append(f"key={quote(url_key, safe='')}")
    if url_bpm:
        query_parts.append(f"bpm={url_bpm}")
    query = "&".join(query_parts)
    audio_url = f"/api/samples/{sample_id}/audio/transformed?{query}"

    parts.append("")
    parts.append("```audio")
    parts.append(audio_url)
    parts.append("```")

    # Warnings and notes
    if abs(n_steps) > 5:
        parts.append("")
        parts.append(
            f"**Note:** This is a large pitch shift ({n_steps:+d} semitones) — listen carefully for artifacts."
        )

    if skipped:
        parts.append("")
        for note in skipped:
            parts.append(f"*{note}*")

    return "\n".join(parts)


def register_transform_tools(agent: Agent[AgentDeps, str]) -> None:
    """Register audio transformation tools with the agent."""
    agent.tool(match_to_context)
