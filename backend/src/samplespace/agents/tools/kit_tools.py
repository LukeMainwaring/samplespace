import asyncio
import logging
from pathlib import Path

from pydantic_ai import RunContext, ToolReturn
from pydantic_ai.ui.vercel_ai.response_types import DataChunk

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.formatting import sample_to_payload
from samplespace.agents.tools.transform_tools import transform_single_sample
from samplespace.schemas.kit import KitResult
from samplespace.services import audio_transform as audio_transform_service
from samplespace.services import kit_builder as kit_builder_service
from samplespace.services import kit_preview as kit_preview_service
from samplespace.services import sample as sample_service

logger = logging.getLogger(__name__)


async def build_kit(
    ctx: RunContext[AgentDeps],
    vibe: str | None = None,
    genre: str | None = None,
    types: list[str] | None = None,
    replacements: dict[str, str] | None = None,
) -> str | ToolReturn:
    """Assemble a multi-sample kit optimized for pairwise compatibility.

    Builds a complete kit (e.g., kick + snare + hihat + bass + pad) by
    searching for candidates per type, then greedily selecting samples
    that maximize inter-sample compatibility while maintaining diversity.

    Song context (key, BPM, vibe, genre) is automatically incorporated
    when available. Explicit vibe/genre args override song context values.

    Args:
        vibe: Sonic character description (e.g., "dark and gritty", "warm analog").
              Falls back to song context vibe if not provided.
        genre: Musical genre (e.g., "techno", "lo-fi hip hop").
               Falls back to song context genre if not provided.
        types: List of sample types to include (e.g., ["kick", "snare", "bass"]).
               Defaults to ["kick", "snare", "hihat", "bass", "pad"].
        replacements: Pin specific samples into slots by type, e.g.
                      {"snare": "abc-123", "bass": "def-456"}.
                      Pinned slots skip CLAP search and use the given sample directly.
                      Use this when the user wants to swap samples in an existing kit.
    """
    try:
        kit = await kit_builder_service.build_kit(
            ctx.deps.db,
            clap_model=ctx.deps.clap_model,
            clap_processor=ctx.deps.clap_processor,
            types=types,
            song_context=ctx.deps.song_context,
            vibe=vibe,
            genre=genre,
            replacements=replacements,
        )

        if not kit.slots:
            msg = "Could not assemble a kit — no suitable samples found."
            if kit.skipped_types:
                msg += f" No candidates for: {', '.join(kit.skipped_types)}."
            return msg

        return _format_kit_result(kit)

    except Exception:
        logger.exception("Error building kit")
        return "An error occurred while building the kit."


async def transform_kit(
    ctx: RunContext[AgentDeps],
    slots: list[dict[str, str]],
    target_key: str | None = None,
    target_bpm: int | None = None,
) -> str | ToolReturn:
    """Transform all samples in a kit to match a target key and BPM.

    Pitch-shifts and time-stretches each sample to align with the target.
    Falls back to song context key/BPM if not explicitly provided.
    Only transforms loops that have the required metadata.

    Args:
        slots: List of kit slots, each with "type" and "sample_id" keys.
               Pass the slots from the most recent build_kit result.
        target_key: Target key (e.g. "A minor"). Falls back to song context.
        target_bpm: Target BPM. Falls back to song context.
    """
    try:
        return await _transform_kit(ctx, slots, target_key, target_bpm)
    except Exception:
        logger.exception("Error transforming kit")
        return "An error occurred while transforming the kit."


async def _transform_kit(
    ctx: RunContext[AgentDeps],
    slots: list[dict[str, str]],
    target_key: str | None,
    target_bpm: int | None,
) -> str | ToolReturn:
    # Resolve targets from song context
    song_ctx = ctx.deps.song_context
    if target_key is None and song_ctx:
        target_key = song_ctx.key
    if target_bpm is None and song_ctx:
        target_bpm = song_ctx.bpm

    if target_key is None and target_bpm is None:
        return "No target key or BPM available. Set a song context first or provide explicit targets."

    transformed_slots: list[dict[str, object]] = []
    transform_notes: list[str] = []

    for i, slot in enumerate(slots):
        sample_id = slot.get("sample_id", "")
        slot_type = slot.get("type", "unknown")

        result = await transform_single_sample(ctx.deps.db, sample_id, target_key, target_bpm)

        if isinstance(result, str):
            transform_notes.append(f"{slot_type}: sample not found, skipped")
            continue

        transformed_slots.append(
            {
                "position": i,
                "requested_type": slot_type,
                "sample": sample_to_payload(result.sample, audio_url=result.audio_url),
            }
        )
        transform_notes.append(f"{slot_type}: {result.note}")

    if not transformed_slots:
        return "No samples could be transformed."

    # Build kit payload (omit scores — they'd be stale after transform)
    payload: dict[str, object] = {
        "slots": transformed_slots,
        "pairwise_scores": [],
    }

    # Build summary
    summary_lines = [f"- {note}" for note in transform_notes]
    summary = "\n".join(summary_lines)
    target_desc = " / ".join(p for p in [target_key, f"{target_bpm} BPM" if target_bpm else None] if p)
    intro = f"Transformed kit to match **{target_desc}**:\n\n{summary}"

    return ToolReturn(
        return_value=intro,
        metadata=DataChunk(type="data-kit", data=payload),
    )


async def preview_kit(
    ctx: RunContext[AgentDeps],
    slots: list[dict[str, str]],
) -> str | ToolReturn:
    """Mix all kit samples into a single layered audio preview.

    Layers all samples on top of each other (padded to the longest duration)
    so the user can hear the full kit playing together as one track.

    Args:
        slots: List of kit slots, each with "type" and "sample_id" keys.
               Pass the slots from the most recent build_kit or transform_kit result.
    """
    try:
        return await _preview_kit(ctx, slots)
    except Exception:
        logger.exception("Error generating kit preview")
        return "An error occurred while generating the kit preview."


async def _preview_kit(
    ctx: RunContext[AgentDeps],
    slots: list[dict[str, str]],
) -> str | ToolReturn:
    file_paths: list[Path] = []

    # Use song context to automatically pick up cached transforms —
    # the agent can't pass transformed audio URLs because ToolReturn
    # metadata goes to the frontend, not back to the LLM.
    song_ctx = ctx.deps.song_context
    target_key = song_ctx.key if song_ctx else None
    target_bpm = song_ctx.bpm if song_ctx else None

    for slot in slots:
        # The LLM passes {"type": ..., "sample_id": ...}, but the DataChunk
        # payload nests the ID inside {"sample": {"id": ...}}. Handle both.
        slot_sample = slot.get("sample")
        sample_data: dict[str, object] = slot_sample if isinstance(slot_sample, dict) else {}
        sample_id = str(sample_data.get("id", "") or slot.get("sample_id", ""))

        raw_sample = await sample_service.get_sample_by_id(ctx.deps.db, sample_id)
        if raw_sample is None:
            continue

        audio_path = sample_service.find_audio_file(raw_sample)
        if audio_path is None:
            continue

        # Use the transformed version if one exists for this song context
        if (target_key or target_bpm) and raw_sample.is_loop:
            resolved, _ = await asyncio.to_thread(
                audio_transform_service.resolve_transform,
                audio_path,
                raw_sample.id,
                sample_key=raw_sample.key,
                sample_bpm=raw_sample.bpm,
                target_key=target_key,
                target_bpm=target_bpm,
            )
            file_paths.append(resolved)
        else:
            file_paths.append(audio_path)

    if not file_paths:
        return "No audio files found for the kit samples."

    preview_id, _ = await asyncio.to_thread(kit_preview_service.mix_audio, file_paths)

    audio_url = f"/api/samples/kit-preview/{preview_id}"

    payload: dict[str, object] = {"audio_url": audio_url}
    if song_ctx:
        if song_ctx.key:
            payload["target_key"] = song_ctx.key
        if song_ctx.bpm:
            payload["target_bpm"] = song_ctx.bpm

    return ToolReturn(
        return_value="Here's the full kit layered together:",
        metadata=DataChunk(type="data-kit-preview", data=payload),
    )


def _format_kit_result(kit: KitResult) -> ToolReturn:
    payload: dict[str, object] = {
        "slots": [
            {
                "position": slot.position,
                "requested_type": slot.requested_type,
                "sample": sample_to_payload(slot.sample),
                "compatibility_score": slot.compatibility_score,
            }
            for slot in kit.slots
        ],
        "overall_score": kit.overall_score,
        "pairwise_scores": [
            {
                "slot_a": p.slot_a,
                "slot_b": p.slot_b,
                "score": p.score,
                "summary": p.summary,
            }
            for p in kit.pairwise_scores
        ],
    }
    if kit.vibe:
        payload["vibe"] = kit.vibe
    if kit.genre:
        payload["genre"] = kit.genre
    if kit.skipped_types:
        payload["skipped_types"] = kit.skipped_types

    # Build intro text with slot IDs for LLM follow-up calls
    type_list = ", ".join(slot.requested_type for slot in kit.slots)
    lines = [f"Here's a {len(kit.slots)}-sample kit ({type_list}) with {kit.overall_score:.2f} overall compatibility:"]

    if kit.skipped_types:
        lines.append(f"*Could not find samples for: {', '.join(kit.skipped_types)}.*")

    for slot in kit.slots:
        lines.append(f"- {slot.requested_type}: {slot.sample.id} ({slot.sample.filename})")

    return ToolReturn(
        return_value="\n".join(lines),
        metadata=DataChunk(type="data-kit", data=payload),
    )
