import asyncio
import json
import logging
from pathlib import Path
from urllib.parse import quote

from pydantic_ai import RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.schemas.kit import KitResult
from samplespace.schemas.sample import SampleSchema
from samplespace.services import audio_transform as audio_transform_service
from samplespace.services import kit_builder as kit_builder_service
from samplespace.services import kit_preview as kit_preview_service
from samplespace.services import music_theory as music_theory_service
from samplespace.services import sample as sample_service

logger = logging.getLogger(__name__)


async def build_kit(
    ctx: RunContext[AgentDeps],
    vibe: str | None = None,
    genre: str | None = None,
    types: list[str] | None = None,
    replacements: dict[str, str] | None = None,
) -> str:
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
) -> str:
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
) -> str:
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

        raw_sample = await sample_service.get_sample_by_id(ctx.deps.db, sample_id)
        if raw_sample is None:
            transform_notes.append(f"{slot_type}: sample not found, skipped")
            continue
        sample = SampleSchema.model_validate(raw_sample)

        if not sample.is_loop:
            # Include as-is with original audio
            transformed_slots.append(
                {
                    "position": i,
                    "requested_type": slot_type,
                    "sample": _sample_to_payload(sample),
                }
            )
            transform_notes.append(f"{slot_type}: one-shot, no transform needed")
            continue

        # Determine what transforms apply
        will_pitch = target_key is not None and sample.key is not None
        will_stretch = target_bpm is not None and sample.bpm is not None

        actual_target_key: str | None = None
        n_steps = 0
        if will_pitch and sample.key and target_key:
            actual_target_key = music_theory_service.compute_target_key(
                sample.key,
                target_key,
            )
            if actual_target_key is None or actual_target_key == sample.key:
                will_pitch = False
            elif sample.key:
                n_steps = (
                    music_theory_service.semitone_delta(
                        sample.key,
                        actual_target_key,
                    )
                    or 0
                )

        bpm_matches = not will_stretch or sample.bpm == target_bpm

        if not will_pitch and bpm_matches:
            # Already matches — include with original audio
            transformed_slots.append(
                {
                    "position": i,
                    "requested_type": slot_type,
                    "sample": _sample_to_payload(sample),
                }
            )
            transform_notes.append(f"{slot_type}: already matches")
            continue

        # Find audio file and transform
        audio_path = sample_service.find_audio_file(raw_sample)
        if audio_path is None:
            transformed_slots.append(
                {
                    "position": i,
                    "requested_type": slot_type,
                    "sample": _sample_to_payload(sample),
                }
            )
            transform_notes.append(f"{slot_type}: audio file not found, skipped")
            continue

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

        transformed_slots.append(
            {
                "position": i,
                "requested_type": slot_type,
                "sample": _sample_to_payload(sample, audio_url=transformed_url),
            }
        )

        # Build note
        note_parts: list[str] = []
        if will_pitch and actual_target_key:
            note_parts.append(f"{sample.key} → {actual_target_key} ({n_steps:+d} semitones)")
        if not bpm_matches:
            note_parts.append(f"{sample.bpm} → {target_bpm} BPM")
        transform_notes.append(f"{slot_type}: {', '.join(note_parts)}")

    if not transformed_slots:
        return "No samples could be transformed."

    # Build kit code fence (omit scores — they'd be stale after transform)
    payload: dict[str, object] = {
        "slots": transformed_slots,
        "pairwise_scores": [],
    }
    json_str = json.dumps(payload, indent=2)

    # Build summary
    summary_lines = [f"- {note}" for note in transform_notes]
    summary = "\n".join(summary_lines)
    target_desc = " / ".join(p for p in [target_key, f"{target_bpm} BPM" if target_bpm else None] if p)
    intro = f"Transformed kit to match **{target_desc}**:\n\n{summary}"

    return f"{intro}\n\n```kit\n{json_str}\n```"


async def preview_kit(
    ctx: RunContext[AgentDeps],
    slots: list[dict[str, str]],
) -> str:
    """Mix all kit samples into a single layered audio preview.

    Layers all samples on top of each other (padded to the longest duration)
    so the user can hear the full kit playing together as one track.

    Args:
        slots: List of kit slots, each with "sample_id" and optionally
               "audio_url" (for transformed samples). Pass the slots from
               the most recent build_kit or transform_kit result.
    """
    try:
        return await _preview_kit(ctx, slots)
    except Exception:
        logger.exception("Error generating kit preview")
        return "An error occurred while generating the kit preview."


async def _preview_kit(
    ctx: RunContext[AgentDeps],
    slots: list[dict[str, str]],
) -> str:
    file_paths: list[Path] = []

    for slot in slots:
        sample_id = slot.get("sample_id", "")
        audio_url = slot.get("audio_url", "")

        # Check if this is a transformed sample URL
        if "/audio/transformed" in audio_url:
            # Parse key/bpm from the URL query params
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(audio_url)
            params = parse_qs(parsed.query)
            t_key = params.get("key", [None])[0]
            t_bpm_str = params.get("bpm", [None])[0]
            t_bpm = int(t_bpm_str) if t_bpm_str else None

            cached = audio_transform_service.get_cached_transform(sample_id, t_key, t_bpm)
            if cached:
                file_paths.append(cached)
                continue

        # Fall back to original audio file
        raw_sample = await sample_service.get_sample_by_id(ctx.deps.db, sample_id)
        if raw_sample is None:
            continue
        audio_path = sample_service.find_audio_file(raw_sample)
        if audio_path:
            file_paths.append(audio_path)

    if not file_paths:
        return "No audio files found for the kit samples."

    preview_id, _ = await asyncio.to_thread(kit_preview_service.mix_audio, file_paths)

    audio_url = f"/api/samples/kit-preview/{preview_id}"
    return f"Here's the full kit layered together:\n\n```audio\n{audio_url}\n```"


def _format_kit_result(kit: KitResult) -> str:
    payload: dict[str, object] = {
        "slots": [
            {
                "position": slot.position,
                "requested_type": slot.requested_type,
                "sample": _sample_to_payload(slot.sample),
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

    json_str = json.dumps(payload, indent=2)

    # Build intro text
    type_list = ", ".join(slot.requested_type for slot in kit.slots)
    intro = f"Here's a {len(kit.slots)}-sample kit ({type_list}) with {kit.overall_score:.2f} overall compatibility:"

    if kit.skipped_types:
        intro += f"\n\n*Could not find samples for: {', '.join(kit.skipped_types)}.*"

    return f"{intro}\n\n```kit\n{json_str}\n```"


def _sample_to_payload(sample: SampleSchema, audio_url: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": sample.id,
        "filename": sample.filename,
        "audio_url": audio_url or f"/api/samples/{sample.id}/audio",
    }
    if sample.sample_type:
        payload["type"] = sample.sample_type
    if sample.is_loop:
        if sample.key:
            payload["key"] = sample.key
        if sample.bpm and sample.bpm > 0:
            payload["bpm"] = sample.bpm
    return payload
