"""Kit builder tools for the sample assistant agent."""

import json
import logging

from pydantic_ai import RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.schemas.kit import KitResult
from samplespace.schemas.sample import SampleSchema
from samplespace.services import kit_builder as kit_builder_service

logger = logging.getLogger(__name__)


async def build_kit(
    ctx: RunContext[AgentDeps],
    vibe: str | None = None,
    genre: str | None = None,
    types: list[str] | None = None,
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


def _format_kit_result(kit: KitResult) -> str:
    """Format a kit result as a kit code fence for frontend rendering."""
    # Build JSON payload for the frontend component
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


def _sample_to_payload(sample: SampleSchema) -> dict[str, object]:
    """Convert a sample schema to a JSON-serializable payload for the frontend."""
    payload: dict[str, object] = {
        "id": sample.id,
        "filename": sample.filename,
        "audio_url": f"/api/samples/{sample.id}/audio",
    }
    if sample.sample_type:
        payload["type"] = sample.sample_type
    if sample.is_loop:
        if sample.key:
            payload["key"] = sample.key
        if sample.bpm and sample.bpm > 0:
            payload["bpm"] = sample.bpm
    return payload
