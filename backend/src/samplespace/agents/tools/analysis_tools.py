import asyncio
import logging

from pydantic_ai import RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.formatting import format_sample_results
from samplespace.services import embedding as embedding_service
from samplespace.services import music_theory as music_theory_service
from samplespace.services import sample as sample_service

logger = logging.getLogger(__name__)


async def analyze_sample(ctx: RunContext[AgentDeps], sample_id: str) -> str:
    """Get full metadata for a specific sample.

    Use this tool when the user wants to know details about a sample —
    its key, BPM, duration, and type.

    Args:
        sample_id: The ID of the sample to analyze.
    """
    try:
        sample = await sample_service.get_sample_by_id(ctx.deps.db, sample_id)
        if sample is None:
            return f"Sample {sample_id} not found."

        lines = [
            f"**{sample.filename}**",
            f"- Type: {sample.sample_type or 'unknown'}",
            f"- Category: {'loop' if sample.is_loop else 'one-shot'}",
        ]
        if sample.is_loop:
            lines.append(f"- Key: {sample.key or 'unknown'}")
            lines.append(f"- BPM: {sample.bpm or 'unknown'}")
        lines.append(f"- Duration: {sample.duration:.1f}s" if sample.duration is not None else "- Duration: unknown")
        return "\n".join(lines)
    except Exception:
        logger.exception("Error analyzing sample")
        return "An error occurred while analyzing the sample."


async def check_key_compatibility(ctx: RunContext[AgentDeps], key1: str, key2: str) -> str:
    """Check if two musical keys are compatible for mixing or layering.

    Use this tool when the user asks whether two samples will sound good together
    based on their keys. Returns compatibility info based on circle of fifths
    and relative major/minor relationships.

    Args:
        key1: First key, e.g., "C major" or "A minor".
        key2: Second key, e.g., "G major" or "E minor".
    """
    if key1 == key2:
        return f"**{key1}** and **{key2}** are the same key — perfectly compatible!"

    if music_theory_service.are_relative_pairs(key1, key2):
        return (
            f"**{key1}** and **{key2}** are relative major/minor pairs — highly compatible! They share the same notes."
        )

    distance = music_theory_service.key_distance(key1, key2)
    if distance is None:
        return f"Could not determine compatibility between **{key1}** and **{key2}**."

    if distance <= 1:
        return (
            f"**{key1}** and **{key2}** are adjacent on the circle of fifths "
            f"(distance: {distance}) — very compatible for mixing!"
        )
    elif distance <= 2:
        return (
            f"**{key1}** and **{key2}** are close on the circle of fifths "
            f"(distance: {distance}) — generally compatible."
        )
    else:
        return (
            f"**{key1}** and **{key2}** are distant on the circle of fifths "
            f"(distance: {distance}) — may clash. Consider pitch-shifting one."
        )


async def suggest_complement(
    ctx: RunContext[AgentDeps],
    sample_id: str,
    desired_type: str | None = None,
) -> str:
    """Suggest samples that complement a given sample based on key compatibility and type.

    Use this tool when the user wants to build a kit or find samples that work
    well together, e.g., "find a bass that goes with this pad" or "suggest
    complementary samples for this kick".

    Args:
        sample_id: The ID of the sample to find complements for.
        desired_type: Optional type filter (e.g., "bass", "pad", "synth").
    """
    try:
        source = await sample_service.get_sample_by_id(ctx.deps.db, sample_id)
        if source is None:
            return f"Sample {sample_id} not found."

        query = f"complement for {source.sample_type or 'sample'}"
        if desired_type:
            query = f"{desired_type} that complements {source.sample_type or 'sample'}"

        if ctx.deps.song_context and ctx.deps.song_context.vibe:
            query = f"{query}, {ctx.deps.song_context.vibe}"

        query_embedding = await asyncio.to_thread(
            embedding_service.embed_text, query, ctx.deps.clap_model, ctx.deps.clap_processor
        )

        results = await sample_service.search_by_text(
            ctx.deps.db,
            query_embedding=query_embedding,
            sample_type=desired_type,
            limit=10,
        )

        filtered = [r for r in results if r.id != sample_id]

        if not filtered:
            return "No complementary samples found."

        reference_key = source.key
        if not reference_key and ctx.deps.song_context:
            reference_key = ctx.deps.song_context.key

        if source.is_loop:
            header = f"Samples that complement **{source.filename}** (key: {source.key or 'unknown'}):"
        elif reference_key and reference_key != source.key:
            header = (
                f"Samples that complement **{source.filename}** (one-shot, using song context key: {reference_key}):"
            )
        else:
            header = f"Samples that complement **{source.filename}**:"

        # Build key compatibility annotations
        annotations: dict[str, str] = {}
        if reference_key:
            for s in filtered[:8]:
                if s.key:
                    if reference_key == s.key:
                        annotations[s.id] = " ✓ same key"
                    elif music_theory_service.are_relative_pairs(reference_key, s.key):
                        annotations[s.id] = " ✓ relative key"

        return format_sample_results(filtered[:8], header, annotations=annotations)
    except Exception:
        logger.exception("Error suggesting complements")
        return "An error occurred while suggesting complements."
