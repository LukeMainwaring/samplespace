"""Audio analysis and music theory tools for the sample assistant agent."""

import logging

from pydantic_ai import Agent, RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.services import sample as sample_service

logger = logging.getLogger(__name__)

# Circle of fifths for key compatibility
CIRCLE_OF_FIFTHS = [
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
    "G#",
    "D#",
    "A#",
    "F",
]

# Relative major/minor pairs
RELATIVE_PAIRS = {
    "C major": "A minor",
    "G major": "E minor",
    "D major": "B minor",
    "A major": "F# minor",
    "E major": "C# minor",
    "B major": "G# minor",
    "F# major": "D# minor",
    "C# major": "A# minor",
    "G# major": "F minor",
    "D# major": "C minor",
    "A# major": "G minor",
    "F major": "D minor",
}
# Add reverse mappings
RELATIVE_PAIRS.update({v: k for k, v in RELATIVE_PAIRS.items()})


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
        lines.append(f"- Duration: {sample.duration:.1f}s" if sample.duration else "- Duration: unknown")
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

    # Check relative major/minor
    if RELATIVE_PAIRS.get(key1) == key2:
        return (
            f"**{key1}** and **{key2}** are relative major/minor pairs — highly compatible! They share the same notes."
        )

    # Check circle of fifths proximity
    root1 = key1.split()[0] if " " in key1 else key1
    root2 = key2.split()[0] if " " in key2 else key2

    if root1 in CIRCLE_OF_FIFTHS and root2 in CIRCLE_OF_FIFTHS:
        idx1 = CIRCLE_OF_FIFTHS.index(root1)
        idx2 = CIRCLE_OF_FIFTHS.index(root2)
        distance = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))

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

    return f"Could not determine compatibility between **{key1}** and **{key2}**."


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
        desired_type: Optional type filter (e.g., "bass", "pad", "lead").
    """
    try:
        source = await sample_service.get_sample_by_id(ctx.deps.db, sample_id)
        if source is None:
            return f"Sample {sample_id} not found."

        # Search with type filter if specified
        from samplespace.services.embedding import embed_text

        query = f"complement for {source.sample_type or 'sample'}"
        if desired_type:
            query = f"{desired_type} that complements {source.sample_type or 'sample'}"

        query_embedding = embed_text(query, ctx.deps.clap_model, ctx.deps.clap_processor)

        results = await sample_service.search_by_text(
            ctx.deps.db,
            query_embedding=query_embedding,
            sample_type=desired_type,
            limit=10,
        )

        # Filter out the source sample and rank by key compatibility
        filtered = [r for r in results if r.id != sample_id]

        if not filtered:
            return "No complementary samples found."

        if source.is_loop:
            header = f"Samples that complement **{source.filename}** (key: {source.key or 'unknown'}):\n"
        else:
            header = f"Samples that complement **{source.filename}** (one-shot):\n"
        lines = [header]
        for i, s in enumerate(filtered[:8], 1):
            compat = ""
            if source.is_loop and source.key and s.key:
                if source.key == s.key:
                    compat = " ✓ same key"
                elif RELATIVE_PAIRS.get(source.key) == s.key:
                    compat = " ✓ relative key"
            parts = [f"{i}. **{s.filename}**"]
            if s.sample_type:
                parts.append(f"type={s.sample_type}")
            if s.is_loop and s.key:
                parts.append(f"key={s.key}{compat}")
            parts.append(f"id={s.id}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)
    except Exception:
        logger.exception("Error suggesting complements")
        return "An error occurred while suggesting complements."


def register_analysis_tools(agent: Agent[AgentDeps, str]) -> None:
    """Register analysis and music theory tools with the agent."""
    agent.tool(analyze_sample)
    agent.tool(check_key_compatibility)
    agent.tool(suggest_complement)
