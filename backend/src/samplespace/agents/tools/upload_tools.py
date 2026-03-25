"""Upload similarity tool for the sample assistant agent."""

import logging

from pydantic_ai import Agent, RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.models.sample import Sample
from samplespace.schemas.sample import SampleSchema

logger = logging.getLogger(__name__)


async def find_similar_to_upload(ctx: RunContext[AgentDeps], sample_id: str) -> str:
    """Find library samples similar to an uploaded reference track using CLAP embeddings.

    Use this tool when the user has uploaded a WAV file (a song, snippet, or reference
    track) and wants to find similar samples from the splice library. The search compares
    the uploaded sample's audio embedding against the library.

    Args:
        sample_id: The ID of the uploaded sample to find similar matches for.
    """
    try:
        sample = await Sample.get(ctx.deps.db, sample_id)
        if sample is None:
            return f"Sample {sample_id} not found."

        if sample.source != "upload":
            return f"Sample {sample_id} is not an uploaded sample. Use find_similar_samples for library samples."

        if sample.clap_embedding is None:
            return f"Sample {sample_id} does not have a CLAP embedding. It may not have been processed correctly."

        # Search the library (exclude uploads) using the uploaded sample's CLAP embedding
        results = await Sample.search_by_clap(
            ctx.deps.db,
            sample.clap_embedding,
            exclude_source="upload",
            limit=10,
        )

        if not results:
            return "No similar library samples found."

        formatted = [SampleSchema.model_validate(s) for s in results]
        return _format_results(formatted, sample.filename)
    except Exception:
        logger.exception("Error in upload similarity search")
        return "An error occurred while finding similar samples."


def _format_results(results: list[SampleSchema], source_name: str) -> str:
    lines = [f'Found {len(results)} library samples similar to uploaded file "{source_name}":\n']
    for i, s in enumerate(results, 1):
        parts = [f"{i}. **{s.filename}**"]
        if s.sample_type:
            parts.append(f"type={s.sample_type}")
        parts.append("loop" if s.is_loop else "one-shot")
        if s.is_loop:
            if s.key:
                parts.append(f"key={s.key}")
            if s.bpm and s.bpm > 0:
                parts.append(f"bpm={s.bpm}")
        if s.duration:
            parts.append(f"duration={s.duration:.1f}s")
        parts.append(f"id={s.id}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def register_upload_tools(agent: Agent[AgentDeps, str]) -> None:
    """Register upload similarity tools with the agent."""
    agent.tool(find_similar_to_upload)
