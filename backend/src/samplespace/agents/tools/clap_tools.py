"""CLAP semantic search tool for the sample assistant agent."""

import logging

from pydantic_ai import Agent, RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.schemas.sample import SampleSchema
from samplespace.services import embedding as embedding_service
from samplespace.services import sample as sample_service

logger = logging.getLogger(__name__)


async def search_by_description(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search for audio samples by natural language description.

    Use this tool when the user describes the sound they're looking for, e.g.,
    "warm analog pad", "punchy kick drum", "bright hi-hat". The search uses
    CLAP embeddings for text-to-audio semantic matching.

    Args:
        query: Natural language description of the desired sound. Use specific
               sonic descriptors (warm, bright, punchy, airy, etc.) for best results.
    """
    try:
        query_embedding = embedding_service.embed_text(query, ctx.deps.clap_model, ctx.deps.clap_processor)
        results = await sample_service.search_by_text(ctx.deps.db, query_embedding=query_embedding, limit=10)
        if not results:
            return "No samples found matching that description."

        return _format_results(results, query)
    except Exception:
        logger.exception("Error in CLAP search")
        return "An error occurred while searching for samples."


def _format_results(results: list[SampleSchema], query: str) -> str:
    lines = [f'Found {len(results)} samples matching "{query}":\n']
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


def register_clap_tools(agent: Agent[AgentDeps, str]) -> None:
    """Register CLAP search tools with the agent."""
    agent.tool(search_by_description)
