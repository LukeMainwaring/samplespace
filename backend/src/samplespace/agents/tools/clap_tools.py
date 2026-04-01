import logging

from pydantic_ai import RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.formatting import format_sample_results
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
        enriched_query = query
        if ctx.deps.song_context and ctx.deps.song_context.vibe:
            enriched_query = f"{query}, {ctx.deps.song_context.vibe}"

        query_embedding = embedding_service.embed_text(enriched_query, ctx.deps.clap_model, ctx.deps.clap_processor)
        results = await sample_service.search_by_text(ctx.deps.db, query_embedding=query_embedding, limit=10)
        if not results:
            return "No samples found matching that description."

        return format_sample_results(
            results,
            f'Found {len(results)} samples matching "{query}":',
            include_duration=True,
        )
    except Exception:
        logger.exception("Error in CLAP search")
        return "An error occurred while searching for samples."
