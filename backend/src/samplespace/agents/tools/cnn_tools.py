import logging

from pydantic_ai import RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.formatting import format_sample_results
from samplespace.services import sample as sample_service

logger = logging.getLogger(__name__)


async def find_similar_samples(ctx: RunContext[AgentDeps], sample_id: str) -> str:
    """Find samples that sound similar to a given sample using CNN embeddings.

    Use this tool when the user has a specific sample and wants to find others
    that are sonically similar, e.g., "find something similar to this kick" or
    "what sounds like this pad?".

    Args:
        sample_id: The ID of the sample to find similar matches for.
    """
    try:
        results = await sample_service.find_similar_by_cnn(ctx.deps.db, sample_id=sample_id, limit=8)
        if not results:
            return "No similar samples found. The sample may not have a CNN embedding yet."

        source = await sample_service.get_sample_by_id(ctx.deps.db, sample_id)
        source_name = source.filename if source else sample_id
        return format_sample_results(
            [r.sample for r in results],
            f'Found {len(results)} samples similar to "{source_name}":',
        )
    except Exception:
        logger.exception("Error in CNN similarity search")
        return "An error occurred while finding similar samples."
