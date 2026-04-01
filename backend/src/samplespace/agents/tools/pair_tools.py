import logging

from pydantic_ai import RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.schemas.pair import PairScore
from samplespace.services import pair_scoring as pair_scoring_service

logger = logging.getLogger(__name__)


async def rate_pair(ctx: RunContext[AgentDeps], sample_a_id: str, sample_b_id: str) -> str:
    """Rate the compatibility of two audio samples across multiple dimensions.

    Use this tool when the user wants to know how well two specific samples
    work together, e.g., "how compatible are these two samples?" or "will
    this kick work with that bass?".

    Returns a composite score (0-1) with breakdowns for key compatibility,
    BPM compatibility, type complementarity, and spectral distance.

    Args:
        sample_a_id: The ID of the first sample.
        sample_b_id: The ID of the second sample.
    """
    try:
        result = await pair_scoring_service.score_pair(ctx.deps.db, sample_a_id, sample_b_id)
        return _format_pair_score(result)
    except Exception:
        logger.exception("Error scoring pair")
        return "An error occurred while scoring the sample pair."


def _format_pair_score(score: PairScore) -> str:
    lines = [
        f"**Compatibility: {score.overall:.2f}/1.0**",
        "",
    ]

    for label, dim in [
        ("Key", score.key_score),
        ("BPM", score.bpm_score),
        ("Type", score.type_score),
        ("Spectral", score.spectral_score),
    ]:
        if dim is not None:
            lines.append(f"- {label}: **{dim.value}** — {dim.explanation}")

    lines.append("")
    lines.append(score.summary)
    return "\n".join(lines)
