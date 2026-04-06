"""Preference learning tools for the sample assistant agent."""

import logging

from pydantic_ai import RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.models.pair_verdict import PairVerdict
from samplespace.services import preference as preference_service

logger = logging.getLogger(__name__)


async def show_preferences(ctx: RunContext[AgentDeps]) -> str:
    """Show what the system has learned from pair feedback.

    Displays the preference model's feature importances as a natural-language
    summary explaining which audio characteristics matter most in pairings.

    Use when the user asks about their preferences, what the system has learned,
    or how their feedback is being used.
    """
    explanation = preference_service.explain()
    if explanation is not None:
        return explanation.summary

    total = await PairVerdict.count_all(ctx.deps.db)
    needed = preference_service.MIN_VERDICTS
    remaining = max(0, needed - total)
    if remaining > 0:
        return (
            f"Not enough feedback yet to learn preferences. "
            f"You have **{total}** verdict{'s' if total != 1 else ''} — "
            f"need at least **{needed}** with both approvals and rejections "
            f"({remaining} more to go). Keep evaluating pairs!"
        )

    # Enough verdicts exist but no model — attempt training now
    # (training may have been skipped at a checkpoint if features weren't ready)
    logger.info(f"No preference model found with {total} verdicts — attempting training now")
    meta = await preference_service.train(ctx.deps.db)
    if meta is not None:
        explanation = preference_service.explain()
        if explanation is not None:
            return explanation.summary

    # Training failed — give a specific reason
    feature_complete = await PairVerdict.count_with_features(ctx.deps.db)
    return (
        f"You have **{total}** verdicts, but only **{feature_complete}** have completed "
        f"feature extraction (need at least **{needed}**). "
        f"Try evaluating a few more pairs to give background processing time to catch up."
    )
