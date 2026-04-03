"""Preference learning tools for the sample assistant agent."""

from pydantic_ai import RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.models.pair_verdict import PairVerdict
from samplespace.services import preference as preference_service


async def show_preferences(ctx: RunContext[AgentDeps]) -> str:
    """Show what the system has learned from pair feedback.

    Displays the preference model's feature importances as a natural-language
    summary explaining which audio characteristics matter most in pairings.

    Use when the user asks about their preferences, what the system has learned,
    or how their feedback is being used.
    """
    explanation = preference_service.explain()
    if explanation is None:
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
        return (
            f"You have **{total}** verdicts, but the model hasn't been trained yet. "
            f"This may mean feature extraction is still in progress or there aren't enough "
            f"of both approvals and rejections (need at least {preference_service.MIN_PER_CLASS} of each)."
        )

    return explanation.summary
