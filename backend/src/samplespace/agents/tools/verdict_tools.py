"""Pair verdict tools for the sample assistant agent.

Handles presenting sample pairs for evaluation and recording user verdicts.
"""

import asyncio
import json
import logging

from pydantic_ai import Agent, RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.dependencies.db import get_async_sqlalchemy_session
from samplespace.models.pair_verdict import PairVerdict
from samplespace.models.sample import Sample
from samplespace.schemas.sample import SampleSchema
from samplespace.services import pair_features as pair_features_service
from samplespace.services import pair_scoring as pair_scoring_service
from samplespace.services import sample as sample_service

logger = logging.getLogger(__name__)

# Prevent background tasks from being garbage-collected before completion
_background_tasks: set[asyncio.Task[None]] = set()


async def present_pair(
    ctx: RunContext[AgentDeps],
    sample_id: str,
    candidate_type: str | None = None,
) -> str:
    """Present a sample pair for the user to evaluate.

    Finds a complementary candidate for the given sample using CNN similarity
    and pair scoring, then returns a formatted pair-verdict block with
    side-by-side playback for the user to approve or reject.

    Use when the user asks to evaluate pairs, rate combinations, or train
    the system's pairing knowledge.

    Args:
        sample_id: The anchor sample to find a pair for.
        candidate_type: Optional sample type to look for (e.g., "pad", "lead").
    """
    try:
        anchor = await Sample.get(ctx.deps.db, sample_id)
        if anchor is None:
            return f"Sample {sample_id} not found."

        # Find candidates via CNN similarity
        candidates = await sample_service.find_similar_by_cnn(ctx.deps.db, sample_id=sample_id, limit=15)

        if not candidates:
            return "No candidates found. The sample may not have a CNN embedding."

        # Filter by type if requested
        if candidate_type:
            type_lower = candidate_type.lower()
            typed = [c for c in candidates if c.sample_type and c.sample_type.lower() == type_lower]
            if typed:
                candidates = typed

        # Score each candidate and pick one in the "interesting" range (0.4-0.8)
        best_candidate = None
        best_score = None
        best_distance_from_target = float("inf")

        for candidate in candidates:
            score = await pair_scoring_service.score_pair(ctx.deps.db, sample_id, candidate.id)
            # Target the 0.6 range — plausible but not obvious
            distance_from_target = abs(score.overall - 0.6)
            if distance_from_target < best_distance_from_target:
                best_distance_from_target = distance_from_target
                best_candidate = candidate
                best_score = score

        if best_candidate is None or best_score is None:
            return "Could not find a suitable pairing candidate."

        anchor_schema = SampleSchema.model_validate(anchor)
        return _format_pair_verdict(anchor_schema, best_candidate, best_score.overall, best_score.summary)

    except Exception:
        logger.exception("Error presenting pair")
        return "An error occurred while finding a pair to evaluate."


async def record_verdict(
    ctx: RunContext[AgentDeps],
    sample_a_id: str,
    sample_b_id: str,
    approved: bool,
) -> str:
    """Record the user's verdict on a sample pair.

    Persists the verdict and triggers background feature extraction.
    Always call this after the user responds to a presented pair with
    a yes/no, thumbs up/down, or approve/reject decision.

    Args:
        sample_a_id: First sample ID.
        sample_b_id: Second sample ID.
        approved: True if the user thinks they work well together, False otherwise.
    """
    try:
        thread_id = ctx.deps.thread_id
        if not thread_id:
            return "No thread context available."

        # Score the pair for the snapshot
        score = await pair_scoring_service.score_pair(ctx.deps.db, sample_a_id, sample_b_id)

        # Create the verdict
        verdict = await PairVerdict.create(
            ctx.deps.db,
            thread_id=thread_id,
            sample_a_id=sample_a_id,
            sample_b_id=sample_b_id,
            verdict=approved,
            pair_score=score.overall,
            pair_score_detail=score.model_dump(),
        )

        # Fire-and-forget background feature extraction
        task = asyncio.create_task(_extract_features_background(verdict.id, sample_a_id, sample_b_id))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        total = await PairVerdict.count_all(ctx.deps.db)
        status = "approved" if approved else "rejected"
        return f"Verdict recorded: **{status}**. You now have {total} total verdict{'s' if total != 1 else ''}."

    except Exception:
        logger.exception("Error recording verdict")
        return "An error occurred while recording the verdict."


async def _extract_features_background(
    verdict_id: int,
    sample_a_id: str,
    sample_b_id: str,
) -> None:
    """Background task: compute relational audio features and update the verdict.

    Uses its own DB session since the request session may close before
    this completes.
    """
    try:
        async with get_async_sqlalchemy_session() as db:
            sample_a = await Sample.get(db, sample_a_id)
            sample_b = await Sample.get(db, sample_b_id)

            if sample_a is None or sample_b is None:
                logger.warning(f"Skipping feature extraction: sample(s) not found for verdict {verdict_id}")
                return

            path_a = sample_service.find_audio_file(sample_a)
            path_b = sample_service.find_audio_file(sample_b)

            if path_a is None or path_b is None:
                logger.warning(f"Skipping feature extraction: audio file(s) not found for verdict {verdict_id}")
                return

            features = await asyncio.to_thread(pair_features_service.compute_pair_features, path_a, path_b)

            await PairVerdict.update_features(db, verdict_id, features)
            logger.info(f"Feature extraction complete for verdict {verdict_id}")

    except Exception:
        logger.exception(f"Background feature extraction failed for verdict {verdict_id}")


def _format_pair_verdict(
    sample_a: SampleSchema,
    sample_b: SampleSchema,
    pair_score: float,
    summary: str,
) -> str:
    """Format a pair for the pair-verdict Streamdown code fence."""
    payload = {
        "sample_a": _sample_to_payload(sample_a),
        "sample_b": _sample_to_payload(sample_b),
        "pair_score": round(pair_score, 2),
        "summary": summary,
    }
    json_str = json.dumps(payload, indent=2)
    return f"Here's a pair to evaluate — listen to both and let me know if they work together:\n\n```pair-verdict\n{json_str}\n```"


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


def register_verdict_tools(agent: Agent[AgentDeps, str]) -> None:
    """Register pair verdict tools with the agent."""
    agent.tool(present_pair)
    agent.tool(record_verdict)
