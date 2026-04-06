"""Pair verdict tools for the sample assistant agent.

Handles presenting sample pairs for evaluation and recording user verdicts.
"""

import asyncio
import logging

from pydantic_ai import RunContext, ToolReturn
from pydantic_ai.ui.vercel_ai.response_types import DataChunk

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.formatting import sample_to_payload
from samplespace.dependencies.db import get_async_sqlalchemy_session
from samplespace.models.pair_verdict import PairVerdict
from samplespace.models.sample import Sample
from samplespace.schemas.sample import SampleSchema
from samplespace.schemas.sample_type import SampleType
from samplespace.services import embedding as embedding_service
from samplespace.services import pair_features as pair_features_service
from samplespace.services import pair_scoring as pair_scoring_service
from samplespace.services import preference as preference_service
from samplespace.services import sample as sample_service
from samplespace.services.candidate_search import build_clap_query, rerank_candidates

logger = logging.getLogger(__name__)

# Prevent background tasks from being garbage-collected before completion
_background_tasks: set[asyncio.Task[None]] = set()

_CANDIDATE_LIMIT = 15


async def present_pair(
    ctx: RunContext[AgentDeps],
    sample_id: str | None = None,
    anchor_type: str | None = None,
    candidate_type: str | None = None,
    is_loop: bool | None = None,
) -> str | ToolReturn:
    """Present a sample pair for the user to evaluate.

    Finds a complementary candidate for the given (or random) anchor sample,
    then returns a formatted pair-verdict block with side-by-side playback
    for the user to approve or reject.

    Use when the user asks to evaluate pairs, rate combinations, or train
    the system's pairing knowledge. For rapid pairing sessions, omit
    sample_id to get a random anchor each time.

    Args:
        sample_id: The anchor sample to find a pair for. If omitted, a random
                   anchor is selected (filtered by anchor_type if provided).
        anchor_type: Sample type for random anchor selection (e.g., "kick").
                     Only used when sample_id is omitted.
        candidate_type: Sample type to look for in the candidate (e.g., "snare").
        is_loop: Filter for loops (True) or one-shots (False). Infer from user
                 language: "one-shot" → False, "loops" → True. Default True for
                 pairing sessions. Omit if unspecified.
    """
    try:
        # Compute recent sample IDs once — used for both anchor and candidate exclusion
        recent_ids: set[str] = set()
        if ctx.deps.thread_id:
            recent_ids = set(await PairVerdict.get_recent_sample_ids(ctx.deps.db, ctx.deps.thread_id))

        if sample_id:
            anchor = await Sample.get(ctx.deps.db, sample_id)
            if anchor is None:
                return f"Sample {sample_id} not found."
        else:
            anchor = await _pick_random_anchor(ctx, anchor_type, is_loop=is_loop, exclude_ids=recent_ids)
            if anchor is None:
                return f"No {anchor_type or 'library'} samples found to use as anchor."
            sample_id = anchor.id

        candidates = await _find_candidates(ctx, sample_id, candidate_type, is_loop=is_loop, exclude_ids=recent_ids)

        if not candidates:
            if candidate_type:
                return f"No {candidate_type} candidates found to pair with this sample."
            return "No candidates found. The sample may not have a CNN embedding."

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
        song_ctx = ctx.deps.song_context
        return _format_pair_verdict(
            anchor_schema,
            best_candidate,
            best_score.overall,
            best_score.summary,
            target_key=song_ctx.key if song_ctx else None,
            target_bpm=song_ctx.bpm if song_ctx else None,
        )

    except Exception:
        logger.exception("Error presenting pair")
        return "An error occurred while finding a pair to evaluate."


async def _pick_random_anchor(
    ctx: RunContext[AgentDeps],
    anchor_type: str | None,
    *,
    is_loop: bool | None = None,
    exclude_ids: set[str] | None = None,
) -> Sample | None:
    """Pick a random anchor sample, avoiding recently evaluated samples."""
    resolved_type: str | None = None
    if anchor_type:
        try:
            resolved_type = SampleType(anchor_type.lower())
        except ValueError:
            pass

    return await Sample.get_random(
        ctx.deps.db, sample_type=resolved_type, is_loop=is_loop, exclude_ids=list(exclude_ids or [])
    )


async def _find_candidates(
    ctx: RunContext[AgentDeps],
    sample_id: str,
    candidate_type: str | None,
    *,
    is_loop: bool | None = None,
    exclude_ids: set[str] | None = None,
) -> list[SampleSchema]:
    """Find candidate samples for pairing.

    When candidate_type is specified, uses CLAP search with song context
    (same approach as kit builder) for context-aware cross-type retrieval.
    When no type is specified, falls back to CNN similarity for finding
    interesting spectrally-related pairs.
    """
    if candidate_type:
        return await _find_candidates_by_clap(ctx, sample_id, candidate_type, is_loop=is_loop, exclude_ids=exclude_ids)
    return await _find_candidates_by_cnn(ctx, sample_id, exclude_ids=exclude_ids)


async def _find_candidates_by_clap(
    ctx: RunContext[AgentDeps],
    sample_id: str,
    candidate_type: str,
    *,
    is_loop: bool | None = None,
    exclude_ids: set[str] | None = None,
) -> list[SampleSchema]:
    """CLAP-based retrieval with song context — used when a specific type is requested."""
    song_ctx = ctx.deps.song_context
    vibe = song_ctx.vibe if song_ctx else None
    genre = song_ctx.genre if song_ctx else None

    query = build_clap_query(candidate_type, vibe=vibe, genre=genre, song_context=song_ctx)
    query_embedding = await asyncio.to_thread(
        embedding_service.embed_text, query, ctx.deps.clap_model, ctx.deps.clap_processor
    )

    # Fetch extra to compensate for exclusions
    fetch_limit = _CANDIDATE_LIMIT + len(exclude_ids) + 1 if exclude_ids else _CANDIDATE_LIMIT

    results = await sample_service.search_by_text(
        ctx.deps.db,
        query_embedding=query_embedding,
        sample_type=candidate_type,
        is_loop=is_loop,
        exclude_source="upload",
        limit=fetch_limit,
    )

    if not results:
        return []

    # Filter out the anchor and recently evaluated samples
    skip = {sample_id} | (exclude_ids or set())
    filtered = [r for r in results if r.id not in skip]

    # Rerank with song context BPM/key if available
    return rerank_candidates(filtered, candidate_type, song_ctx, limit=_CANDIDATE_LIMIT)


async def _find_candidates_by_cnn(
    ctx: RunContext[AgentDeps],
    sample_id: str,
    *,
    exclude_ids: set[str] | None = None,
) -> list[SampleSchema]:
    """CNN similarity — used when no specific type is requested."""
    fetch_limit = _CANDIDATE_LIMIT + len(exclude_ids) if exclude_ids else _CANDIDATE_LIMIT
    similar_results = await sample_service.find_similar_by_cnn(ctx.deps.db, sample_id=sample_id, limit=fetch_limit)
    candidates = [r.sample for r in similar_results]
    if exclude_ids:
        candidates = [c for c in candidates if c.id not in exclude_ids]
    return candidates[:_CANDIDATE_LIMIT]


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

        score = await pair_scoring_service.score_pair(ctx.deps.db, sample_a_id, sample_b_id)

        verdict = await PairVerdict.create(
            ctx.deps.db,
            thread_id=thread_id,
            sample_a_id=sample_a_id,
            sample_b_id=sample_b_id,
            verdict=approved,
            pair_score=score.overall,
            pair_score_detail=score.model_dump(),
        )
        # Commit so the background task's separate session can see the verdict
        await ctx.deps.db.commit()

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

            # Check if we should retrain the preference model
            total = await PairVerdict.count_all(db)
            if preference_service.should_retrain(total):
                logger.info(f"Retrain threshold met ({total} verdicts), training preference model")
                await preference_service.train(db)

    except Exception:
        logger.exception(f"Background feature extraction failed for verdict {verdict_id}")


def _format_pair_verdict(
    sample_a: SampleSchema,
    sample_b: SampleSchema,
    pair_score: float,
    summary: str,
    *,
    target_key: str | None = None,
    target_bpm: int | None = None,
) -> ToolReturn:
    payload: dict[str, object] = {
        "sample_a": sample_to_payload(sample_a),
        "sample_b": sample_to_payload(sample_b),
        "pair_score": round(pair_score, 2),
        "summary": summary,
    }
    if target_key or target_bpm:
        payload["song_context"] = {"key": target_key, "bpm": target_bpm}
    intro = (
        "Here's a pair to evaluate — listen to both and let me know if they work together:\n"
        f"- Sample A: {sample_a.id} ({sample_a.filename})\n"
        f"- Sample B: {sample_b.id} ({sample_b.filename})"
    )
    return ToolReturn(
        return_value=intro,
        metadata=DataChunk(type="data-pair-verdict", data=payload),
    )
