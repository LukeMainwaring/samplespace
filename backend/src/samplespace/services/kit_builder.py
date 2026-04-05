"""Kit builder service — greedy assembly of multi-sample kits.

Assembles a kit by retrieving candidates per type via CLAP search,
then greedily selecting samples that maximize pairwise compatibility
while maintaining spectral diversity.
"""

import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from transformers import ClapModel, ClapProcessor

from samplespace.models.sample import Sample
from samplespace.schemas.kit import KitResult, KitSlot, PairwiseEntry
from samplespace.schemas.sample import SampleSchema
from samplespace.schemas.sample_type import SAMPLE_TYPES, SampleType
from samplespace.schemas.thread import SongContext
from samplespace.services import embedding as embedding_service
from samplespace.services import music_theory as music_theory_service
from samplespace.services import pair_scoring as pair_scoring_service
from samplespace.services import sample as sample_service
from samplespace.services.candidate_search import bpm_compatibility, build_clap_query, rerank_candidates
from samplespace.services.pair_scoring import DEFAULT_TYPE_SCORE, TYPE_COMPLEMENTARITY, cosine_similarity

logger = logging.getLogger(__name__)

_VALID_TYPES = set(SAMPLE_TYPES)


def _resolve_type(raw: str) -> str | None:
    """Resolve a free-form type string to a valid SampleType value.

    Handles cases like "drum loop" -> "drum", "synth lead" -> "synth".
    """
    normalized = raw.strip().lower()
    if normalized in _VALID_TYPES:
        return normalized
    # Check if any valid type is a prefix of the input (e.g. "drum loop" -> "drum")
    for t in _VALID_TYPES:
        if normalized.startswith(t):
            return t
    logger.warning(f"Kit builder: unrecognized type '{raw}', skipping")
    return None


# Default kit template
DEFAULT_TYPES = [SampleType.KICK, SampleType.SNARE, SampleType.HIHAT, SampleType.BASS, SampleType.PAD]

# Candidates to retrieve per type slot (larger pool for re-ranking)
CANDIDATES_PER_TYPE = 20

# Final candidates per type after re-ranking with song context
RERANK_LIMIT = 10

# Weight for CNN diversity penalty during greedy selection
DIVERSITY_ALPHA = 0.15

# Tonal types that have meaningful key/BPM — used for greedy ordering tiebreaker.
# Lower index = filled earlier (more constrained).
_TONAL_PRIORITY: dict[str, int] = {
    SampleType.PAD: 0,
    SampleType.BASS: 1,
    SampleType.SYNTH: 2,
    SampleType.VOCAL: 3,
    SampleType.KICK: 4,
    SampleType.SNARE: 5,
    SampleType.HIHAT: 6,
}


async def build_kit(
    db: AsyncSession,
    *,
    clap_model: ClapModel,
    clap_processor: ClapProcessor,
    types: list[str] | None = None,
    song_context: SongContext | None = None,
    vibe: str | None = None,
    genre: str | None = None,
    replacements: dict[str, str] | None = None,
) -> KitResult:
    """Assemble a kit using greedy pairwise optimization.

    Phase 1: Retrieve candidates per type via CLAP search (or use pinned samples).
    Phase 2: Greedy assembly using fast inline scoring.
    Phase 3: Final scoring via full pair_scoring service.

    Args:
        replacements: Map of {sample_type: sample_id} to pin specific samples
                      into slots, skipping CLAP search for those types.
    """
    # Resolve free-form type names to valid SampleType values
    raw_types = types or DEFAULT_TYPES
    kit_types = []
    for t in raw_types:
        resolved = _resolve_type(t)
        if resolved and resolved not in kit_types:
            kit_types.append(resolved)

    effective_vibe = vibe or (song_context.vibe if song_context else None)
    effective_genre = genre or (song_context.genre if song_context else None)

    # Resolve pinned replacements upfront
    pinned: dict[str, SampleSchema] = {}
    if replacements:
        for sample_type, sample_id in replacements.items():
            sample = await Sample.get(db, sample_id)
            if sample:
                pinned[sample_type.lower()] = SampleSchema.model_validate(sample)
            else:
                logger.warning(
                    f"Kit builder: pinned sample '{sample_id}' for type '{sample_type}' not found, will search"
                )

    candidates_by_type: dict[str, list[SampleSchema]] = {}
    all_candidate_ids: list[str] = []
    skipped_types: list[str] = []

    for sample_type in kit_types:
        # Use pinned sample if provided for this type
        if sample_type.lower() in pinned:
            candidates_by_type[sample_type] = [pinned[sample_type.lower()]]
            all_candidate_ids.append(pinned[sample_type.lower()].id)
            continue

        query = build_clap_query(sample_type, effective_vibe, effective_genre, song_context)
        query_embedding = await asyncio.to_thread(embedding_service.embed_text, query, clap_model, clap_processor)

        results = await sample_service.search_by_text(
            db,
            query_embedding=query_embedding,
            sample_type=sample_type,
            is_loop=True,
            exclude_source="upload",
            limit=CANDIDATES_PER_TYPE,
        )

        if not results:
            skipped_types.append(sample_type)
            logger.info(f"Kit builder: no candidates found for type '{sample_type}', skipping")
            continue

        results = rerank_candidates(results, sample_type, song_context)
        candidates_by_type[sample_type] = results
        all_candidate_ids.extend(r.id for r in results)

    if not candidates_by_type:
        return KitResult(
            slots=[],
            overall_score=0.0,
            pairwise_scores=[],
            vibe=effective_vibe,
            genre=effective_genre,
            skipped_types=skipped_types,
        )

    all_samples = await Sample.get_many(db, all_candidate_ids)
    cnn_embeddings: dict[str, list[float]] = {s.id: s.cnn_embedding for s in all_samples if s.cnn_embedding is not None}

    # Most-constrained-first, tonal elements as tiebreaker
    ordered_types = sorted(
        candidates_by_type.keys(),
        key=lambda t: (len(candidates_by_type[t]), _TONAL_PRIORITY.get(t.lower(), 99)),
    )

    selected: list[tuple[str, SampleSchema]] = []  # (type, sample)
    selected_ids: set[str] = set()

    for sample_type in ordered_types:
        candidates = [c for c in candidates_by_type[sample_type] if c.id not in selected_ids]
        if not candidates:
            skipped_types.append(sample_type)
            continue

        if not selected:
            # First slot: best CLAP match (already sorted by relevance)
            best = candidates[0]
        else:
            best = _pick_best_candidate(candidates, selected, cnn_embeddings)

        selected.append((sample_type, best))
        selected_ids.add(best.id)

    if not selected:
        return KitResult(
            slots=[],
            overall_score=0.0,
            pairwise_scores=[],
            vibe=effective_vibe,
            genre=effective_genre,
            skipped_types=skipped_types,
        )

    type_order = {t: i for i, t in enumerate(kit_types)}
    selected.sort(key=lambda x: type_order.get(x[0], 99))

    pairwise_scores: list[PairwiseEntry] = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            ps = await pair_scoring_service.score_pair(db, selected[i][1].id, selected[j][1].id)
            pairwise_scores.append(PairwiseEntry(slot_a=i, slot_b=j, score=ps.overall, summary=ps.summary))

    overall = sum(p.score for p in pairwise_scores) / len(pairwise_scores) if pairwise_scores else 0.0

    slot_compat: dict[int, list[float]] = {i: [] for i in range(len(selected))}
    for p in pairwise_scores:
        slot_compat[p.slot_a].append(p.score)
        slot_compat[p.slot_b].append(p.score)

    slots = [
        KitSlot(
            position=i,
            requested_type=sample_type,
            sample=sample,
            compatibility_score=round(sum(slot_compat[i]) / len(slot_compat[i]) if slot_compat[i] else 0.0, 2),
        )
        for i, (sample_type, sample) in enumerate(selected)
    ]

    return KitResult(
        slots=slots,
        overall_score=round(overall, 2),
        pairwise_scores=pairwise_scores,
        vibe=effective_vibe,
        genre=effective_genre,
        skipped_types=skipped_types,
    )


def _pick_best_candidate(
    candidates: list[SampleSchema],
    selected: list[tuple[str, SampleSchema]],
    cnn_embeddings: dict[str, list[float]],
) -> SampleSchema:
    best: SampleSchema = candidates[0]
    best_score = -1.0

    for candidate in candidates:
        # Average fast compatibility with all selected samples
        compat_scores = [_fast_compatibility(candidate, sel_sample) for _, sel_sample in selected]
        avg_compat = sum(compat_scores) / len(compat_scores)

        # CNN diversity penalty — penalize spectral similarity to all selected samples
        # to encourage timbral variety across the kit
        diversity_penalty = 0.0
        cand_cnn = cnn_embeddings.get(candidate.id)
        if cand_cnn is not None:
            for _, sel_sample in selected:
                sel_cnn = cnn_embeddings.get(sel_sample.id)
                if sel_cnn is not None:
                    diversity_penalty += cosine_similarity(cand_cnn, sel_cnn)

        composite = avg_compat - DIVERSITY_ALPHA * diversity_penalty
        if composite > best_score:
            best_score = composite
            best = candidate

    return best


def _fast_compatibility(sample_a: SampleSchema, sample_b: SampleSchema) -> float:
    scores: list[float] = []

    # Type complementarity
    if sample_a.sample_type and sample_b.sample_type:
        pair_key = frozenset({sample_a.sample_type.lower(), sample_b.sample_type.lower()})
        scores.append(TYPE_COMPLEMENTARITY.get(pair_key, DEFAULT_TYPE_SCORE))

    # Key compatibility (only for loops with keys)
    if sample_a.is_loop and sample_b.is_loop and sample_a.key and sample_b.key:
        value, _ = music_theory_service.key_compatibility_score(sample_a.key, sample_b.key)
        scores.append(value)

    # BPM compatibility (only for loops with BPMs)
    if sample_a.is_loop and sample_b.is_loop and sample_a.bpm and sample_b.bpm:
        scores.append(bpm_compatibility(sample_a.bpm, sample_b.bpm))

    return sum(scores) / len(scores) if scores else 0.5
