"""Sample business logic — CRUD operations and search queries."""

import logging
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from samplespace.core.config import get_settings
from samplespace.models.sample import Sample
from samplespace.schemas.sample import SampleListResponse, SampleSchema
from samplespace.services.audio_analysis import analyze_and_classify

logger = logging.getLogger(__name__)


async def create_sample(
    db: AsyncSession,
    *,
    filename: str,
    file_path: str,
    relative_path: str,
    source: str = "local",
    sample_type: str | None = None,
    pack_name: str | None = None,
) -> Sample:
    """Create a sample from an audio file: analyze metadata and persist."""
    result = analyze_and_classify(file_path)

    sample = Sample(
        id=str(uuid.uuid4()),
        filename=filename,
        relative_path=relative_path,
        source=source,
        pack_name=pack_name,
        key=result.metadata.key,
        bpm=result.metadata.bpm,
        duration=result.metadata.duration,
        sample_type=sample_type,
        is_loop=result.is_loop,
    )
    db.add(sample)
    await db.flush()

    logger.info(f"Created sample {sample.id}: {filename}")
    return sample


async def get_samples(
    db: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    source: str | None = None,
) -> SampleListResponse:
    """List samples with pagination."""
    samples, total = await Sample.get_all(db, limit=limit, offset=offset, source=source)

    return SampleListResponse(
        samples=[SampleSchema.model_validate(s) for s in samples],
        total=total,
    )


async def get_sample_by_id(db: AsyncSession, sample_id: str) -> Sample | None:
    """Get a single sample by ID."""
    return await Sample.get(db, sample_id)


async def search_by_text(
    db: AsyncSession,
    *,
    query_embedding: list[float],
    key: str | None = None,
    bpm_min: int | None = None,
    bpm_max: int | None = None,
    sample_type: str | None = None,
    is_loop: bool | None = None,
    exclude_source: str | None = None,
    limit: int = 20,
) -> list[SampleSchema]:
    """Search samples by CLAP text embedding with optional metadata filters.

    Uses pgvector cosine distance for semantic similarity ranking.
    """
    results = await Sample.search_by_clap(
        db,
        query_embedding,
        key=key,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        sample_type=sample_type,
        is_loop=is_loop,
        exclude_source=exclude_source,
        limit=limit,
    )

    return [SampleSchema.model_validate(s) for s in results]


async def find_similar_by_cnn(
    db: AsyncSession,
    *,
    sample_id: str,
    limit: int = 10,
) -> list[SampleSchema]:
    """Find similar samples using CNN embedding nearest neighbors.

    Uses pgvector cosine distance on the 128-dim CNN embeddings.
    """
    source = await Sample.get(db, sample_id)
    if source is None or source.cnn_embedding is None:
        return []

    results = await Sample.find_similar_by_cnn(
        db,
        source.cnn_embedding,
        exclude_id=sample_id,
        limit=limit,
    )

    return [SampleSchema.model_validate(s) for s in results]


def find_audio_file(sample: Sample) -> Path | None:
    """Locate an audio file using source-aware path resolution."""
    settings = get_settings()

    if sample.source == "splice" and settings.SPLICE_DIR:
        candidate = Path(settings.SPLICE_DIR) / sample.relative_path
        if candidate.exists():
            return candidate
        return None

    if sample.source == "upload":
        candidate = Path(settings.UPLOAD_DIR) / sample.relative_path
        return candidate if candidate.exists() else None

    # Local source: resolve against SAMPLES_DIR
    samples_dir = Path(settings.SAMPLES_DIR)
    candidate = samples_dir / sample.relative_path
    if candidate.exists():
        return candidate

    # Fallback: try rglob for backwards compatibility
    matches = list(samples_dir.rglob(sample.filename))
    return matches[0] if matches else None
