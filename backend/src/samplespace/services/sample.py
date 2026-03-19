"""Sample business logic — CRUD operations and search queries."""

import logging
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from samplespace.models.sample import Sample
from samplespace.schemas.sample import SampleListResponse, SampleSchema
from samplespace.services.audio_analysis import analyze_audio

logger = logging.getLogger(__name__)


async def create_sample(
    db: AsyncSession,
    *,
    filename: str,
    file_path: str,
    sample_type: str | None = None,
) -> Sample:
    """Create a sample from an audio file: analyze metadata and persist."""
    metadata = analyze_audio(file_path)

    sample = Sample(
        id=str(uuid.uuid4()),
        filename=filename,
        key=metadata["key"],
        bpm=metadata["bpm"],
        duration=metadata["duration"],
        sample_type=sample_type,
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
) -> SampleListResponse:
    """List samples with pagination."""
    total_result = await db.execute(select(func.count()).select_from(Sample))
    total = total_result.scalar_one()

    result = await db.execute(
        select(Sample).order_by(Sample.created_at.desc()).limit(limit).offset(offset),
    )
    samples = result.scalars().all()

    return SampleListResponse(
        samples=[SampleSchema.model_validate(s) for s in samples],
        total=total,
    )


async def get_sample_by_id(db: AsyncSession, sample_id: str) -> Sample | None:
    """Get a single sample by ID."""
    result = await db.execute(select(Sample).where(Sample.id == sample_id))
    return result.scalar_one_or_none()


async def search_by_text(
    db: AsyncSession,
    *,
    query_embedding: list[float],
    key: str | None = None,
    bpm_min: float | None = None,
    bpm_max: float | None = None,
    sample_type: str | None = None,
    limit: int = 20,
) -> list[SampleSchema]:
    """Search samples by CLAP text embedding with optional metadata filters.

    Uses pgvector cosine distance for semantic similarity ranking.
    """
    # Cosine distance: lower = more similar
    distance = Sample.clap_embedding.cosine_distance(cast(query_embedding, Vector(512)))

    stmt = select(Sample, cast(distance, Float).label("distance")).where(Sample.clap_embedding.is_not(None))

    if key is not None:
        stmt = stmt.where(Sample.key == key)
    if bpm_min is not None:
        stmt = stmt.where(Sample.bpm >= bpm_min)
    if bpm_max is not None:
        stmt = stmt.where(Sample.bpm <= bpm_max)
    if sample_type is not None:
        stmt = stmt.where(Sample.sample_type == sample_type)

    stmt = stmt.order_by(distance).limit(limit)

    result = await db.execute(stmt)
    rows = result.all()

    return [SampleSchema.model_validate(row.Sample) for row in rows]
