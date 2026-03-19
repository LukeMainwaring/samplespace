"""Sample business logic — CRUD operations and search queries."""

import logging
import uuid

from sqlalchemy import func, select
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
