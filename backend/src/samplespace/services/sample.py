import logging
import uuid
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from samplespace.core.config import get_settings
from samplespace.core.paths import SAMPLES_DIR, SPECTROGRAMS_DIR, TRANSFORMS_DIR, UPLOADS_DIR
from samplespace.models.pair_verdict import PairVerdict
from samplespace.models.sample import Sample, SampleNotFound
from samplespace.schemas.sample import (
    ListSamplesParams,
    SampleListResponse,
    SampleSchema,
    SampleUpdateSchema,
    SimilarSampleSchema,
)
from samplespace.services.audio_analysis import analyze_and_classify

logger = logging.getLogger(__name__)


async def create_sample(
    db: AsyncSession,
    *,
    filename: str,
    file_path: str,
    relative_path: str,
    source: str = "library",
    sample_type: str | None = None,
    pack_name: str | None = None,
) -> Sample:
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
    params: ListSamplesParams,
) -> SampleListResponse:
    samples, total = await Sample.get_all(db, params)

    return SampleListResponse(
        samples=[SampleSchema.model_validate(s) for s in samples],
        total=total,
    )


async def get_sample_by_id(db: AsyncSession, sample_id: str) -> Sample | None:
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
) -> list[SimilarSampleSchema]:
    source = await Sample.get(db, sample_id)
    if source is None or source.cnn_embedding is None:
        return []

    results = await Sample.find_similar_by_cnn(
        db,
        source.cnn_embedding,
        exclude_id=sample_id,
        limit=limit,
    )

    return [SimilarSampleSchema(sample=SampleSchema.model_validate(s), distance=d) for s, d in results]


async def update_sample(
    db: AsyncSession,
    sample_id: str,
    updates: SampleUpdateSchema,
) -> SampleSchema:
    sample = await Sample.get(db, sample_id)
    if sample is None:
        raise SampleNotFound()
    if sample.source != "upload":
        raise HTTPException(status_code=403, detail="Only uploaded samples can be updated")

    fields = updates.model_dump(exclude_unset=True)
    if fields:
        updated = await Sample.update(db, sample_id, **fields)
        if updated is None:
            raise SampleNotFound()
        sample = updated

    return SampleSchema.model_validate(sample)


async def delete_sample(db: AsyncSession, sample_id: str) -> None:
    sample = await Sample.get(db, sample_id)
    if sample is None:
        raise SampleNotFound()
    if sample.source != "upload":
        raise HTTPException(status_code=403, detail="Only uploaded samples can be deleted")

    # Clean up related verdicts
    await PairVerdict.delete_by_sample(db, sample_id)

    # Clean up disk artifacts
    audio_path = UPLOADS_DIR / sample.relative_path
    audio_path.unlink(missing_ok=True)

    for mode in ("full", "cnn"):
        (SPECTROGRAMS_DIR / f"{sample_id}_{mode}.png").unlink(missing_ok=True)

    for transform_file in TRANSFORMS_DIR.glob(f"{sample_id}_*"):
        transform_file.unlink(missing_ok=True)

    await Sample.delete_by_id(db, sample_id)
    logger.info(f"Deleted sample {sample_id}: {sample.filename}")


def find_audio_file(sample: Sample) -> Path | None:
    """Locate an audio file using source-aware path resolution."""
    settings = get_settings()

    if sample.source == "library" and settings.SAMPLE_LIBRARY_DIR:
        candidate = Path(settings.SAMPLE_LIBRARY_DIR) / sample.relative_path
        if candidate.exists():
            return candidate
        return None

    if sample.source == "upload":
        candidate = UPLOADS_DIR / sample.relative_path
        return candidate if candidate.exists() else None

    candidate = SAMPLES_DIR / sample.relative_path
    if candidate.exists():
        return candidate

    matches = list(SAMPLES_DIR.rglob(sample.filename))
    return matches[0] if matches else None
