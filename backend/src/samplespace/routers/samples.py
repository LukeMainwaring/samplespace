from fastapi import HTTPException
from fastapi.routing import APIRouter
from sqlalchemy import func, select

from samplespace.dependencies.db import AsyncPostgresSessionDep
from samplespace.models.sample import Sample
from samplespace.schemas.sample import SampleListResponse, SampleSchema

samples_router = APIRouter(
    prefix="/samples",
    tags=["samples"],
)


@samples_router.get("/")
async def list_samples(
    db: AsyncPostgresSessionDep,
    limit: int = 50,
    offset: int = 0,
) -> SampleListResponse:
    """List all samples with pagination."""
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


@samples_router.get("/{sample_id}")
async def get_sample(
    sample_id: str,
    db: AsyncPostgresSessionDep,
) -> SampleSchema:
    """Get a single sample by ID."""
    result = await db.execute(select(Sample).where(Sample.id == sample_id))
    sample = result.scalar_one_or_none()

    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    return SampleSchema.model_validate(sample)
