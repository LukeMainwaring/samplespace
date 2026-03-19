from fastapi import HTTPException
from fastapi.routing import APIRouter

from samplespace.dependencies.db import AsyncPostgresSessionDep
from samplespace.schemas.sample import SampleListResponse, SampleSchema
from samplespace.services import sample as sample_service

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
    return await sample_service.get_samples(db, limit=limit, offset=offset)


@samples_router.get("/{sample_id}")
async def get_sample(
    sample_id: str,
    db: AsyncPostgresSessionDep,
) -> SampleSchema:
    """Get a single sample by ID."""
    sample = await sample_service.get_sample_by_id(db, sample_id)

    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    return SampleSchema.model_validate(sample)
