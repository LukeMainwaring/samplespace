from fastapi import HTTPException
from fastapi.routing import APIRouter

from samplespace.dependencies.clap import ClapModelsDep
from samplespace.dependencies.db import AsyncPostgresSessionDep
from samplespace.schemas.sample import SampleListResponse, SampleSchema, SampleSearchRequest
from samplespace.services import embedding as embedding_service
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


@samples_router.post("/search")
async def search_samples(
    body: SampleSearchRequest,
    db: AsyncPostgresSessionDep,
    clap: ClapModelsDep,
) -> list[SampleSchema]:
    """Search samples by natural language query with optional metadata filters."""
    if not body.query:
        raise HTTPException(status_code=400, detail="Query is required for search")

    query_embedding = embedding_service.embed_text(body.query, clap.model, clap.processor)

    return await sample_service.search_by_text(
        db,
        query_embedding=query_embedding,
        key=body.key,
        bpm_min=body.bpm_min,
        bpm_max=body.bpm_max,
        sample_type=body.sample_type,
        limit=body.limit,
    )


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


@samples_router.get("/{sample_id}/similar")
async def get_similar_samples(
    sample_id: str,
    db: AsyncPostgresSessionDep,
    limit: int = 10,
) -> list[SampleSchema]:
    """Find similar samples using CNN embedding nearest neighbors."""
    sample = await sample_service.get_sample_by_id(db, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    return await sample_service.find_similar_by_cnn(db, sample_id=sample_id, limit=limit)
