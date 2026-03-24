import mimetypes

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter

from samplespace.dependencies.clap import ClapModelsDep
from samplespace.dependencies.db import AsyncPostgresSessionDep
from samplespace.models.sample import AudioFileNotFound, SampleNotFound
from samplespace.schemas.sample import SampleListResponse, SampleSchema, SampleSearchRequest
from samplespace.services import audio_transform as audio_transform_service
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
        is_loop=body.is_loop,
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
        raise SampleNotFound()

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
        raise SampleNotFound()

    return await sample_service.find_similar_by_cnn(db, sample_id=sample_id, limit=limit)


@samples_router.get("/{sample_id}/audio/transformed")
async def get_transformed_audio(
    sample_id: str,
    key: str | None = None,
    bpm: int | None = None,
) -> FileResponse:
    """Serve a cached transformed audio file.

    The agent tool pre-warms the cache via match_to_context. This endpoint
    is a pure file server — no on-demand transformation.
    """
    if key is None and bpm is None:
        raise HTTPException(status_code=400, detail="At least one of key or bpm is required")

    cached = audio_transform_service.get_cached_transform(sample_id, key, bpm)
    if cached is None:
        raise HTTPException(status_code=404, detail="Transformed audio not found in cache")

    return FileResponse(
        path=str(cached),
        media_type="audio/wav",
        filename=cached.name,
    )


@samples_router.get("/{sample_id}/audio")
async def get_sample_audio(
    sample_id: str,
    db: AsyncPostgresSessionDep,
) -> FileResponse:
    """Stream the audio file for a sample."""
    sample = await sample_service.get_sample_by_id(db, sample_id)
    if sample is None:
        raise SampleNotFound()

    file_path = sample_service.find_audio_file(sample)
    if file_path is None:
        raise AudioFileNotFound()

    content_type = mimetypes.guess_type(sample.filename)[0] or "audio/wav"

    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        filename=sample.filename,
    )
