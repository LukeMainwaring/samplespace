import asyncio
import mimetypes
from pathlib import Path
from typing import Annotated, Literal

from fastapi import HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter

from samplespace.dependencies.clap import ClapModelsDep
from samplespace.dependencies.db import AsyncPostgresSessionDep
from samplespace.models.sample import AudioFileNotFound, Sample, SampleNotFound
from samplespace.schemas.sample import (
    ListSamplesParams,
    SampleListResponse,
    SampleSchema,
    SampleSearchRequest,
    SampleUpdateSchema,
    SimilarSampleSchema,
)
from samplespace.services import audio_transform as audio_transform_service
from samplespace.services import embedding as embedding_service
from samplespace.services import kit_preview as kit_preview_service
from samplespace.services import sample as sample_service
from samplespace.services import spectrogram as spectrogram_service
from samplespace.services import upload as upload_service

samples_router = APIRouter(
    prefix="/samples",
    tags=["samples"],
)


@samples_router.get("/kit-preview/{preview_id}")
async def get_kit_preview(preview_id: str) -> FileResponse:
    """Serve a cached kit preview mixdown."""
    cached = kit_preview_service.get_cached_preview(preview_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="Kit preview not found in cache")

    return FileResponse(
        path=str(cached),
        media_type="audio/wav",
        filename=cached.name,
    )


@samples_router.get("/pair-preview/{sample_a_id}/{sample_b_id}")
async def get_pair_preview(
    sample_a_id: str,
    sample_b_id: str,
    db: AsyncPostgresSessionDep,
    key: str | None = None,
    bpm: int | None = None,
) -> FileResponse:
    """Mix two samples together for audition.

    When key/bpm are provided, loops are pitch-shifted and time-stretched
    to the target before mixing (aligns samples to song context).
    """
    sample_a = await sample_service.get_sample_by_id(db, sample_a_id)
    sample_b = await sample_service.get_sample_by_id(db, sample_b_id)

    if sample_a is None or sample_b is None:
        raise HTTPException(status_code=404, detail="One or both samples not found")

    paths: list[Path] = []
    for sample in [sample_a, sample_b]:
        audio_path = sample_service.find_audio_file(sample)
        if audio_path is None:
            raise HTTPException(status_code=404, detail=f"Audio file not found for {sample.filename}")

        # Transform loops to target key/bpm when song context is provided
        if (key or bpm) and sample.is_loop:
            transformed = await _transform_for_preview(audio_path, sample, key, bpm)
            paths.append(transformed)
        else:
            paths.append(audio_path)

    preview_id, cache_path = await asyncio.to_thread(kit_preview_service.mix_audio, paths)

    return FileResponse(
        path=str(cache_path),
        media_type="audio/wav",
        filename=f"pair_preview_{preview_id}.wav",
    )


async def _transform_for_preview(
    audio_path: Path,
    sample: Sample,
    target_key: str | None,
    target_bpm: int | None,
) -> Path:
    """Transform a sample to the target key/bpm, returning the (cached) file path."""
    resolved, _ = await asyncio.to_thread(
        audio_transform_service.resolve_transform,
        audio_path,
        sample.id,
        sample_key=sample.key,
        sample_bpm=sample.bpm,
        target_key=target_key,
        target_bpm=target_bpm,
    )
    return resolved


@samples_router.get("/")
async def list_samples(
    db: AsyncPostgresSessionDep,
    params: Annotated[ListSamplesParams, Query()],
) -> SampleListResponse:
    """List all samples with pagination and optional filters."""
    return await sample_service.get_samples(db, params=params)


@samples_router.patch("/{sample_id}")
async def update_sample(
    sample_id: str,
    body: SampleUpdateSchema,
    db: AsyncPostgresSessionDep,
) -> SampleSchema:
    """Update metadata for an uploaded sample."""
    return await sample_service.update_sample(db, sample_id, body)


@samples_router.delete("/{sample_id}", status_code=204)
async def delete_sample(
    sample_id: str,
    db: AsyncPostgresSessionDep,
) -> None:
    """Delete an uploaded sample and its associated data."""
    await sample_service.delete_sample(db, sample_id)


@samples_router.post("/upload")
async def upload_sample(
    file: UploadFile,
    db: AsyncPostgresSessionDep,
    clap: ClapModelsDep,
) -> SampleSchema:
    """Upload a WAV file, analyze it, and generate CLAP embeddings."""
    sample = await upload_service.process_upload(db, file, clap.model, clap.processor)
    return SampleSchema.model_validate(sample)


@samples_router.post("/search")
async def search_samples(
    body: SampleSearchRequest,
    db: AsyncPostgresSessionDep,
    clap: ClapModelsDep,
) -> list[SampleSchema]:
    """Search samples by natural language query with optional metadata filters."""
    if not body.query:
        raise HTTPException(status_code=400, detail="Query is required for search")

    query_embedding = await asyncio.to_thread(embedding_service.embed_text, body.query, clap.model, clap.processor)

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
) -> list[SimilarSampleSchema]:
    """Find similar samples using CNN embedding nearest neighbors."""
    sample = await sample_service.get_sample_by_id(db, sample_id)
    if sample is None:
        raise SampleNotFound()

    return await sample_service.find_similar_by_cnn(db, sample_id=sample_id, limit=limit)


@samples_router.get(
    "/{sample_id}/spectrogram",
    response_class=FileResponse,
    responses={200: {"content": {"image/png": {}}}},
)
async def get_sample_spectrogram(
    sample_id: str,
    db: AsyncPostgresSessionDep,
    mode: Literal["full", "cnn"] = "full",
) -> FileResponse:
    """Generate and serve a mel spectrogram PNG for a sample."""
    sample = await sample_service.get_sample_by_id(db, sample_id)
    if sample is None:
        raise SampleNotFound()

    file_path = sample_service.find_audio_file(sample)
    if file_path is None:
        raise AudioFileNotFound()

    spectrogram_path = await spectrogram_service.generate_spectrogram(file_path, sample_id, mode=mode)

    return FileResponse(
        path=str(spectrogram_path),
        media_type="image/png",
        filename=f"{sample_id}_{mode}_spectrogram.png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


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
