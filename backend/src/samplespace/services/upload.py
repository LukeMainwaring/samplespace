"""Upload processing — validate, store, analyze, and embed uploaded audio files."""

import asyncio
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from transformers import ClapModel, ClapProcessor

from samplespace.core.config import get_settings
from samplespace.models.sample import Sample
from samplespace.services import embedding as embedding_service
from samplespace.services import sample as sample_service

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/wave"}
MAX_DURATION_SECONDS = 60


def _validate_wav_header(content: bytes) -> None:
    """Validate that file content starts with a RIFF/WAVE header."""
    if len(content) < 12 or content[:4] != b"RIFF" or content[8:12] != b"WAVE":
        raise HTTPException(status_code=422, detail="File is not a valid WAV file")


async def process_upload(
    db: AsyncSession,
    file: UploadFile,
    clap_model: ClapModel,
    clap_processor: ClapProcessor,
) -> Sample:
    """Process an uploaded WAV file: validate, store, analyze, and embed.

    1. Validates file type, content header, and size
    2. Saves to UPLOAD_DIR with a UUID filename
    3. Creates a Sample record with analyzed metadata (key, BPM, duration, type)
    4. Rejects if duration exceeds limit
    5. Generates CLAP embedding

    Returns the persisted Sample object.
    """
    settings = get_settings()
    max_bytes = settings.UPLOAD_MAX_SIZE_MB * 1024 * 1024

    # Validate content type (when provided)
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=422, detail="Only WAV files are supported")

    # Validate filename extension
    original_filename = file.filename or "upload.wav"
    if not original_filename.lower().endswith(".wav"):
        raise HTTPException(status_code=422, detail="Only WAV files are supported")

    # Read file content and check size
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.UPLOAD_MAX_SIZE_MB}MB",
        )

    # Validate actual WAV file content (RIFF/WAVE header)
    _validate_wav_header(content)

    # Save to temp file, then move to permanent location
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_uuid = str(uuid.uuid4())
        relative_path = f"{file_uuid}.wav"
        permanent_path = upload_dir / relative_path
        shutil.move(str(tmp_path), str(permanent_path))
    finally:
        # Clean up temp file if move failed (move removes source on success)
        tmp_path.unlink(missing_ok=True)

    # Create sample record (runs analyze_and_classify internally for metadata).
    # If this or subsequent steps fail, clean up the permanent file.
    try:
        sample = await sample_service.create_sample(
            db,
            filename=original_filename,
            file_path=str(permanent_path),
            relative_path=relative_path,
            source="upload",
        )

        # Check duration using the already-computed metadata (avoids duplicate librosa load)
        if sample.duration is not None and sample.duration > MAX_DURATION_SECONDS:
            raise HTTPException(
                status_code=422,
                detail=f"File duration ({sample.duration:.1f}s) exceeds maximum of {MAX_DURATION_SECONDS} seconds",
            )

        # Generate CLAP embedding (CPU-bound, run in thread)
        try:
            clap_embedding = await asyncio.to_thread(
                embedding_service.embed_audio,
                str(permanent_path),
                clap_model,
                clap_processor,
            )
            sample.clap_embedding = clap_embedding
            await db.flush()
            logger.info(f"Generated CLAP embedding for upload {sample.id}")
        except Exception:
            logger.exception(f"Failed to generate CLAP embedding for upload {sample.id}")
    except Exception:
        permanent_path.unlink(missing_ok=True)
        raise

    return sample
