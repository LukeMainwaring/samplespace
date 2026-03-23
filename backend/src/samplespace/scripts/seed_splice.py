"""Seed the database with audio samples from a Splice library.

Scans the configured SPLICE_DIR for audio files, infers sample_type from
directory structure, analyzes metadata (key, BPM, duration), and inserts
into the database with source="splice".

Usage:
    uv run seed-splice                  # ingest up to 100 samples (default)
    uv run seed-splice --limit 500      # ingest up to 500 samples
    uv run seed-splice --limit 0        # ingest all samples (no limit)
"""

import argparse
import asyncio
import logging
import uuid
from pathlib import Path

from sqlalchemy import select

from samplespace.core.config import get_settings
from samplespace.dependencies.db import get_async_sqlalchemy_session
from samplespace.models.sample import Sample
from samplespace.services.audio_analysis import analyze_and_classify
from samplespace.services.path_inference import extract_pack_name, infer_sample_type_from_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff"}

BATCH_SIZE = 50


def scan_splice_dir(splice_dir: Path) -> list[tuple[Path, str]]:
    """Scan Splice directory for audio files.

    Returns list of (absolute_path, relative_path) tuples.
    """
    found: list[tuple[Path, str]] = []
    for path in sorted(splice_dir.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if path.name.startswith("."):
            continue
        relative = str(path.relative_to(splice_dir))
        found.append((path, relative))

    logger.info(f"Found {len(found)} audio files in {splice_dir}")
    return found


async def seed_splice(samples: list[tuple[Path, str]], *, limit: int) -> int:
    """Analyze Splice audio files and insert sample records into the database."""
    inserted = 0
    batch_count = 0
    async with get_async_sqlalchemy_session() as db:
        # Get existing relative_paths for splice source to skip duplicates
        result = await db.execute(select(Sample.relative_path).where(Sample.source == "splice"))
        existing = {row[0] for row in result.all()}

        for abs_path, relative_path in samples:
            if limit > 0 and inserted >= limit:
                logger.info(f"Reached limit of {limit} samples")
                break

            if relative_path in existing:
                continue

            try:
                analysis = analyze_and_classify(str(abs_path))
            except Exception:
                logger.warning(f"  Failed to analyze: {relative_path}")
                continue

            sample_type = infer_sample_type_from_path(relative_path)
            pack_name = extract_pack_name(relative_path)

            sample = Sample(
                id=str(uuid.uuid4()),
                filename=abs_path.name,
                relative_path=relative_path,
                source="splice",
                pack_name=pack_name,
                key=analysis.metadata.key,
                bpm=analysis.metadata.bpm,
                duration=analysis.metadata.duration,
                sample_type=sample_type,
                is_loop=analysis.is_loop,
            )
            db.add(sample)
            inserted += 1
            batch_count += 1
            logger.info(
                f"  [{inserted}] {abs_path.name} "
                f"(type={sample_type}, pack={pack_name}, "
                f"bpm={analysis.metadata.bpm}, key={analysis.metadata.key}, "
                f"is_loop={analysis.is_loop})"
            )

            if batch_count >= BATCH_SIZE:
                await db.commit()
                batch_count = 0
                logger.info(f"  Committed batch ({inserted} total so far)")

    logger.info(f"Seeded {inserted} new Splice samples (skipped {len(existing)} existing)")
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the database with Splice samples")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of new samples to ingest (0 for unlimited, default: 100)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.SPLICE_DIR:
        logger.error("SPLICE_DIR is not configured. Set it in .env or as an environment variable.")
        return

    splice_dir = Path(settings.SPLICE_DIR)
    if not splice_dir.is_dir():
        logger.error(f"SPLICE_DIR does not exist: {splice_dir}")
        return

    samples = scan_splice_dir(splice_dir)
    if not samples:
        logger.warning(f"No audio files found in {splice_dir}")
        return

    asyncio.run(seed_splice(samples, limit=args.limit))


if __name__ == "__main__":
    main()
