"""Seed the database with audio samples from a local sample library.

Scans the configured SAMPLE_LIBRARY_DIR for audio files, infers sample_type from
directory structure, analyzes metadata (key, BPM, duration), and inserts
into the database with source="library".

Usage:
    uv run seed-samples                  # ingest up to 100 samples (default)
    uv run seed-samples --limit 500      # ingest up to 500 samples
    uv run seed-samples --limit 0        # ingest all samples (no limit)
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


def scan_library_dir(library_dir: Path) -> list[tuple[Path, str]]:
    """Scan sample library directory for audio files.

    Returns list of (absolute_path, relative_path) tuples.
    """
    found: list[tuple[Path, str]] = []
    for path in sorted(library_dir.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if path.name.startswith("."):
            continue
        relative = str(path.relative_to(library_dir))
        found.append((path, relative))

    logger.info(f"Found {len(found)} audio files in {library_dir}")
    return found


async def seed_samples(samples: list[tuple[Path, str]], *, limit: int) -> int:
    """Analyze audio files and insert sample records into the database."""
    inserted = 0
    batch_count = 0
    async with get_async_sqlalchemy_session() as db:
        result = await db.execute(select(Sample.relative_path).where(Sample.source == "library"))
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
                source="library",
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
            loop_label = "loop" if analysis.is_loop else "one_shot"
            logger.info(
                f"  [{inserted}] {abs_path.name}\n"
                f"    sample_type: {sample_type}\n"
                f"    bpm: {analysis.metadata.bpm}\n"
                f"    key: {analysis.metadata.key}\n"
                f"    category: {loop_label}\n"
                f"    -------------------------"
            )

            if batch_count >= BATCH_SIZE:
                await db.commit()
                batch_count = 0
                logger.info(f"  Committed batch ({inserted} total so far)")

    logger.info(f"Seeded {inserted} new samples (skipped {len(existing)} existing)")
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the database from a local sample library")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of new samples to ingest (0 for unlimited, default: 100)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.SAMPLE_LIBRARY_DIR:
        logger.error("SAMPLE_LIBRARY_DIR is not configured. Set it in .env or as an environment variable.")
        return

    library_dir = Path(settings.SAMPLE_LIBRARY_DIR)
    if not library_dir.is_dir():
        logger.error(f"SAMPLE_LIBRARY_DIR does not exist: {library_dir}")
        return

    samples = scan_library_dir(library_dir)
    if not samples:
        logger.warning(f"No audio files found in {library_dir}")
        return

    asyncio.run(seed_samples(samples, limit=args.limit))


if __name__ == "__main__":
    main()
