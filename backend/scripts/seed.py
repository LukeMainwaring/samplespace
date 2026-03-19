"""Seed the database with audio samples from data/samples/.

Scans data/samples/ for audio files, analyzes metadata (key, BPM, duration),
and inserts into the database. Organize files in subdirectories (e.g., kick/,
snare/, pad/) to auto-infer sample_type.

Usage:
    python scripts/seed.py                        # scan data/samples/
    python scripts/seed.py --sample-type kick      # override sample_type for all files
"""

import argparse
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from samplespace.core.config import get_settings
from samplespace.models import Base, Sample
from samplespace.services.audio_analysis import analyze_audio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).parent.parent.parent / "data" / "samples"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff"}


def scan_local_samples(
    sample_type_override: str | None = None,
) -> list[tuple[Path, str | None]]:
    """Scan data/samples/ for local audio files.

    Returns list of (file_path, sample_type) tuples.
    Infers sample_type from parent directory name if files are organized in subdirectories.
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    found: list[tuple[Path, str | None]] = []

    for path in sorted(SAMPLES_DIR.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if path.name.startswith("."):
            continue

        # Infer type from subdirectory name, or use override
        if sample_type_override:
            sample_type = sample_type_override
        elif path.parent != SAMPLES_DIR:
            sample_type = path.parent.name
        else:
            sample_type = None

        found.append((path, sample_type))

    logger.info(f"Found {len(found)} audio files in {SAMPLES_DIR}")
    return found


def seed_database(samples: list[tuple[Path, str | None]]) -> int:
    """Analyze audio files and insert sample records into the database."""
    config = get_settings()
    sync_url = f"postgresql+psycopg://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"

    engine = create_engine(sync_url)

    # Ensure pgvector extension exists
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(engine)

    inserted = 0
    with Session(engine) as session:
        # Get existing filenames to avoid duplicates
        existing = {row[0] for row in session.execute(text("SELECT filename FROM samples")).fetchall()}

        for file_path, sample_type in samples:
            filename = file_path.name
            if filename in existing:
                logger.info(f"  Skipping duplicate: {filename}")
                continue

            try:
                metadata = analyze_audio(str(file_path))
            except Exception:
                logger.warning(f"  Failed to analyze: {filename}")
                continue

            sample = Sample(
                id=str(uuid.uuid4()),
                filename=filename,
                key=metadata["key"],
                bpm=metadata["bpm"],
                duration=metadata["duration"],
                sample_type=sample_type,
            )
            session.add(sample)
            inserted += 1
            logger.info(
                f"  Inserted: {filename} "
                f"(key={metadata['key']}, bpm={metadata['bpm']}, "
                f"duration={metadata['duration']:.1f}s, type={sample_type})"
            )

        session.commit()

    logger.info(f"Seeded {inserted} new samples (skipped {len(samples) - inserted} duplicates)")
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the SampleSpace database with audio samples")
    parser.add_argument(
        "--sample-type",
        type=str,
        default=None,
        help="Override sample_type for all files",
    )
    args = parser.parse_args()

    samples = scan_local_samples(args.sample_type)

    if not samples:
        logger.warning(
            "No samples found. Place audio files in data/samples/ "
            "(organize in subdirectories like kick/, snare/, pad/ to auto-infer type)."
        )
        sys.exit(0)

    seed_database(samples)


if __name__ == "__main__":
    main()
