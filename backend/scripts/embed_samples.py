"""Batch-generate CLAP embeddings for all samples in the database.

Loads the CLAP model, reads each sample's audio file, generates a 512-dim
embedding, and stores it in the clap_embedding column.

Usage:
    python scripts/embed_samples.py
    python scripts/embed_samples.py --force   # re-embed samples that already have embeddings
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session

from samplespace.core.config import get_settings
from samplespace.models.sample import Sample
from samplespace.services.embedding import embed_audio, load_clap_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).parent.parent.parent / "data" / "samples"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CLAP embeddings for all samples")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed samples that already have embeddings",
    )
    args = parser.parse_args()

    config = get_settings()
    sync_url = (
        f"postgresql+psycopg://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}"
        f"@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
    )
    engine = create_engine(sync_url)

    # Load CLAP model
    model, processor = load_clap_model()

    with Session(engine) as session:
        # Query samples that need embedding
        stmt = select(Sample)
        if not args.force:
            stmt = stmt.where(Sample.clap_embedding.is_(None))

        samples = session.execute(stmt).scalars().all()
        logger.info(f"Found {len(samples)} samples to embed")

        embedded = 0
        for sample in samples:
            # Find audio file — check category subdirectories
            file_path = None
            if sample.sample_type:
                candidate = SAMPLES_DIR / sample.sample_type / sample.filename
                if candidate.exists():
                    file_path = candidate

            if file_path is None:
                candidate = SAMPLES_DIR / sample.filename
                if candidate.exists():
                    file_path = candidate

            if file_path is None:
                # Search all subdirectories
                matches = list(SAMPLES_DIR.rglob(sample.filename))
                if matches:
                    file_path = matches[0]

            if file_path is None:
                logger.warning(f"  Audio file not found: {sample.filename}")
                continue

            try:
                embedding = embed_audio(str(file_path), model, processor)
                session.execute(update(Sample).where(Sample.id == sample.id).values(clap_embedding=embedding))
                embedded += 1
                logger.info(f"  Embedded: {sample.filename}")
            except Exception:
                logger.warning(f"  Failed to embed: {sample.filename}", exc_info=True)
                continue

        session.commit()

    logger.info(f"Embedded {embedded}/{len(samples)} samples")


if __name__ == "__main__":
    main()
