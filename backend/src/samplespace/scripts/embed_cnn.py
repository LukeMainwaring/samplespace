"""Batch-generate CNN embeddings for all samples in the database.

Loads the trained SampleCNN, processes each sample's audio file through
the model, and stores the 128-dim embedding in the cnn_embedding column.

Usage:
    uv run embed-cnn
    uv run embed-cnn --force   # re-embed all
"""

import argparse
import logging
from pathlib import Path

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session

from samplespace.core.config import get_settings
from samplespace.ml.predict import load_model, predict
from samplespace.models.sample import Sample

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CNN embeddings for all samples")
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
    SAMPLES_DIR = Path(config.SAMPLES_DIR)

    model = load_model()

    with Session(engine) as session:
        stmt = select(Sample)
        if not args.force:
            stmt = stmt.where(Sample.cnn_embedding.is_(None))

        samples = session.execute(stmt).scalars().all()
        logger.info(f"Found {len(samples)} samples to embed")

        embedded = 0
        for sample in samples:
            # Find audio file
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
                matches = list(SAMPLES_DIR.rglob(sample.filename))
                if matches:
                    file_path = matches[0]

            if file_path is None:
                logger.warning(f"  Audio file not found: {sample.filename}")
                continue

            try:
                result = predict(str(file_path), model)
                session.execute(update(Sample).where(Sample.id == sample.id).values(cnn_embedding=result.embedding))
                embedded += 1
                logger.info(
                    f"  Embedded: {sample.filename} "
                    f"(predicted: {result.predicted_type}, "
                    f"confidence: {result.confidence:.2%})"
                )
            except Exception:
                logger.warning(f"  Failed to embed: {sample.filename}", exc_info=True)
                continue

        session.commit()

    logger.info(f"Embedded {embedded}/{len(samples)} samples")


if __name__ == "__main__":
    main()
