"""Batch-generate CNN embeddings for all samples in the database.

Loads the trained SampleCNN, processes each sample's audio file through
the model, and stores the 128-dim embedding in the cnn_embedding column.

Usage:
    uv run embed-cnn
    uv run embed-cnn --force   # re-embed all
"""

import argparse
import asyncio
import logging

from sqlalchemy import select, update

from samplespace.dependencies.db import get_async_sqlalchemy_session
from samplespace.models.sample import Sample
from samplespace.scripts import find_audio_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def generate_embeddings(*, force: bool = False) -> None:
    from samplespace.ml.predict import load_model, predict

    model = load_model()

    async with get_async_sqlalchemy_session() as db:
        stmt = select(Sample)
        if not force:
            stmt = stmt.where(Sample.cnn_embedding.is_(None))

        result = await db.execute(stmt)
        samples = result.scalars().all()
        logger.info(f"Found {len(samples)} samples to embed")

        embedded = 0
        for sample in samples:
            file_path = find_audio_file(sample)
            if file_path is None:
                logger.warning(f"  Audio file not found: {sample.filename}")
                continue

            try:
                pred_result = predict(str(file_path), model)
                await db.execute(
                    update(Sample).where(Sample.id == sample.id).values(cnn_embedding=pred_result.embedding)
                )
                embedded += 1
                logger.info(
                    f"  [{embedded}/{len(samples)}] Embedded: {sample.filename} "
                    f"(predicted: {pred_result.predicted_type}, "
                    f"confidence: {pred_result.confidence:.2%})"
                )
            except Exception:
                logger.warning(f"  Failed to embed: {sample.filename}", exc_info=True)
                continue

            if embedded % 50 == 0:
                await db.commit()

    logger.info(f"Embedded {embedded}/{len(samples)} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CNN embeddings for all samples")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed samples that already have embeddings",
    )
    args = parser.parse_args()

    asyncio.run(generate_embeddings(force=args.force))


if __name__ == "__main__":
    main()
