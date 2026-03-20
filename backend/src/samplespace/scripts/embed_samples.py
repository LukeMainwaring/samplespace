"""Batch-generate CLAP embeddings for all samples in the database.

Loads the CLAP model, reads each sample's audio file, generates a 512-dim
embedding, and stores it in the clap_embedding column.

Usage:
    uv run embed-samples
    uv run embed-samples --force   # re-embed samples that already have embeddings
"""

import argparse
import asyncio
import logging

from sqlalchemy import select, update

from samplespace.dependencies.db import get_async_sqlalchemy_session
from samplespace.models.sample import Sample
from samplespace.scripts import find_audio_file
from samplespace.services.embedding import embed_audio, load_clap_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def generate_embeddings(*, force: bool = False) -> None:
    # Load CLAP model
    model, processor = load_clap_model()

    async with get_async_sqlalchemy_session() as db:
        # Query samples that need embedding
        stmt = select(Sample)
        if not force:
            stmt = stmt.where(Sample.clap_embedding.is_(None))

        result = await db.execute(stmt)
        samples = result.scalars().all()
        logger.info(f"Found {len(samples)} samples to embed")

        embedded = 0
        for sample in samples:
            file_path = find_audio_file(sample.filename, sample.sample_type)
            if file_path is None:
                logger.warning(f"  Audio file not found: {sample.filename}")
                continue

            try:
                embedding = embed_audio(str(file_path), model, processor)
                await db.execute(update(Sample).where(Sample.id == sample.id).values(clap_embedding=embedding))
                embedded += 1
                logger.info(f"  Embedded: {sample.filename}")
            except Exception:
                logger.warning(f"  Failed to embed: {sample.filename}", exc_info=True)
                continue

    logger.info(f"Embedded {embedded}/{len(samples)} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CLAP embeddings for all samples")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed samples that already have embeddings",
    )
    args = parser.parse_args()

    asyncio.run(generate_embeddings(force=args.force))


if __name__ == "__main__":
    main()
