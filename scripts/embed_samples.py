"""Batch-generate CLAP embeddings for all samples in the database.

Loads the CLAP model, reads each sample's audio file, generates a 512-dim
embedding, and stores it in the clap_embedding column.

Usage:
    python scripts/embed_samples.py
    python scripts/embed_samples.py --force   # re-embed samples that already have embeddings

NOTE: Requires the CLAP model (~600MB, cached by HuggingFace in ~/.cache/huggingface/).
      This will be implemented in Phase 2.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add backend/src to path so we can import samplespace
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CLAP embeddings for all samples")
    parser.add_argument("--force", action="store_true", help="Re-embed samples that already have embeddings")
    args = parser.parse_args()

    # Phase 2: Import CLAP model and embedding service
    # from samplespace.services.embedding import embed_audio, load_clap_model

    logger.error(
        "CLAP embedding generation is not yet implemented. "
        "This will be available after Phase 2 (CLAP Embeddings + Semantic Search)."
    )
    sys.exit(1)

    # Phase 2 implementation will:
    # 1. Load CLAP model
    # 2. Query DB for samples missing clap_embedding (or all if --force)
    # 3. For each sample, load audio from SAMPLES_DIR / filename
    # 4. Generate 512-dim CLAP embedding
    # 5. Update the sample record with the embedding
    _ = args  # Will be used in Phase 2


if __name__ == "__main__":
    main()
