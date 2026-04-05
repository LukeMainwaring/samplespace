"""Canonical data paths — single source of truth for all data directory references."""

from pathlib import Path

# 4 parents: core/ → samplespace/ → src/ → backend/ (host) or /app/ (Docker)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = _BACKEND_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
UPLOADS_DIR = DATA_DIR / "uploads"
TRANSFORMS_DIR = DATA_DIR / "transforms"
RUNS_DIR = DATA_DIR / "runs"
MODELS_DIR = DATA_DIR / "models"
SPECTROGRAMS_DIR = DATA_DIR / "spectrograms"
