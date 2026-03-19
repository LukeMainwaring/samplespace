"""Inference wrapper for the trained SampleCNN model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from samplespace.ml.dataset import SAMPLE_TYPES, _load_and_preprocess
from samplespace.ml.model import SampleCNN

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "sample_cnn_best.pt"


@dataclass
class PredictionResult:
    """Result from CNN inference."""

    embedding: list[float]
    predicted_type: str
    confidence: float
    type_probabilities: dict[str, float]


def load_model(checkpoint_path: str | Path | None = None) -> SampleCNN:
    """Load a trained SampleCNN from a checkpoint."""
    path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT

    if not path.exists():
        msg = f"Checkpoint not found: {path}. Run `python -m samplespace.ml.train` first."
        raise FileNotFoundError(msg)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SampleCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Loaded SampleCNN from {path}")
    return model


def predict(file_path: str, model: SampleCNN) -> PredictionResult:
    """Run inference on a single audio file.

    Returns the 128-dim embedding, predicted type, and confidence.
    """
    mel_spec = _load_and_preprocess(file_path)
    mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        logits, embedding = model(mel_spec)

    # Classification
    probabilities = torch.softmax(logits[0], dim=0)
    predicted_idx = int(probabilities.argmax().item())
    confidence = probabilities[predicted_idx].item()
    predicted_type = SAMPLE_TYPES[predicted_idx]

    type_probs = {SAMPLE_TYPES[i]: round(probabilities[i].item(), 4) for i in range(len(SAMPLE_TYPES))}

    return PredictionResult(
        embedding=embedding[0].tolist(),
        predicted_type=predicted_type,
        confidence=round(confidence, 4),
        type_probabilities=type_probs,
    )
