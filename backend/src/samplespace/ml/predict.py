from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from samplespace.ml.dataset import _load_and_preprocess
from samplespace.ml.model import SampleCNN
from samplespace.schemas.sample_type import SAMPLE_TYPES

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "sample_cnn_best.pt"


@dataclass
class PredictionResult:
    embedding: list[float]
    predicted_type: str
    confidence: float
    type_probabilities: dict[str, float]


def load_model(checkpoint_path: str | Path | None = None) -> SampleCNN:
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


def predict_batch(file_paths: list[str], model: SampleCNN) -> list[PredictionResult]:
    if not file_paths:
        return []

    # Load and preprocess all files
    spectrograms = []
    for fp in file_paths:
        mel_spec = _load_and_preprocess(fp)
        spectrograms.append(mel_spec)

    batch = torch.stack(spectrograms)

    with torch.no_grad():
        logits, embeddings = model(batch)

    # Convert batch outputs to individual results
    probabilities = torch.softmax(logits, dim=1)
    results: list[PredictionResult] = []

    for i in range(len(file_paths)):
        probs = probabilities[i]
        predicted_idx = int(probs.argmax().item())
        confidence = probs[predicted_idx].item()
        predicted_type = SAMPLE_TYPES[predicted_idx]

        type_probs = {SAMPLE_TYPES[j]: round(probs[j].item(), 4) for j in range(len(SAMPLE_TYPES))}

        results.append(
            PredictionResult(
                embedding=embeddings[i].tolist(),
                predicted_type=predicted_type,
                confidence=round(confidence, 4),
                type_probabilities=type_probs,
            )
        )

    return results
