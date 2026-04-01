from __future__ import annotations

import logging
from typing import Any

import librosa
import torch
from transformers import ClapModel, ClapProcessor

logger = logging.getLogger(__name__)

CLAP_MODEL_NAME = "laion/clap-htsat-unfused"
CLAP_EMBEDDING_DIM = 512
SAMPLE_RATE = 48000  # CLAP expects 48kHz audio


def load_clap_model() -> tuple[ClapModel, ClapProcessor]:
    """~600MB, cached by HuggingFace. Called once at startup via lifespan."""
    logger.info(f"Loading CLAP model: {CLAP_MODEL_NAME}...")
    model: ClapModel = ClapModel.from_pretrained(CLAP_MODEL_NAME)
    processor: ClapProcessor = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)
    model.eval()  # type: ignore[no-untyped-call]
    logger.info("CLAP model loaded successfully")
    return model, processor


def embed_audio(
    file_path: str,
    model: ClapModel,
    processor: ClapProcessor,
) -> list[float]:
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    inputs: dict[str, Any] = processor(audio=y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    audio_keys = {k: v for k, v in inputs.items() if k.startswith("input_features")}

    with torch.no_grad():
        audio_output = model.audio_model(**audio_keys)
        projected: torch.Tensor = model.audio_projection(audio_output.pooler_output)

    embedding = projected[0]
    embedding = embedding / embedding.norm()

    result: list[float] = embedding.tolist()
    return result


def embed_text(
    query: str,
    model: ClapModel,
    processor: ClapProcessor,
) -> list[float]:
    inputs: dict[str, Any] = processor(text=query, return_tensors="pt", padding=True)
    text_keys = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

    with torch.no_grad():
        text_output = model.text_model(**text_keys)
        projected: torch.Tensor = model.text_projection(text_output.pooler_output)

    embedding = projected[0]
    embedding = embedding / embedding.norm()

    result: list[float] = embedding.tolist()
    return result
