from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from samplespace.core.paths import SPECTROGRAMS_DIR
from samplespace.ml.dataset import DURATION_SEC, HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE

logger = logging.getLogger(__name__)

_generation_locks: dict[tuple[str, str], asyncio.Lock] = {}


def _get_spectrogram_dir() -> Path:
    return SPECTROGRAMS_DIR


def _get_cache_path(sample_id: str, mode: str) -> Path:
    return _get_spectrogram_dir() / f"{sample_id}_{mode}.png"


def _render_spectrogram(
    audio_path: Path,
    output_path: Path,
    *,
    mode: Literal["full", "cnn"],
) -> None:
    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

    if mode == "cnn":
        target_length = int(SAMPLE_RATE * DURATION_SEC)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

    S = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(1, 1, figsize=(6, 2.5), dpi=150)
    librosa.display.specshow(
        S_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=6, colors="#888888")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=True,
    )
    plt.close(fig)


async def generate_spectrogram(
    audio_path: Path,
    sample_id: str,
    mode: Literal["full", "cnn"] = "full",
) -> Path:
    cache_path = _get_cache_path(sample_id, mode)

    if cache_path.exists():
        return cache_path

    lock_key = (sample_id, mode)
    lock = _generation_locks.setdefault(lock_key, asyncio.Lock())

    async with lock:
        if cache_path.exists():
            return cache_path

        _get_spectrogram_dir().mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(_render_spectrogram, audio_path, cache_path, mode=mode)
        logger.info(f"Generated {mode} spectrogram for {sample_id}")

    return cache_path
