"""Kit preview service — mix multiple audio files into a single layered preview."""

import contextlib
import hashlib
import logging
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from samplespace.core.paths import TRANSFORMS_DIR

logger = logging.getLogger(__name__)

_SR = 44100


def _preview_cache_dir() -> Path:
    return TRANSFORMS_DIR.resolve() / "kit_previews"


def _cache_path_for_id(preview_id: str) -> Path:
    cache_dir = _preview_cache_dir()
    path = (cache_dir / f"{preview_id}.wav").resolve()
    if not str(path).startswith(str(cache_dir)):
        raise ValueError("Invalid preview ID")
    return path


def get_cached_preview(preview_id: str) -> Path | None:
    path = _cache_path_for_id(preview_id)
    return path if path.exists() else None


def generate_preview_id(audio_paths: list[str]) -> str:
    key = "|".join(sorted(audio_paths))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _to_stereo(y: np.ndarray) -> np.ndarray:
    """Normalize a mono or stereo array to stereo (2, samples)."""
    if y.ndim == 1:
        return np.stack([y, y])
    return y


def _tile_to_length(y: np.ndarray, target_len: int) -> np.ndarray:
    """Tile (loop) a stereo array along the sample axis to fill target length."""
    n_samples = y.shape[1]
    if n_samples >= target_len:
        return y[:, :target_len]
    repeats = (target_len // n_samples) + 1
    return np.tile(y, (1, repeats))[:, :target_len]


def mix_audio(file_paths: list[Path]) -> tuple[str, Path]:
    """Mix multiple audio files into a single layered stereo WAV.

    Shorter samples are looped to fill the duration of the longest sample.
    Returns (preview_id, cache_path).
    """
    path_strings = [str(p) for p in file_paths]
    preview_id = generate_preview_id(path_strings)
    cache_path = _cache_path_for_id(preview_id)

    if cache_path.exists():
        logger.info(f"Kit preview cache hit: {preview_id}")
        return preview_id, cache_path

    arrays: list[np.ndarray] = []
    for p in file_paths:
        y, _ = librosa.load(str(p), sr=_SR, mono=False)
        arrays.append(_to_stereo(y))

    max_len = max(a.shape[1] for a in arrays)
    tiled = [_tile_to_length(a, max_len) for a in arrays]

    mixed: np.ndarray = np.sum(tiled, axis=0) / len(tiled)

    # Normalize to prevent clipping
    peak = float(np.max(np.abs(mixed)))
    if peak > 0:
        mixed = mixed * (0.9 / peak)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".wav")
    try:
        os.close(fd)
        sf.write(tmp_path, mixed.T, _SR, subtype="PCM_24")
        os.rename(tmp_path, cache_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    logger.info(f"Cached kit preview: {preview_id} ({len(file_paths)} samples, {max_len / _SR:.1f}s)")
    return preview_id, cache_path
