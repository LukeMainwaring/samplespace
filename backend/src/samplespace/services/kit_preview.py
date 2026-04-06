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
# Crossfade duration at loop/tile boundaries to prevent clicks (in samples)
_CROSSFADE_SAMPLES = int(0.01 * _SR)  # 10ms


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
    """Tile (loop) a stereo array to fill target length with crossfade at seams.

    A short cosine crossfade at each loop boundary prevents clicks from
    discontinuities where the waveform doesn't end at a zero crossing.
    """
    n_samples = y.shape[1]
    if n_samples >= target_len:
        return y[:, :target_len]

    cf = min(_CROSSFADE_SAMPLES, n_samples // 4)
    result = np.zeros((y.shape[0], target_len), dtype=y.dtype)
    pos = 0

    while pos < target_len:
        remaining = target_len - pos
        chunk_len = min(n_samples, remaining)
        chunk = y[:, :chunk_len]

        if pos > 0 and cf > 0 and chunk_len > cf:
            # Crossfade: blend the end of the previous iteration with the
            # start of this one using a cosine curve for a smooth transition
            fade = np.cos(np.linspace(0, np.pi / 2, cf)) ** 2
            result[:, pos : pos + cf] *= fade
            result[:, pos : pos + cf] += chunk[:, :cf] * (1 - fade)
            result[:, pos + cf : pos + chunk_len] = chunk[:, cf:]
        else:
            result[:, pos : pos + chunk_len] = chunk

        pos += chunk_len

    return result


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

    # Sum all layers then normalize the mix as a whole.
    # Using summation (not averaging) preserves transient punch;
    # the peak normalization handles final level.
    mixed: np.ndarray = np.sum(tiled, axis=0)

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
