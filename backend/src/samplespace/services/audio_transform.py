import contextlib
import logging
import os
import re
import tempfile
from pathlib import Path

import librosa
import soundfile as sf

from samplespace.core.config import get_settings
from samplespace.services import music_theory as music_theory_service

logger = logging.getLogger(__name__)

# STFT resolution for pitch/time operations (higher = better frequency definition)
_N_FFT = 4096

# Only allow safe characters in cache path components
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_for_filename(value: str) -> str:
    result = value.replace(" ", "_").replace("#", "sharp")
    return _SAFE_FILENAME_RE.sub("", result)


def _get_cache_path(sample_id: str, target_key: str | None, target_bpm: int | None) -> Path:
    """Build a deterministic, safe cache path for a transformed audio file.

    Raises ValueError if the resolved path escapes the cache directory.
    """
    settings = get_settings()
    cache_dir = Path(settings.TRANSFORM_CACHE_DIR).resolve()

    parts = [_sanitize_for_filename(sample_id)]
    if target_key:
        parts.append(f"key-{_sanitize_for_filename(target_key)}")
    if target_bpm:
        parts.append(f"bpm-{target_bpm}")
    filename = "_".join(parts) + ".wav"

    path = (cache_dir / filename).resolve()
    if not str(path).startswith(str(cache_dir)):
        raise ValueError("Invalid cache path")
    return path


def get_cached_transform(sample_id: str, target_key: str | None, target_bpm: int | None) -> Path | None:
    path = _get_cache_path(sample_id, target_key, target_bpm)
    return path if path.exists() else None


def transform_sample(
    source_path: Path,
    sample_id: str,
    *,
    source_key: str | None,
    target_key: str | None,
    source_bpm: int | None,
    target_bpm: int | None,
) -> Path:
    """Pitch-shift and/or time-stretch an audio file, returning the cached result.

    This is CPU-bound — call via asyncio.to_thread() from async contexts.

    At least one of (target_key, target_bpm) must differ from the source.
    """
    cache_path = _get_cache_path(sample_id, target_key, target_bpm)
    if cache_path.exists():
        logger.info(f"Cache hit: {cache_path.name}")
        return cache_path

    y, sr = librosa.load(str(source_path), sr=None, mono=False)

    # Pitch shift
    if source_key and target_key:
        n_steps = music_theory_service.semitone_delta(source_key, target_key)
        if n_steps is not None and n_steps != 0:
            logger.info(f"Pitch shifting {sample_id} by {n_steps:+d} semitones")
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps, n_fft=_N_FFT)

    # Time stretch
    if source_bpm and target_bpm:
        rate = target_bpm / source_bpm
        if abs(rate - 1.0) > 0.01:
            logger.info(f"Time stretching {sample_id} by rate {rate:.3f}")
            y = librosa.effects.time_stretch(y, rate=rate, n_fft=_N_FFT)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file then rename to avoid partial reads
    # from concurrent requests for the same transformation.
    fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".wav")
    try:
        os.close(fd)
        sf.write(tmp_path, y.T if y.ndim == 2 else y, sr, subtype="PCM_24")
        os.rename(tmp_path, cache_path)
    except BaseException:
        # Clean up temp file on any failure
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    logger.info(f"Cached transform: {cache_path.name}")
    return cache_path
