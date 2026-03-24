"""Audio transformation service — pitch shifting and time stretching with caching."""

import logging
from pathlib import Path

import librosa
import soundfile as sf

from samplespace.core.config import get_settings
from samplespace.services import music_theory as music_theory_service

logger = logging.getLogger(__name__)

# librosa sample rate for loading audio
_SR = 22050


def _sanitize_for_filename(key: str) -> str:
    """Sanitize a key string for use in filenames."""
    return key.replace(" ", "_").replace("#", "sharp")


def _get_cache_path(sample_id: str, target_key: str | None, target_bpm: int | None) -> Path:
    """Build a deterministic cache path for a transformed audio file."""
    settings = get_settings()
    parts = [sample_id]
    if target_key:
        parts.append(f"key-{_sanitize_for_filename(target_key)}")
    if target_bpm:
        parts.append(f"bpm-{target_bpm}")
    filename = "_".join(parts) + ".wav"
    return Path(settings.TRANSFORM_CACHE_DIR) / filename


def get_cached_transform(sample_id: str, target_key: str | None, target_bpm: int | None) -> Path | None:
    """Return the cached transform path if it exists, otherwise None."""
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

    y, sr = librosa.load(str(source_path), sr=_SR, mono=True)

    # Pitch shift
    if source_key and target_key:
        n_steps = music_theory_service.semitone_delta(source_key, target_key)
        if n_steps is not None and n_steps != 0:
            logger.info(f"Pitch shifting {sample_id} by {n_steps:+d} semitones")
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # Time stretch
    if source_bpm and target_bpm:
        rate = target_bpm / source_bpm
        if abs(rate - 1.0) > 0.01:
            logger.info(f"Time stretching {sample_id} by rate {rate:.3f}")
            y = librosa.effects.time_stretch(y, rate=rate)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(cache_path), y, sr)
    logger.info(f"Cached transform: {cache_path.name}")
    return cache_path
