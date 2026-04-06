import contextlib
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

from samplespace.core.paths import TRANSFORMS_DIR
from samplespace.services import music_theory as music_theory_service

logger = logging.getLogger(__name__)

# Only allow safe characters in cache path components
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_for_filename(value: str) -> str:
    result = value.replace(" ", "_").replace("#", "sharp")
    return _SAFE_FILENAME_RE.sub("", result)


def _get_cache_path(sample_id: str, target_key: str | None, target_bpm: int | None) -> Path:
    """Build a deterministic, safe cache path for a transformed audio file.

    Raises ValueError if the resolved path escapes the cache directory.
    """
    cache_dir = TRANSFORMS_DIR.resolve()

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
    """Pitch-shift and/or time-stretch an audio file using Rubber Band R3.

    Feeds the source file directly to the rubberband CLI — no intermediate
    numpy loading or processing. R3 (--fine) is the highest-quality engine
    and handles transient detection, phase coherence, and windowing internally.

    No per-sample normalization is applied; original level relationships are
    preserved so that kit mixes maintain natural dynamics.

    This is CPU-bound — call via asyncio.to_thread() from async contexts.
    """
    cache_path = _get_cache_path(sample_id, target_key, target_bpm)
    if cache_path.exists():
        logger.info(f"Cache hit: {cache_path.name}")
        return cache_path

    # Build rubberband command
    cmd: list[str] = ["rubberband", "--fine"]

    if source_key and target_key:
        n_steps = music_theory_service.semitone_delta(source_key, target_key)
        if n_steps is not None and n_steps != 0:
            logger.info(f"Pitch shifting {sample_id} by {n_steps:+d} semitones")
            cmd += ["-p", str(n_steps)]

    if source_bpm and target_bpm:
        rate = target_bpm / source_bpm
        if abs(rate - 1.0) > 0.01:
            logger.info(f"Time stretching {sample_id}: {source_bpm} → {target_bpm} BPM (rate {rate:.3f})")
            # -T is tempo ratio (speed multiplier): -T 1.3 = 1.3x faster
            cmd += ["-T", str(rate)]

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: rubberband writes to temp file, then rename
    fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".wav")
    try:
        os.close(fd)
        cmd += [str(source_path), tmp_path]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"rubberband failed: {result.stderr.decode(errors='replace')}")
        os.rename(tmp_path, cache_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    logger.info(f"Cached transform: {cache_path.name}")
    return cache_path


def resolve_transform(
    audio_path: Path,
    sample_id: str,
    *,
    sample_key: str | None,
    sample_bpm: int | None,
    target_key: str | None,
    target_bpm: int | None,
) -> tuple[Path, bool]:
    """Resolve the best available audio for a sample given a target key/BPM.

    Returns (path, was_transformed). Uses a cached transform if available,
    otherwise transforms on the spot. Returns the original path unchanged
    if no transform is needed.

    This is CPU-bound — call via asyncio.to_thread() from async contexts.
    """
    actual_target_key: str | None = None
    if target_key and sample_key:
        actual_target_key = music_theory_service.compute_target_key(sample_key, target_key)
        if actual_target_key == sample_key:
            actual_target_key = None

    effective_bpm = target_bpm if target_bpm and sample_bpm and sample_bpm != target_bpm else None

    if actual_target_key is None and effective_bpm is None:
        return audio_path, False

    cached = get_cached_transform(sample_id, actual_target_key, effective_bpm)
    if cached:
        return cached, True

    result = transform_sample(
        audio_path,
        sample_id,
        source_key=sample_key if actual_target_key else None,
        target_key=actual_target_key,
        source_bpm=sample_bpm if effective_bpm else None,
        target_bpm=effective_bpm,
    )
    return result, True
