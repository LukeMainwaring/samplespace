import contextlib
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt  # type: ignore[import-untyped]

from samplespace.core.paths import TRANSFORMS_DIR
from samplespace.services import music_theory as music_theory_service

logger = logging.getLogger(__name__)

# Only allow safe characters in cache path components
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_-]")

# Content-type sets for adaptive Rubber Band --crisp level
_TRANSIENT_TYPES = {"kick", "snare", "hihat", "clap", "cymbal", "percussion", "drum", "fx"}
_SUSTAINED_TYPES = {"pad", "strings", "vocal", "synth", "bass", "keys", "guitar", "horn"}


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


def _crisp_level(sample_type: str | None) -> int:
    """Select Rubber Band transient detector sensitivity by content type.

    Level 6 = maximum transient preservation (drums/percussive).
    Level 0 = smooth steady-state (pads/sustained).
    Level 3 = balanced default.
    """
    if sample_type and sample_type.lower() in _TRANSIENT_TYPES:
        return 6
    if sample_type and sample_type.lower() in _SUSTAINED_TYPES:
        return 0
    return 3


def _highpass(y: np.ndarray, sr: int | float, cutoff: float = 20.0) -> np.ndarray:
    """Apply a gentle high-pass filter to remove subsonic content.

    Subsonic energy gets amplified by pitch shifting (especially downward)
    and wastes headroom.
    """
    sos = butter(2, cutoff, btype="high", fs=sr, output="sos")
    if y.ndim == 1:
        filtered: np.ndarray = sosfiltfilt(sos, y).astype(y.dtype)
        return filtered
    return np.stack([sosfiltfilt(sos, ch).astype(y.dtype) for ch in y])


def _run_rubberband(
    input_path: Path,
    output_path: Path,
    *,
    n_steps: int | None = None,
    rate: float | None = None,
    sample_type: str | None = None,
) -> None:
    """Invoke Rubber Band CLI for pitch/time transformation.

    Requires `rubberband` on PATH (brew install rubberband / apt install rubberband-cli).
    When both n_steps and rate are provided, Rubber Band processes them in a
    single pass — better quality than chaining two separate operations.
    """
    cmd: list[str] = ["rubberband"]
    if n_steps is not None and n_steps != 0:
        cmd += ["-p", str(n_steps)]
    if rate is not None and abs(rate - 1.0) > 0.01:
        cmd += ["-t", str(rate)]
    cmd += ["--crisp", str(_crisp_level(sample_type))]
    cmd += [str(input_path), str(output_path)]

    result = subprocess.run(cmd, check=True, capture_output=True)
    if result.returncode != 0:
        logger.error(f"rubberband failed: {result.stderr.decode()}")


def transform_sample(
    source_path: Path,
    sample_id: str,
    *,
    source_key: str | None,
    target_key: str | None,
    source_bpm: int | None,
    target_bpm: int | None,
    sample_type: str | None = None,
) -> Path:
    """Pitch-shift and/or time-stretch an audio file using Rubber Band.

    This is CPU-bound — call via asyncio.to_thread() from async contexts.

    At least one of (target_key, target_bpm) must differ from the source.
    """
    cache_path = _get_cache_path(sample_id, target_key, target_bpm)
    if cache_path.exists():
        logger.info(f"Cache hit: {cache_path.name}")
        return cache_path

    y, sr = librosa.load(str(source_path), sr=None, mono=False)

    # --- Pre-processing ---

    # Remove DC offset (prevents low-frequency thumps after pitch shift)
    if y.ndim == 1:
        y = y - np.mean(y)
    else:
        y = y - np.mean(y, axis=-1, keepdims=True)

    # High-pass at 20 Hz to remove subsonic content
    y = _highpass(y, sr)

    # --- Compute transform parameters ---

    n_steps: int | None = None
    if source_key and target_key:
        n_steps = music_theory_service.semitone_delta(source_key, target_key)
        if n_steps is not None and n_steps != 0:
            logger.info(f"Pitch shifting {sample_id} by {n_steps:+d} semitones")
        else:
            n_steps = None

    rate: float | None = None
    if source_bpm and target_bpm:
        rate = target_bpm / source_bpm
        if abs(rate - 1.0) > 0.01:
            logger.info(f"Time stretching {sample_id} by rate {rate:.3f}")
        else:
            rate = None

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Rubber Band: single-pass pitch + time ---

    # Write pre-processed audio to temp file for rubberband input
    fd_in, tmp_in = tempfile.mkstemp(dir=cache_path.parent, suffix=".wav")
    fd_out, tmp_out = tempfile.mkstemp(dir=cache_path.parent, suffix=".wav")
    try:
        os.close(fd_in)
        os.close(fd_out)
        sf.write(tmp_in, y.T if y.ndim == 2 else y, sr, subtype="PCM_24")

        _run_rubberband(
            Path(tmp_in),
            Path(tmp_out),
            n_steps=n_steps,
            rate=rate,
            sample_type=sample_type,
        )

        # Load result for normalization
        y_out, sr_out = sf.read(tmp_out, always_2d=True)
        # sf.read returns (samples, channels); transpose to (channels, samples)
        y_out = y_out.T

        # --- Post-processing ---

        # Peak normalize to 0.95 (-0.4 dB headroom for DAW stacking)
        peak = float(np.max(np.abs(y_out)))
        if peak > 0.0:
            y_out = y_out * (0.95 / peak)

        # Hard clip guard (safety net)
        y_out = np.clip(y_out, -1.0, 1.0)

        # Write final output
        sf.write(tmp_out, y_out.T if y_out.ndim == 2 else y_out, sr_out, subtype="PCM_24")
        os.rename(tmp_out, cache_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_out)
        raise
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_in)

    logger.info(f"Cached transform: {cache_path.name}")
    return cache_path
