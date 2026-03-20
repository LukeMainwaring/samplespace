"""Shared utilities for CLI scripts."""

from pathlib import Path

from samplespace.core.config import get_settings


def find_audio_file(filename: str, sample_type: str | None) -> Path | None:
    """Locate an audio file by checking type subdirectory, root, then rglob."""
    samples_dir = Path(get_settings().SAMPLES_DIR)

    if sample_type:
        candidate = samples_dir / sample_type / filename
        if candidate.exists():
            return candidate

    candidate = samples_dir / filename
    if candidate.exists():
        return candidate

    matches = list(samples_dir.rglob(filename))
    return matches[0] if matches else None
