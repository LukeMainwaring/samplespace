"""Shared utilities for CLI scripts."""

from pathlib import Path

from samplespace.core.config import get_settings
from samplespace.models.sample import Sample


def find_audio_file(sample: Sample) -> Path | None:
    """Locate an audio file using source-aware path resolution."""
    settings = get_settings()

    if sample.source == "splice" and settings.SPLICE_DIR:
        candidate = Path(settings.SPLICE_DIR) / sample.relative_path
        if candidate.exists():
            return candidate
        return None

    # Local source: resolve against SAMPLES_DIR
    samples_dir = Path(settings.SAMPLES_DIR)
    candidate = samples_dir / sample.relative_path
    if candidate.exists():
        return candidate

    # Fallback: try rglob for backwards compatibility
    matches = list(samples_dir.rglob(sample.filename))
    return matches[0] if matches else None
