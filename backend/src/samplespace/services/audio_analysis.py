"""Audio analysis service using librosa for key, BPM, and duration extraction."""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def analyze_audio(file_path: str) -> dict[str, float | str | None]:
    """Analyze an audio file and extract key metadata.

    Returns dict with keys: key, bpm, duration.
    """
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    # Duration
    duration = float(librosa.get_duration(y=y, sr=sr))

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.round(tempo[0], 1)) if hasattr(tempo, "__len__") else float(np.round(tempo, 1))

    # Key detection via chroma features
    key = _detect_key(y, sr)

    logger.info(f"Analyzed {file_path}: key={key}, bpm={bpm}, duration={duration:.1f}s")

    return {
        "key": key,
        "bpm": bpm,
        "duration": round(duration, 2),
    }


def _detect_key(y: np.ndarray, sr: int) -> str | None:  # type: ignore[type-arg]
    """Detect musical key from audio using chroma features."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Major and minor profiles (Krumhansl-Schmuckler)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_corr = -1.0
    best_key = None

    for i in range(12):
        major_corr = float(np.corrcoef(np.roll(major_profile, i), chroma_mean)[0, 1])
        minor_corr = float(np.corrcoef(np.roll(minor_profile, i), chroma_mean)[0, 1])

        if major_corr > best_corr:
            best_corr = major_corr
            best_key = f"{key_names[i]} major"
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = f"{key_names[i]} minor"

    return best_key
