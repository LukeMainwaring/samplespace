"""Audio analysis service using librosa for key, BPM, and duration extraction."""

import logging

import librosa
import numpy as np

from samplespace.schemas.audio import AudioMetadata

logger = logging.getLogger(__name__)

LOOP_KEYWORDS = {"loop", "loops", "looped"}
ONE_SHOT_KEYWORDS = {"one-shot", "oneshot", "one_shot", "one shot", "hit", "hits", "single"}


def infer_is_loop(file_path: str) -> bool:
    """Infer whether an audio file is a loop or a one-shot.

    Uses a tiered approach:
    1. Check filepath for explicit keywords (loop, one-shot, hit, etc.)
    2. Fall back to audio heuristics (duration, onset density, onset regularity)
    """
    path_lower = file_path.lower()

    # Tier 1: Filepath keywords
    for keyword in LOOP_KEYWORDS:
        if keyword in path_lower:
            logger.info(f"Inferred is_loop=True for {file_path} (keyword: {keyword})")
            return True
    for keyword in ONE_SHOT_KEYWORDS:
        if keyword in path_lower:
            logger.info(f"Inferred is_loop=False for {file_path} (keyword: {keyword})")
            return False

    # Tier 2: Audio heuristics
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    num_onsets = len(onsets)

    is_short = duration < 2.0
    is_long = duration > 4.0
    few_onsets = num_onsets <= 2

    # Check onset regularity for potential loops
    has_regular_rhythm = False
    if num_onsets >= 4:
        onset_times = librosa.frames_to_time(onsets, sr=sr)
        intervals = np.diff(onset_times)
        if len(intervals) > 0 and np.mean(intervals) > 0:
            cv = float(np.std(intervals) / np.mean(intervals))
            has_regular_rhythm = cv < 0.25

    if is_short and few_onsets:
        result = False
    elif is_long or has_regular_rhythm:
        result = True
    else:
        result = False  # Default to one-shot for ambiguous cases

    logger.info(f"Inferred is_loop={result} for {file_path} (duration={duration:.1f}s, onsets={num_onsets}, heuristic)")
    return result


def analyze_audio(file_path: str, *, is_loop: bool = True) -> AudioMetadata:
    """Analyze an audio file and extract key metadata.

    When is_loop is False, only computes duration (key/BPM are meaningless for one-shots).
    """
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    # Duration (always computed)
    duration = float(librosa.get_duration(y=y, sr=sr))

    if not is_loop:
        logger.info(f"Analyzed {file_path}: one-shot, duration={duration:.1f}s (skipped key/BPM)")
        return AudioMetadata(key=None, bpm=None, duration=round(duration, 2))

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    bpm = round(tempo_val, 1)

    # Key detection via chroma features
    key = _detect_key(y, int(sr))

    logger.info(f"Analyzed {file_path}: key={key}, bpm={bpm}, duration={duration:.1f}s")

    return AudioMetadata(key=key, bpm=bpm, duration=round(duration, 2))


def _detect_key(y: np.ndarray, sr: int) -> str | None:
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
