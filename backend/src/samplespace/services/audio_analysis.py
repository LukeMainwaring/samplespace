"""Audio analysis service using librosa for key, BPM, and duration extraction."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from samplespace.schemas.audio import AudioMetadata

logger = logging.getLogger(__name__)

_LOOP_PATTERN = re.compile(r"\b(?:loop|loops|looped)\b", re.IGNORECASE)
_ONE_SHOT_PATTERN = re.compile(r"\b(?:one[-_ ]?shot|oneshot|hit|hits|single)\b", re.IGNORECASE)
_BPM_KEYWORD_PATTERN = re.compile(r"(\d{2,3})[\s_-]*bpm|bpm[\s_-]*(\d{2,3})", re.IGNORECASE)
_NOTE_BEFORE_NUMBER_PATTERN = re.compile(r"[A-Ga-g][#b]?$")


@dataclass
class AnalysisResult:
    """Combined result of loop inference and audio analysis."""

    is_loop: bool
    metadata: AudioMetadata


def analyze_and_classify(file_path: str) -> AnalysisResult:
    """Infer one-shot/loop classification and extract audio metadata in a single pass.

    Uses a tiered approach for loop inference:
    1. Check filepath for explicit keywords (loop, one-shot, hit, etc.)
    2. Fall back to audio heuristics (duration, onset density, onset regularity)

    When classified as a one-shot, skips key/BPM extraction (meaningless for single hits).
    """
    # Tier 1: Filepath keywords (word-boundary matching to avoid false positives)
    loop_match = _LOOP_PATTERN.search(file_path)
    if loop_match:
        logger.info(f"Inferred is_loop=True for {file_path} (keyword: {loop_match.group()})")
        return AnalysisResult(
            is_loop=True,
            metadata=_analyze_audio(file_path),
        )
    one_shot_match = _ONE_SHOT_PATTERN.search(file_path)
    if one_shot_match:
        logger.info(f"Inferred is_loop=False for {file_path} (keyword: {one_shot_match.group()})")
        return AnalysisResult(
            is_loop=False,
            metadata=_analyze_duration_only(file_path),
        )

    # Tier 2: Audio heuristics (load once, reuse for analysis)
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    is_loop = _infer_from_audio(file_path, y, int(sr))

    if is_loop:
        metadata = _extract_full_metadata(file_path, y, int(sr))
    else:
        duration = round(float(librosa.get_duration(y=y, sr=sr)), 2)
        metadata = AudioMetadata(key=None, bpm=None, duration=duration)
        logger.info(f"Analyzed {file_path}: one-shot, duration={duration:.1f}s (skipped key/BPM)")

    return AnalysisResult(is_loop=is_loop, metadata=metadata)


def _infer_from_audio(file_path: str, y: np.ndarray, sr: int) -> bool:
    """Infer one-shot/loop from audio features (tier 2 heuristic)."""
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
        if np.mean(intervals) > 0:
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


def _analyze_audio(file_path: str) -> AudioMetadata:
    """Full audio analysis (key, BPM, duration) — loads file from disk."""
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    return _extract_full_metadata(file_path, y, int(sr))


def _analyze_duration_only(file_path: str) -> AudioMetadata:
    """Duration-only analysis for one-shots — loads file from disk."""
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    duration = round(float(librosa.get_duration(y=y, sr=sr)), 2)
    logger.info(f"Analyzed {file_path}: one-shot, duration={duration:.1f}s (skipped key/BPM)")
    return AudioMetadata(key=None, bpm=None, duration=duration)


def _extract_bpm_from_filename(filename: str) -> float | None:
    """Extract BPM from a sample filename using tiered heuristics.

    Tier 1 (high confidence): number adjacent to "bpm" keyword (e.g., 123bpm, bpm_120).
    Tier 2 (medium confidence): sole number in 50-200 range after filtering leading IDs,
    out-of-range values, and note-adjacent octave numbers.

    Returns None if no confident match, signaling fallback to librosa.
    """
    stem = Path(filename).stem

    # Tier 1: explicit "bpm" keyword adjacent to a number
    keyword_match = _BPM_KEYWORD_PATTERN.search(stem)
    if keyword_match:
        value = int(keyword_match.group(1) or keyword_match.group(2))
        if 50 <= value <= 200:
            logger.info(f"BPM={value} from filename keyword: {filename}")
            return float(value)

    # Tier 2: positional heuristics for bare numbers
    candidates: list[int] = []
    for m in re.finditer(r"\d+", stem):
        value = int(m.group())
        if not (50 <= value <= 200):
            continue
        if m.start() == 0:
            continue
        prefix = stem[: m.start()]
        if _NOTE_BEFORE_NUMBER_PATTERN.search(prefix):
            continue
        candidates.append(value)

    if len(candidates) == 1:
        logger.info(f"BPM={candidates[0]} from filename heuristic: {filename}")
        return float(candidates[0])

    return None


def _extract_full_metadata(file_path: str, y: np.ndarray, sr: int) -> AudioMetadata:
    """Extract key, BPM, and duration from already-loaded audio."""
    duration = float(librosa.get_duration(y=y, sr=sr))

    filename = Path(file_path).name
    filename_bpm = _extract_bpm_from_filename(filename)

    if filename_bpm is not None:
        bpm = filename_bpm
    else:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        bpm = round(tempo_val, 1)

    key = _detect_key(y, sr)

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
