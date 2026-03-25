"""Relational audio feature extraction for sample pairs.

Computes 6 features that characterize how two audio samples relate to each
other sonically. All functions are synchronous (CPU-bound) — call via
asyncio.to_thread() from async contexts.

Uses librosa exclusively (consistent with audio_analysis.py).
"""

import logging
from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Standard sample rate for all analysis
_SR = 22050

# Standard analysis duration — truncate/pad both samples to this
_ANALYSIS_DURATION_S = 4.0
_ANALYSIS_SAMPLES = int(_SR * _ANALYSIS_DURATION_S)


def _load_and_normalize(path: Path) -> NDArray[np.floating]:
    """Load audio at standard sample rate, mono, truncated/padded to fixed length."""
    y, _ = librosa.load(str(path), sr=_SR, mono=True)
    if len(y) > _ANALYSIS_SAMPLES:
        y = y[:_ANALYSIS_SAMPLES]
    elif len(y) < _ANALYSIS_SAMPLES:
        y = np.pad(y, (0, _ANALYSIS_SAMPLES - len(y)))
    return y


def _spectral_overlap(
    y_a: NDArray[np.floating],
    y_b: NDArray[np.floating],
) -> float:
    """Frequency spectrum overlap as intersection-over-union.

    High values (close to 1) mean the samples compete for the same
    frequency bands. Low values mean they occupy distinct spectral regions.
    """
    S_a = np.abs(librosa.stft(y_a))
    S_b = np.abs(librosa.stft(y_b))

    # Normalize each to sum=1 so we're comparing spectral shape, not amplitude
    S_a_norm = S_a / (S_a.sum() + 1e-10)
    S_b_norm = S_b / (S_b.sum() + 1e-10)

    intersection = np.minimum(S_a_norm, S_b_norm).sum()
    union = np.maximum(S_a_norm, S_b_norm).sum()

    if union < 1e-10:
        return 0.0
    return float(np.clip(intersection / union, 0.0, 1.0))


def _onset_alignment(
    y_a: NDArray[np.floating],
    y_b: NDArray[np.floating],
) -> float:
    """Rhythmic onset correlation.

    High values mean transients tend to co-occur (potentially masking).
    Low values mean transients interleave (complementary rhythms).
    """
    onset_a = librosa.onset.onset_strength(y=y_a, sr=_SR)
    onset_b = librosa.onset.onset_strength(y=y_b, sr=_SR)

    # Ensure same length
    min_len = min(len(onset_a), len(onset_b))
    onset_a = onset_a[:min_len]
    onset_b = onset_b[:min_len]

    # Normalize
    norm_a = np.linalg.norm(onset_a)
    norm_b = np.linalg.norm(onset_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    correlation = np.correlate(onset_a / norm_a, onset_b / norm_b, mode="full")
    peak = float(np.max(correlation))
    return float(np.clip(peak, 0.0, 1.0))


def _timbral_contrast(
    y_a: NDArray[np.floating],
    y_b: NDArray[np.floating],
) -> float:
    """MFCC-based timbral distance.

    High values mean very different timbral character.
    Low values mean similar timbral quality.
    """
    mfcc_a = librosa.feature.mfcc(y=y_a, sr=_SR, n_mfcc=13)
    mfcc_b = librosa.feature.mfcc(y=y_b, sr=_SR, n_mfcc=13)

    mean_a = mfcc_a.mean(axis=1)
    mean_b = mfcc_b.mean(axis=1)

    # Cosine distance
    dot = np.dot(mean_a, mean_b)
    norm_a = np.linalg.norm(mean_a)
    norm_b = np.linalg.norm(mean_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.5

    cosine_sim = dot / (norm_a * norm_b)
    cosine_dist = 1.0 - cosine_sim
    # Cosine distance is in [0, 2]; normalize to [0, 1]
    return float(np.clip(cosine_dist / 2.0, 0.0, 1.0))


def _harmonic_consonance(
    y_a: NDArray[np.floating],
    y_b: NDArray[np.floating],
) -> float:
    """Chroma-based harmonic consonance.

    High values mean harmonically consonant content (shared key/scale).
    Low values mean dissonant or unrelated harmonic content.
    """
    chroma_a = librosa.feature.chroma_cqt(y=y_a, sr=_SR)
    chroma_b = librosa.feature.chroma_cqt(y=y_b, sr=_SR)

    mean_a = chroma_a.mean(axis=1)
    mean_b = chroma_b.mean(axis=1)

    # Pearson correlation of mean chroma vectors
    correlation = np.corrcoef(mean_a, mean_b)[0, 1]
    if np.isnan(correlation):
        return 0.5

    # Map from [-1, 1] to [0, 1]
    return float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))


def _spectral_centroid_gap(
    y_a: NDArray[np.floating],
    y_b: NDArray[np.floating],
) -> float:
    """Normalized spectral centroid difference.

    High values mean the samples occupy very different frequency registers
    (e.g., bass vs treble). Low values mean similar register.
    """
    centroid_a = librosa.feature.spectral_centroid(y=y_a, sr=_SR)
    centroid_b = librosa.feature.spectral_centroid(y=y_b, sr=_SR)

    mean_a = float(centroid_a.mean())
    mean_b = float(centroid_b.mean())

    # Normalize by Nyquist frequency
    nyquist = _SR / 2.0
    gap = abs(mean_a - mean_b) / nyquist
    return float(np.clip(gap, 0.0, 1.0))


def _rms_energy_ratio(
    y_a: NDArray[np.floating],
    y_b: NDArray[np.floating],
) -> float:
    """Normalized RMS energy ratio.

    Values near 0.5 mean balanced loudness. Values near 0 or 1 mean
    one sample is much louder than the other.
    """
    rms_a = float(np.sqrt(np.mean(y_a**2)))
    rms_b = float(np.sqrt(np.mean(y_b**2)))

    if rms_a < 1e-10 and rms_b < 1e-10:
        return 0.5

    # Log ratio, clamped to +-3 (~20dB range), then normalized to [0, 1]
    if rms_b < 1e-10:
        log_ratio = 3.0
    elif rms_a < 1e-10:
        log_ratio = -3.0
    else:
        log_ratio = float(np.log10(rms_a / rms_b))
        log_ratio = float(np.clip(log_ratio, -3.0, 3.0))

    # Map [-3, 3] to [0, 1]
    return float((log_ratio + 3.0) / 6.0)


def compute_pair_features(audio_path_a: Path, audio_path_b: Path) -> dict[str, float]:
    """Compute all 6 relational audio features between two samples.

    CPU-bound — call via asyncio.to_thread() from async contexts.

    Returns:
        Dict with keys: spectral_overlap, onset_alignment, timbral_contrast,
        harmonic_consonance, spectral_centroid_gap, rms_energy_ratio.
        All values normalized to [0, 1].
    """
    logger.info(f"Computing pair features: {audio_path_a.name} + {audio_path_b.name}")

    y_a = _load_and_normalize(audio_path_a)
    y_b = _load_and_normalize(audio_path_b)

    features = {
        "spectral_overlap": _spectral_overlap(y_a, y_b),
        "onset_alignment": _onset_alignment(y_a, y_b),
        "timbral_contrast": _timbral_contrast(y_a, y_b),
        "harmonic_consonance": _harmonic_consonance(y_a, y_b),
        "spectral_centroid_gap": _spectral_centroid_gap(y_a, y_b),
        "rms_energy_ratio": _rms_energy_ratio(y_a, y_b),
    }

    logger.info(f"Pair features computed: {features}")
    return features
