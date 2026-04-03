"""Preference model service for learning pairing taste from verdicts.

Trains a logistic regression on 10-dimensional feature vectors (4 pair scores +
6 relational audio features) to predict whether a user will accept a sample pair.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from samplespace.models.pair_verdict import PairVerdict
from samplespace.schemas.preference import PreferenceExplanation, PreferenceMeta

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

FEATURE_NAMES: list[str] = [
    "key_score",
    "bpm_score",
    "type_score",
    "spectral_score",
    "spectral_overlap",
    "onset_alignment",
    "timbral_contrast",
    "harmonic_consonance",
    "spectral_centroid_gap",
    "rms_energy_ratio",
]

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "key_score": "key compatibility",
    "bpm_score": "BPM compatibility",
    "type_score": "type complementarity",
    "spectral_score": "spectral similarity",
    "spectral_overlap": "spectral overlap",
    "onset_alignment": "onset alignment",
    "timbral_contrast": "timbral contrast",
    "harmonic_consonance": "harmonic consonance",
    "spectral_centroid_gap": "spectral centroid gap",
    "rms_energy_ratio": "energy balance",
}

# Descriptions keyed by (feature_name, direction) where direction is "positive"
# (higher value → more likely to accept) or "negative" (higher value → reject).
_FEATURE_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "key_score": (
        "You strongly prefer pairs that are **harmonically compatible**",
        "You tend to favor pairs with **contrasting keys**",
    ),
    "bpm_score": (
        "**BPM alignment** is a top priority in your pairings",
        "You're comfortable with **tempo variation** between paired samples",
    ),
    "type_score": (
        "You prefer **complementary sample roles** (e.g., kick + pad over kick + kick)",
        "You favor pairing **similar sample types** together",
    ),
    "spectral_score": (
        "**Spectral compatibility** matters a lot to your ear",
        "You gravitate toward pairs with **contrasting spectral profiles**",
    ),
    "spectral_overlap": (
        "You like pairs that **share frequency space** — dense, layered textures",
        "You prefer pairs that occupy **distinct frequency bands** — clean separation",
    ),
    "onset_alignment": (
        "**Rhythmic synchrony** matters — you prefer transients that land together",
        "You favor pairs with **interleaving transients** — complementary rhythm",
    ),
    "timbral_contrast": (
        "You strongly prefer pairs with **distinct timbral character**",
        "You lean toward pairs with **similar timbral quality** — cohesive sound",
    ),
    "harmonic_consonance": (
        "**Harmonic consonance** is important — you prefer pairs that share tonal content",
        "You're drawn to **harmonically independent** pairs",
    ),
    "spectral_centroid_gap": (
        "You favor pairs that occupy **different frequency registers** (e.g., bass + treble)",
        "You prefer pairs in a **similar frequency range**",
    ),
    "rms_energy_ratio": (
        "**Energy balance** matters — you notice loudness differences between paired samples",
        "Loudness differences between samples have **minimal influence** on your preferences",
    ),
}

MIN_VERDICTS = 15
MIN_PER_CLASS = 3
RETRAIN_INTERVAL = 5

_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "models"
MODEL_PATH = _DATA_DIR / "preference_model.joblib"
META_PATH = _DATA_DIR / "preference_meta.json"

# --------------------------------------------------------------------------- #
# In-memory cache (single-process dev server)
# --------------------------------------------------------------------------- #

_cached_model: Pipeline | None = None
_cached_meta: PreferenceMeta | None = None
_cache_loaded: bool = False


def _invalidate_cache() -> None:
    global _cached_model, _cached_meta, _cache_loaded
    _cached_model = None
    _cached_meta = None
    _cache_loaded = False


def _ensure_cache() -> None:
    global _cached_model, _cached_meta, _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True

    if MODEL_PATH.exists():
        try:
            _cached_model = joblib.load(MODEL_PATH)
        except Exception:
            logger.warning("Failed to load preference model from disk", exc_info=True)
            _cached_model = None

    if META_PATH.exists():
        try:
            _cached_meta = PreferenceMeta.model_validate_json(META_PATH.read_text())
        except Exception:
            logger.warning("Failed to load preference metadata from disk", exc_info=True)
            _cached_meta = None


# --------------------------------------------------------------------------- #
# Feature vector construction
# --------------------------------------------------------------------------- #

_PAIR_SCORE_FIELDS = ["key_score", "bpm_score", "type_score", "spectral_score"]
_AUDIO_FEATURE_FIELDS = [
    "spectral_overlap",
    "onset_alignment",
    "timbral_contrast",
    "harmonic_consonance",
    "spectral_centroid_gap",
    "rms_energy_ratio",
]
_NEUTRAL = 0.5


def build_feature_vector(
    pair_score_detail: dict[str, Any],
    pair_features: dict[str, Any],
) -> list[float]:
    """Build a 10-dimensional feature vector from a verdict's stored data."""
    vector: list[float] = []

    # 4 pair scoring dimensions (impute 0.5 for missing)
    for field in _PAIR_SCORE_FIELDS:
        dim = pair_score_detail.get(field)
        vector.append(dim["value"] if isinstance(dim, dict) and "value" in dim else _NEUTRAL)

    # 6 relational audio features
    for field in _AUDIO_FEATURE_FIELDS:
        vector.append(float(pair_features.get(field, _NEUTRAL)))

    return vector


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


async def train(db: AsyncSession) -> PreferenceMeta | None:
    """Train the preference model on all verdicts with extracted features.

    Returns metadata on success, None if insufficient data.
    """
    stmt = select(PairVerdict).where(PairVerdict.pair_features.isnot(None))
    result = await db.execute(stmt)
    verdicts = result.scalars().all()

    X: list[list[float]] = []
    y: list[int] = []

    for v in verdicts:
        if v.pair_score_detail is None or v.pair_features is None:
            continue
        X.append(build_feature_vector(v.pair_score_detail, v.pair_features))
        y.append(1 if v.verdict else 0)

    if len(X) < MIN_VERDICTS:
        logger.info(f"Only {len(X)} complete verdicts with features (need {MIN_VERDICTS})")
        return None

    y_arr = np.array(y)
    class_counts = np.bincount(y_arr, minlength=2)
    if class_counts[0] < MIN_PER_CLASS or class_counts[1] < MIN_PER_CLASS:
        logger.info(
            f"Insufficient class balance: {class_counts[0]} rejected, {class_counts[1]} approved "
            f"(need {MIN_PER_CLASS} of each)"
        )
        return None

    X_arr = np.array(X)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]
    )

    # Cross-validation: LOOCV for small n, stratified k-fold otherwise
    n = len(y)
    min_class = int(class_counts.min())
    if min_class < 5:
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)

    scores = cross_val_score(pipeline, X_arr, y_arr, cv=cv, scoring="accuracy")
    accuracy = float(scores.mean())

    # Train final model on all data
    pipeline.fit(X_arr, y_arr)

    # Extract feature importances from logistic regression coefficients
    clf: LogisticRegression = pipeline.named_steps["clf"]
    raw_importances = np.abs(clf.coef_[0])
    total = raw_importances.sum()
    normalized = (raw_importances / total) if total > 0 else raw_importances
    feature_importances = {name: float(normalized[i]) for i, name in enumerate(FEATURE_NAMES)}

    # Determine version from existing metadata
    _ensure_cache()
    prev_version = _cached_meta.version if _cached_meta else 0

    meta = PreferenceMeta(
        version=prev_version + 1,
        accuracy=accuracy,
        verdict_count=n,
        feature_importances=feature_importances,
        trained_at=datetime.now(timezone.utc),
    )

    # Save to disk
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    META_PATH.write_text(meta.model_dump_json(indent=2))

    _invalidate_cache()
    logger.info(f"Trained preference model v{meta.version} — accuracy: {accuracy:.1%}, verdicts: {n}")

    return meta


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


def predict(features: list[float]) -> float:
    """Return P(accept) for a feature vector. Returns 0.5 if no model exists."""
    _ensure_cache()
    if _cached_model is None:
        return _NEUTRAL

    proba = _cached_model.predict_proba(np.array([features]))[0]
    # Class 1 = accepted
    return float(proba[1])


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #


def explain() -> PreferenceExplanation | None:
    """Return a structured explanation of learned preferences, or None if no model."""
    _ensure_cache()
    if _cached_meta is None:
        return None

    meta = _cached_meta

    # Load raw coefficients for direction (positive = increases P(accept))
    coef_directions: dict[str, str] = {}
    if _cached_model is not None:
        clf: LogisticRegression = _cached_model.named_steps["clf"]
        for i, name in enumerate(FEATURE_NAMES):
            coef_directions[name] = "positive" if clf.coef_[0][i] >= 0 else "negative"

    # Sort by importance descending
    sorted_features = sorted(meta.feature_importances.items(), key=lambda x: -x[1])

    top_features: list[tuple[str, float, str]] = []
    summary_lines: list[str] = [
        f"Based on your feedback ({meta.verdict_count} verdicts, {meta.accuracy:.0%} prediction accuracy):\n"
    ]

    for name, importance in sorted_features:
        display_name = FEATURE_DISPLAY_NAMES.get(name, name)
        direction = coef_directions.get(name, "positive")
        direction_idx = 0 if direction == "positive" else 1
        description = _FEATURE_DESCRIPTIONS.get(name, (display_name, display_name))[direction_idx]
        top_features.append((display_name, importance, direction))

        if importance >= 0.12:
            summary_lines.append(f"- {description} (importance: {importance:.0%})")
        elif importance >= 0.08:
            summary_lines.append(f"- {display_name} has moderate influence ({importance:.0%})")
        else:
            summary_lines.append(f"- {display_name} has low influence ({importance:.0%})")

    summary = "\n".join(summary_lines)

    return PreferenceExplanation(meta=meta, summary=summary, top_features=top_features)


# --------------------------------------------------------------------------- #
# Retrain check
# --------------------------------------------------------------------------- #


def should_retrain(verdict_count: int) -> bool:
    return verdict_count >= MIN_VERDICTS and verdict_count % RETRAIN_INTERVAL == 0
