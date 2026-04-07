"""Tests for the preference model service (feature vectors, retrain logic, prediction)."""

from unittest.mock import patch

from samplespace.services.preference import (
    _NEUTRAL,
    MIN_VERDICTS,
    RETRAIN_INTERVAL,
    build_feature_vector,
    predict,
    should_retrain,
)


class TestBuildFeatureVector:
    def test_complete_inputs(self) -> None:
        pair_score_detail = {
            "key_score": {"value": 0.95, "weight": 0.3},
            "bpm_score": {"value": 0.8, "weight": 0.2},
            "type_score": {"value": 0.7, "weight": 0.25},
            "spectral_score": {"value": 0.6, "weight": 0.25},
        }
        pair_features = {
            "spectral_overlap": 0.45,
            "onset_alignment": 0.55,
            "timbral_contrast": 0.65,
            "harmonic_consonance": 0.75,
            "spectral_centroid_gap": 0.35,
            "rms_energy_ratio": 0.5,
        }
        vector = build_feature_vector(pair_score_detail, pair_features)
        assert len(vector) == 10
        assert vector == [0.95, 0.8, 0.7, 0.6, 0.45, 0.55, 0.65, 0.75, 0.35, 0.5]

    def test_missing_pair_scores_impute_neutral(self) -> None:
        vector = build_feature_vector({}, {})
        assert len(vector) == 10
        assert vector == [_NEUTRAL] * 10

    def test_partial_pair_scores(self) -> None:
        pair_score_detail = {
            "key_score": {"value": 1.0, "weight": 0.3},
            # bpm_score, type_score, spectral_score missing
        }
        pair_features = {
            "spectral_overlap": 0.9,
            # rest missing
        }
        vector = build_feature_vector(pair_score_detail, pair_features)
        assert vector[0] == 1.0  # key_score present
        assert vector[1] == _NEUTRAL  # bpm_score missing
        assert vector[4] == 0.9  # spectral_overlap present
        assert vector[5] == _NEUTRAL  # onset_alignment missing

    def test_malformed_pair_score_imputes_neutral(self) -> None:
        """A pair score without 'value' key should impute to neutral."""
        pair_score_detail = {
            "key_score": {"explanation": "no value field"},
        }
        vector = build_feature_vector(pair_score_detail, {})
        assert vector[0] == _NEUTRAL


class TestShouldRetrain:
    def test_below_minimum_returns_false(self) -> None:
        assert should_retrain(MIN_VERDICTS - 1) is False

    def test_at_minimum_returns_true(self) -> None:
        assert should_retrain(MIN_VERDICTS) is True

    def test_between_intervals_returns_false(self) -> None:
        assert should_retrain(MIN_VERDICTS + 1) is False

    def test_at_next_interval_returns_true(self) -> None:
        assert should_retrain(MIN_VERDICTS + RETRAIN_INTERVAL) is True

    def test_zero_verdicts(self) -> None:
        assert should_retrain(0) is False


class TestPredict:
    def test_no_model_returns_neutral(self) -> None:
        with (
            patch("samplespace.services.preference._cached_model", None),
            patch("samplespace.services.preference._cache_loaded", True),
        ):
            result = predict([0.5] * 10)
            assert result == _NEUTRAL
