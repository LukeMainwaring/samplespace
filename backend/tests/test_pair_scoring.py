"""Tests for pair scoring dimensions (key, BPM, type, spectral) and music theory utilities."""

import numpy as np

from samplespace.services.music_theory import (
    are_relative_pairs,
    key_compatibility_score,
    key_distance,
    normalize_bpm,
    semitone_delta,
)
from samplespace.services.pair_scoring import (
    DEFAULT_TYPE_SCORE,
    TYPE_COMPLEMENTARITY,
    _compute_bpm_score,
    _compute_spectral_score,
    _compute_type_score,
    cosine_similarity,
)


class TestKeyCompatibility:
    def test_same_key_is_perfect(self) -> None:
        score, _ = key_compatibility_score("D minor", "D minor")
        assert score == 1.0

    def test_relative_major_minor(self) -> None:
        score, _ = key_compatibility_score("C major", "A minor")
        assert score == 0.95

    def test_adjacent_on_circle_of_fifths(self) -> None:
        score, _ = key_compatibility_score("C major", "G major")
        assert score == 0.85

    def test_distant_keys_score_low(self) -> None:
        score, _ = key_compatibility_score("C major", "F# major")
        assert score <= 0.2

    def test_unparseable_key_returns_neutral(self) -> None:
        score, _ = key_compatibility_score("C major", "unknown")
        assert score == 0.5


class TestKeyDistance:
    def test_same_root_is_zero(self) -> None:
        assert key_distance("C major", "C minor") == 0

    def test_adjacent_fifths(self) -> None:
        assert key_distance("C major", "G major") == 1

    def test_tritone_is_six(self) -> None:
        assert key_distance("C major", "F# major") == 6

    def test_symmetry(self) -> None:
        assert key_distance("A minor", "D major") == key_distance("D major", "A minor")


class TestSemitoneDelta:
    def test_unison(self) -> None:
        assert semitone_delta("C major", "C minor") == 0

    def test_positive_shift(self) -> None:
        assert semitone_delta("C major", "D major") == 2

    def test_negative_shift(self) -> None:
        assert semitone_delta("D major", "C major") == -2

    def test_tritone_is_positive_six(self) -> None:
        assert semitone_delta("C major", "F# major") == 6


class TestRelativePairs:
    def test_c_major_a_minor(self) -> None:
        assert are_relative_pairs("C major", "A minor") is True

    def test_reverse_direction(self) -> None:
        assert are_relative_pairs("A minor", "C major") is True

    def test_non_relative(self) -> None:
        assert are_relative_pairs("C major", "D minor") is False


class TestNormalizeBPM:
    def test_already_in_range(self) -> None:
        assert normalize_bpm(120) == 120

    def test_halves_high_bpm(self) -> None:
        assert normalize_bpm(240) == 120

    def test_doubles_low_bpm(self) -> None:
        assert normalize_bpm(55) == 110

    def test_zero_returns_zero(self) -> None:
        assert normalize_bpm(0) == 0


class TestBPMScore:
    def test_identical_bpm(self) -> None:
        result = _compute_bpm_score(120, 120)
        assert result.value == 1.0

    def test_double_half_compatible(self) -> None:
        result = _compute_bpm_score(120, 240)
        assert result.value == 1.0  # 240 normalizes to 120

    def test_close_bpm(self) -> None:
        result = _compute_bpm_score(120, 125)
        assert result.value > 0.9

    def test_distant_bpm(self) -> None:
        result = _compute_bpm_score(80, 140)
        assert result.value < 0.8


class TestTypeScore:
    def test_kick_hihat_high_complementarity(self) -> None:
        result = _compute_type_score("kick", "hihat")
        assert result.value == 0.9

    def test_same_type_low_score(self) -> None:
        result = _compute_type_score("kick", "kick")
        assert result.value == 0.2

    def test_unknown_type_gets_default(self) -> None:
        result = _compute_type_score("unknown_a", "unknown_b")
        assert result.value == DEFAULT_TYPE_SCORE

    def test_type_complementarity_is_symmetric(self) -> None:
        for pair_key, score in TYPE_COMPLEMENTARITY.items():
            types = list(pair_key)
            if len(types) == 2:
                result_ab = _compute_type_score(types[0], types[1])
                result_ba = _compute_type_score(types[1], types[0])
                assert result_ab.value == result_ba.value


class TestSpectralScore:
    def test_identical_embeddings_complementary_types(self) -> None:
        emb = [1.0] * 128
        result = _compute_spectral_score(emb, emb, types_are_complementary=True)
        # Identical embeddings = zero distance; for complementary types, distance IS the score
        assert result.value == 0.0

    def test_identical_embeddings_same_type(self) -> None:
        emb = [1.0] * 128
        result = _compute_spectral_score(emb, emb, types_are_complementary=False)
        # Identical embeddings = high similarity; for same type, similarity IS the score
        assert result.value == 1.0

    def test_orthogonal_embeddings_complementary_types(self) -> None:
        emb_a = [1.0] + [0.0] * 127
        emb_b = [0.0] + [1.0] + [0.0] * 126
        result = _compute_spectral_score(emb_a, emb_b, types_are_complementary=True)
        assert result.value == 1.0  # max distance = best for complementary


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert np.isclose(cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self) -> None:
        assert np.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_vectors(self) -> None:
        assert np.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
