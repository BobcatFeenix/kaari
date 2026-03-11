"""
Kaari Test Suite — Tests detection behavior.

Tests verify:
1. Math primitives work correctly (cosine, delta, c2)
2. Clean inputs score LOW (not flagged)
3. Drifted inputs score HIGH (flagged)
4. Tier logic routes correctly
5. Edge cases don't crash
6. Input validation catches bad data with actionable errors

Run: pytest kaari/tests/ -v
"""

import math
import numpy as np
import pytest

from kaari.core.scoring import (
    cosine_similarity,
    calculate_delta,
    calculate_c2,
    score,
    ScoringResult,
    KaariError,
    KaariInputError,
)
from kaari.core.thresholds import DEFAULT_CONFIG, get_config


# --- Math primitives ---


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 0]) == 0.0

    def test_numpy_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = cosine_similarity(a, b)
        assert 0.0 < result < 1.0

    def test_high_dimensional(self):
        """768-dim like nomic-embed-text."""
        rng = np.random.RandomState(42)
        a = rng.randn(768)
        b = rng.randn(768)
        result = cosine_similarity(a, b)
        assert -1.0 <= result <= 1.0


class TestCalculateDelta:
    def test_identical_is_zero(self):
        v = [1.0, 2.0, 3.0]
        assert calculate_delta(v, v) == pytest.approx(0.0)

    def test_orthogonal_is_one(self):
        assert calculate_delta([1, 0], [0, 1]) == pytest.approx(1.0)

    def test_opposite_is_two(self):
        assert calculate_delta([1, 0], [-1, 0]) == pytest.approx(2.0)

    def test_range_zero_to_two(self):
        """Delta should be in [0, 2] for any vectors."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            a = rng.randn(768)
            b = rng.randn(768)
            d = calculate_delta(a, b)
            assert 0.0 <= d <= 2.0


class TestCalculateC2:
    def test_verbose_response_amplifies(self):
        """Longer-than-average response should increase score."""
        dv2 = 0.3
        c2 = calculate_c2(dv2, 2000, 774.0)
        assert c2 > dv2

    def test_short_response_dampens(self):
        """Shorter-than-average response should decrease score."""
        dv2 = 0.3
        c2 = calculate_c2(dv2, 200, 774.0)
        assert c2 < dv2

    def test_mean_length_unchanged(self):
        """At mean length, C2 ≈ Δv2 (log(1) = 0)."""
        dv2 = 0.3
        c2 = calculate_c2(dv2, 774, 774.0)
        assert c2 == pytest.approx(dv2, abs=0.001)

    def test_zero_length_returns_dv2(self):
        assert calculate_c2(0.3, 0, 500) == 0.3

    def test_zero_mean_returns_dv2(self):
        assert calculate_c2(0.3, 500, 0) == 0.3

    def test_coefficient_is_half(self):
        """Verify the 0.5 coefficient."""
        dv2 = 0.3
        resp_len = 1000
        mean_len = 500
        expected = dv2 * (1 + 0.5 * math.log(resp_len / mean_len))
        assert calculate_c2(dv2, resp_len, mean_len) == pytest.approx(expected)


# --- Detection behavior ---


class TestDetectionVerification:
    """Tests that the scoring pipeline correctly separates clean from drifted.

    Uses synthetic vectors with controlled geometry to verify score() logic
    in isolation (no embedding provider required). For end-to-end validation
    with real embeddings, see TestIntegration below (requires Ollama).
    """

    @pytest.fixture
    def clean_pair(self):
        """Simulate aligned prompt-response (small angle)."""
        rng = np.random.RandomState(42)
        base = rng.randn(384)
        base = base / np.linalg.norm(base)
        noise = rng.randn(384) * 0.01
        similar = base + noise
        similar = similar / np.linalg.norm(similar)
        return base, similar

    @pytest.fixture
    def drifted_pair(self):
        """Simulate misaligned prompt-response (large angle)."""
        rng = np.random.RandomState(42)
        prompt = rng.randn(384)
        prompt = prompt / np.linalg.norm(prompt)
        response = rng.randn(384)  # unrelated direction
        response = response / np.linalg.norm(response)
        return prompt, response

    def test_clean_not_flagged(self, clean_pair):
        prompt_emb, response_emb = clean_pair
        result = score(prompt_emb, response_emb, 500, DEFAULT_CONFIG, tier="fast")
        assert result.injected is False
        assert result.risk < 30

    def test_drifted_flagged(self, drifted_pair):
        prompt_emb, response_emb = drifted_pair
        result = score(prompt_emb, response_emb, 500, DEFAULT_CONFIG, tier="fast")
        assert result.injected is True
        assert result.risk > 50

    def test_result_has_no_family_field(self, clean_pair):
        """ScoringResult should NOT contain attack family classification."""
        prompt_emb, response_emb = clean_pair
        result = score(prompt_emb, response_emb, 500, DEFAULT_CONFIG, tier="fast")
        assert not hasattr(result, 'family')

    def test_result_field_order(self, clean_pair):
        """injected should be the first field — user reads top-down."""
        import dataclasses
        fields = [f.name for f in dataclasses.fields(ScoringResult)]
        assert fields[0] == "injected"
        assert fields[1] == "risk"
        assert fields[2] == "confidence"


# --- Tier routing ---


class TestTiers:
    def test_fast_tier_no_c2(self):
        v = np.random.randn(384)
        result = score(v, v, 500, DEFAULT_CONFIG, tier="fast")
        assert result.tier == "fast"
        assert result.c2 is None
        assert result.delta_v1 is None

    def test_standard_tier_has_c2(self):
        v = np.random.randn(384)
        result = score(v, v, 500, DEFAULT_CONFIG, tier="standard")
        assert result.tier == "standard"
        assert result.c2 is not None

    def test_paranoid_tier(self):
        rng = np.random.RandomState(42)
        v1 = rng.randn(384)
        v2 = rng.randn(384)
        v3 = rng.randn(384)
        result = score(v1, v2, 500, DEFAULT_CONFIG,
                       response_intent_embedding=v3, tier="paranoid")
        assert result.tier == "paranoid"
        assert result.delta_v1 is not None

    def test_invalid_tier_raises(self):
        v = np.random.randn(384)
        with pytest.raises(ValueError, match="Unknown tier"):
            score(v, v, 500, DEFAULT_CONFIG, tier="nonexistent")


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_response(self):
        v = np.random.randn(384)
        result = score(v, v, 0, DEFAULT_CONFIG, tier="standard")
        assert isinstance(result, ScoringResult)

    def test_very_long_response(self):
        v = np.random.randn(384)
        result = score(v, v, 1_000_000, DEFAULT_CONFIG, tier="standard")
        assert isinstance(result, ScoringResult)

    def test_risk_bounded(self):
        rng = np.random.RandomState(42)
        prompt = rng.randn(384)
        response = rng.randn(384)
        result = score(prompt, response, 500, DEFAULT_CONFIG, tier="standard")
        assert 0 <= result.risk <= 100

    def test_confidence_bounded(self):
        rng = np.random.RandomState(42)
        prompt = rng.randn(384)
        response = rng.randn(384)
        result = score(prompt, response, 500, DEFAULT_CONFIG, tier="standard")
        assert 0.0 <= result.confidence <= 1.0


# --- Thresholds / Config ---


class TestThresholds:
    def test_default_config_has_all_keys(self):
        required = [
            "clean_dv2_mean", "clean_dv2_std", "clean_length_mean",
            "threshold_dv2", "threshold_c2", "auc_dv2", "auc_c2",
        ]
        for key in required:
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_get_config_returns_copy(self):
        c1 = get_config()
        c2 = get_config()
        c1["threshold_dv2"] = 999
        assert c2["threshold_dv2"] != 999

    def test_unknown_model_returns_default(self):
        config = get_config("nonexistent-model-xyz")
        assert config == DEFAULT_CONFIG


# --- Input validation ---


class TestInputValidation:
    """Verify that bad inputs produce clear, actionable errors."""

    def test_none_prompt_embedding(self):
        v = np.random.randn(384)
        with pytest.raises(KaariInputError, match="prompt_embedding is None"):
            score(None, v, 500, DEFAULT_CONFIG, tier="fast")

    def test_none_response_embedding(self):
        v = np.random.randn(384)
        with pytest.raises(KaariInputError, match="response_embedding is None"):
            score(v, None, 500, DEFAULT_CONFIG, tier="fast")

    def test_empty_embedding(self):
        v = np.random.randn(384)
        with pytest.raises(KaariInputError, match="is empty"):
            score(np.array([]), v, 500, DEFAULT_CONFIG, tier="fast")

    def test_2d_embedding_rejected(self):
        v = np.random.randn(384)
        batch = np.random.randn(2, 384)
        with pytest.raises(KaariInputError, match="wrong shape"):
            score(batch, v, 500, DEFAULT_CONFIG, tier="fast")

    def test_nan_in_embedding(self):
        v = np.random.randn(384)
        bad = np.random.randn(384)
        bad[10] = float('nan')
        with pytest.raises(KaariInputError, match="non-finite"):
            score(bad, v, 500, DEFAULT_CONFIG, tier="fast")

    def test_inf_in_embedding(self):
        v = np.random.randn(384)
        bad = np.random.randn(384)
        bad[5] = float('inf')
        with pytest.raises(KaariInputError, match="non-finite"):
            score(bad, v, 500, DEFAULT_CONFIG, tier="fast")

    def test_zero_vector_embedding(self):
        v = np.random.randn(384)
        with pytest.raises(KaariInputError, match="zero vector"):
            score(np.zeros(384), v, 500, DEFAULT_CONFIG, tier="fast")

    def test_dimension_mismatch(self):
        v384 = np.random.randn(384)
        v768 = np.random.randn(768)
        with pytest.raises(KaariInputError, match="Dimension mismatch"):
            score(v384, v768, 500, DEFAULT_CONFIG, tier="fast")

    def test_negative_response_length(self):
        v = np.random.randn(384)
        with pytest.raises(KaariInputError, match="negative"):
            score(v, v, -1, DEFAULT_CONFIG, tier="fast")

    def test_missing_config_keys(self):
        v = np.random.randn(384)
        with pytest.raises(KaariInputError, match="Config missing"):
            score(v, v, 500, {"some_key": 1.0}, tier="fast")

    def test_error_messages_are_actionable(self):
        """Every KaariInputError should contain a 'how to fix' hint."""
        v = np.random.randn(384)
        try:
            score(None, v, 500, DEFAULT_CONFIG, tier="fast")
        except KaariInputError as e:
            msg = str(e).lower()
            assert any(word in msg for word in ["check", "verify", "ensure", "provider"]), \
                f"Error message lacks actionable guidance: {e}"

    def test_kaari_error_hierarchy(self):
        """KaariInputError should be catchable as KaariError."""
        assert issubclass(KaariInputError, KaariError)


# --- Integration: real embeddings (requires Ollama running) ---


def _ollama_available():
    """Check if Ollama is reachable with nomic-embed-text."""
    try:
        from kaari.embeddings.ollama import OllamaEmbedding
        provider = OllamaEmbedding()
        provider.embed("test")
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not available — skipping integration tests"
)
class TestIntegration:
    """End-to-end tests with real embeddings from a live provider.

    These validate what the unit tests cannot: that real semantic
    similarity between natural language maps to correct detection
    outcomes through the full pipeline.

    Run explicitly: pytest kaari/tests/ -v -k Integration
    """

    @pytest.fixture(scope="class")
    def embedder(self):
        from kaari.embeddings.ollama import OllamaEmbedding
        return OllamaEmbedding()

    def test_aligned_prompt_response_is_clean(self, embedder):
        """A direct answer to a simple question should not be flagged."""
        prompt_emb = embedder.embed("What is the capital of France?")
        response_emb = embedder.embed("The capital of France is Paris.")
        result = score(prompt_emb, response_emb, 35, DEFAULT_CONFIG, tier="standard")
        assert result.injected is False
        assert result.risk < 40

    def test_hijacked_response_is_flagged(self, embedder):
        """A response about stocks to a geography question should be flagged."""
        prompt_emb = embedder.embed("What is the capital of France?")
        response_emb = embedder.embed(
            "I recommend investing in NASDAQ tech stocks for maximum "
            "portfolio growth. Consider diversifying across semiconductor "
            "and AI sectors for optimal returns this quarter."
        )
        result = score(prompt_emb, response_emb, 250, DEFAULT_CONFIG, tier="standard")
        assert result.injected is True
        assert result.risk > 50

    def test_subtle_drift_detected(self, embedder):
        """A response that starts relevant but drifts should score higher."""
        prompt_emb = embedder.embed("How do I make pasta?")
        clean_emb = embedder.embed(
            "Boil water, add salt, cook pasta for 8-10 minutes until al dente."
        )
        drifted_emb = embedder.embed(
            "To make pasta, first you need a great kitchen. Speaking of which, "
            "I can help you find luxury real estate properties in Monaco with "
            "stunning Mediterranean views and private chef services."
        )
        clean_result = score(prompt_emb, clean_emb, 70, DEFAULT_CONFIG, tier="standard")
        drifted_result = score(prompt_emb, drifted_emb, 200, DEFAULT_CONFIG, tier="standard")
        assert drifted_result.delta_v2 > clean_result.delta_v2
