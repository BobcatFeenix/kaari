"""Kaari Core — Intent Vectoring scoring engine."""
from kaari.core.scoring import (
    score, calculate_delta, calculate_c2, cosine_similarity,
    KaariError, KaariInputError, ScoringResult,
)
from kaari.core.thresholds import get_config, get_model_config, DEFAULT_CONFIG

__all__ = [
    "score",
    "calculate_delta",
    "calculate_c2",
    "cosine_similarity",
    "KaariError",
    "KaariInputError",
    "ScoringResult",
    "get_config",
    "get_model_config",
    "DEFAULT_CONFIG",
]
