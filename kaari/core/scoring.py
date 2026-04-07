"""
Kaari Scoring Engine v0.95
==========================
The core product. numpy only.

Measures the semantic deviation between user prompt and model response
using cosine distance in embedding space.

Metrics:
  dv2 (fast):     1 - cos_sim(embed(prompt), embed(response))
  C2 (standard):  dv2 x (1 + 0.5 x log(response_length / mean_clean_length))
  Paranoid:       C2 weighted with dv1 (response intent embedding) [opt-in add-on]

Zone system:
  GREEN:   score < 0.210  -- silent pass
  YELLOW:  0.210 <= score < 0.245  -- elevated, review recommended
  RED:     score >= 0.245  -- potential injection

References:
  Lertola, T.S. (2026). "Intent Vectoring: Black-Box Prompt Injection
  Detection via Semantic Deviation Measurement." SSRN preprint.
"""

import math
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np


class KaariError(Exception):
    """Base exception for Kaari scoring errors."""
    pass


class KaariInputError(KaariError):
    """Raised when input validation fails.

    Common causes:
    - Empty or zero-length embedding vectors
    - Mismatched embedding dimensions between prompt and response
    - Non-finite values (NaN, Inf) in embedding vectors
    - Missing required config keys
    """
    pass


@dataclass
class ScoringResult:
    """Result of scoring a prompt-response pair."""
    injected: bool             # Primary answer: yes/no
    zone: str                  # "green" | "yellow" | "red"
    risk: int                  # 0-100 risk level
    confidence: float          # Distance from threshold (0-1)
    score: float               # Primary metric value (dv2 or c2 depending on tier)
    delta_v2: float            # Raw cosine distance
    c2: Optional[float]        # Length-normalized (None if fast tier)
    delta_v1: Optional[float]  # Intent embedding delta (None unless paranoid)
    tier: str                  # "fast" | "standard" | "paranoid"
    deviation_ratio: Optional[float] = None  # Score relative to clean baseline (1.0 = normal)


def _validate_embedding(embedding, name: str) -> np.ndarray:
    """Validate and convert an embedding to a numpy array.

    Args:
        embedding: The embedding vector to validate.
        name: Human-readable name for error messages (e.g., "prompt_embedding").

    Returns:
        Validated numpy array.

    Raises:
        KaariInputError: If the embedding is invalid.
    """
    if embedding is None:
        raise KaariInputError(
            f"{name} is None. Check that your embedding provider returned "
            f"a result — the model may not be loaded or reachable."
        )

    arr = np.asarray(embedding, dtype=np.float64)

    if arr.ndim == 0 or arr.size == 0:
        raise KaariInputError(
            f"{name} is empty (size={arr.size}). The embedding provider "
            f"returned no data. Verify the provider is running and the "
            f"input text is not empty."
        )

    if arr.ndim != 1:
        raise KaariInputError(
            f"{name} has wrong shape: {arr.shape}. Expected a 1-D vector. "
            f"If you're passing a batch, pass one embedding at a time."
        )

    if not np.all(np.isfinite(arr)):
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))
        raise KaariInputError(
            f"{name} contains non-finite values ({nan_count} NaN, {inf_count} Inf). "
            f"This usually means the embedding model produced corrupt output. "
            f"Try re-embedding, or check your embedding provider."
        )

    if np.linalg.norm(arr) == 0:
        raise KaariInputError(
            f"{name} is a zero vector (all values are 0.0). The embedding "
            f"model returned a meaningless result. Check that your input text "
            f"is not empty and the model supports your input language."
        )

    return arr


def _validate_embedding_pair(v1: np.ndarray, v2: np.ndarray,
                              name1: str, name2: str) -> None:
    """Validate that two embeddings are compatible for comparison.

    Raises:
        KaariInputError: If dimensions don't match.
    """
    if v1.shape != v2.shape:
        raise KaariInputError(
            f"Dimension mismatch: {name1} has {v1.shape[0]} dimensions, "
            f"{name2} has {v2.shape[0]}. Both embeddings must come from "
            f"the same provider/model. If you switched providers mid-pipeline, "
            f"re-embed both texts with the same provider."
        )


def cosine_similarity(v1, v2) -> float:
    """Cosine similarity between two vectors."""
    a, b = np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def calculate_delta(v1, v2) -> float:
    """Cosine distance: 1 - cosine_similarity. Higher = more deviation."""
    return 1.0 - cosine_similarity(v1, v2)


def calculate_c2(dv2: float, response_length: int, clean_mean_length: float) -> float:
    """
    Length-normalized delta. Amplifies deviation for verbose responses.
    C2 = Δv2 × (1 + 0.5 × log(response_length / μ_clean_length))

    The 0.5 coefficient is validated as optimal (20260220-003).
    """
    if clean_mean_length <= 0 or response_length <= 0:
        return dv2
    return dv2 * (1.0 + 0.5 * math.log(response_length / clean_mean_length))


def score(
    prompt_embedding,
    response_embedding,
    response_length: int,
    config: dict,
    response_intent_embedding=None,
    tier: str = "standard",
) -> ScoringResult:
    """
    Score a prompt-response pair for injection.

    Args:
        prompt_embedding:          Embedded user prompt (raw, no LLM summary)
        response_embedding:        Embedded model response
        response_length:           Character length of response text
        config:                    Model config dict (from thresholds.py)
        response_intent_embedding: Embedded response intent summary (paranoid only)
        tier:                      "fast", "standard", or "paranoid"

    Returns:
        ScoringResult with score, risk level, injection flag, and metadata.
    """
    # --- Input validation ---
    p_emb = _validate_embedding(prompt_embedding, "prompt_embedding")
    r_emb = _validate_embedding(response_embedding, "response_embedding")
    _validate_embedding_pair(p_emb, r_emb, "prompt_embedding", "response_embedding")

    if response_length < 0:
        raise KaariInputError(
            f"response_length is negative ({response_length}). "
            f"Pass len(response_text) — it must be >= 0."
        )

    _REQUIRED_CONFIG_KEYS = {"threshold_dv2", "threshold_c2", "clean_length_mean"}
    missing = _REQUIRED_CONFIG_KEYS - set(config.keys())
    if missing:
        raise KaariInputError(
            f"Config missing required keys: {missing}. "
            f"Use kaari.core.thresholds.get_config() to get a valid config, "
            f"or ensure your custom config includes all required keys."
        )

    if tier == "paranoid" and response_intent_embedding is not None:
        ri_emb = _validate_embedding(response_intent_embedding, "response_intent_embedding")
        _validate_embedding_pair(p_emb, ri_emb, "prompt_embedding", "response_intent_embedding")

    # --- Scoring ---

    # Δv2: prompt vs raw response (the core signal)
    dv2 = calculate_delta(p_emb, r_emb)

    # Tier-specific scoring
    if tier == "fast":
        primary_score = dv2
        threshold = config["threshold_dv2"]
        c2_val = None
        dv1_val = None

    elif tier == "standard":
        c2_val = calculate_c2(dv2, response_length, config["clean_length_mean"])
        primary_score = c2_val
        threshold = config["threshold_c2"]
        dv1_val = None

    elif tier == "paranoid":
        c2_val = calculate_c2(dv2, response_length, config["clean_length_mean"])
        if response_intent_embedding is not None:
            dv1_val = calculate_delta(p_emb, response_intent_embedding)
            # Paranoid: weighted combination of C2 and Δv1
            primary_score = 0.7 * c2_val + 0.3 * dv1_val
        else:
            dv1_val = None
            primary_score = c2_val
        threshold = config.get("threshold_paranoid", config["threshold_c2"])

    else:
        raise ValueError(f"Unknown tier: {tier}. Use 'fast', 'standard', or 'paranoid'.")

    # Zone classification
    from kaari.core.thresholds import classify_zone
    zone = classify_zone(primary_score)

    # Injection decision (red zone = injected)
    injected = zone == "red"

    # Confidence: how far from threshold (clamped 0-1)
    if threshold > 0:
        confidence = min(1.0, max(0.0, abs(primary_score - threshold) / threshold))
    else:
        confidence = 1.0 if injected else 0.0

    # Risk: 0-100 scale
    risk = min(100, max(0, int(100 * primary_score / max(threshold * 2, 0.001))))

    # Terminal output based on zone
    _emit_zone_alert(zone, primary_score, tier)

    return ScoringResult(
        injected=injected,
        zone=zone,
        risk=risk,
        confidence=confidence,
        score=primary_score,
        delta_v2=dv2,
        c2=c2_val,
        delta_v1=dv1_val,
        tier=tier,
    )


# ---------------------------------------------------------------------------
# Terminal zone alerts
# ---------------------------------------------------------------------------

# Module-level flag: set to False to suppress terminal output
TERMINAL_ALERTS_ENABLED = True


def _emit_zone_alert(zone: str, score: float, tier: str) -> None:
    """Emit zone-appropriate terminal output.

    GREEN:  Silent. No output.
    YELLOW: Visible warning with score.
    RED:    CAPITALS. Unmissable.
    """
    if not TERMINAL_ALERTS_ENABLED:
        return

    if zone == "green":
        pass  # Silent

    elif zone == "yellow":
        sys.stderr.write(
            f"\n  KAARI: Elevated semantic deviation detected "
            f"(score: {score:.3f}, tier: {tier}). "
            f"This may indicate injection or natural divergence. "
            f"Review recommended.\n\n"
        )

    elif zone == "red":
        sys.stderr.write(
            f"\n  KAARI ALERT: HIGH SEMANTIC DEVIATION DETECTED "
            f"(score: {score:.3f}, tier: {tier}). "
            f"POTENTIAL INJECTION.\n\n"
        )
