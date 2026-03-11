"""
Kaari — Intent Vectoring for Prompt Injection Detection v0,21
========================================================
Black-box prompt injection detection via semantic deviation measurement.

Quick start:
    import kaari

    # Initialize with local Ollama (free)
    k = kaari.Kaari()

    # Score a prompt-response pair
    result = k.score("What is 2+2?", "The answer is 4.")
    print(result.injected)  # False
    print(result.risk)      # Low number

    # Use as decorator
    @k.guard
    def my_llm_call(prompt):
        return call_my_model(prompt)

References:
    Lertola, T.S. (2026). "Intent Vectoring: Black-Box Prompt Injection
    Detection via Semantic Deviation Measurement."
"""

__version__ = "0.2.1"

from kaari.client import Kaari, InjectionDetected
from kaari.core.scoring import KaariError, KaariInputError

__all__ = ["Kaari", "InjectionDetected", "KaariError", "KaariInputError", "__version__"]
