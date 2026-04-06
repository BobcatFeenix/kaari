"""
Kaari v0.97 — Intent Vectoring for Prompt Injection Detection
=============================================================
Black-box prompt injection detection via semantic deviation measurement.

Quick start:
    import kaari

    k = kaari.Kaari()
    # -> "KAARI v0.97 online. Detection active (standard tier, threshold 0.245).
    #     GREEN/YELLOW/RED zone alerts will appear here."

    result = k.score("What is 2+2?", "The answer is 4.")
    print(result.zone)      # "green"
    print(result.injected)  # False

    k.pause()    # Silence detection
    k.resume()   # Reactivate
    k.status()   # Print config and state
    k.report()   # Print scan counts (when reporting=True)

Zone system:
    GREEN:   score < 0.210  — silent pass
    YELLOW:  0.210-0.245    — elevated, review recommended
    RED:     score >= 0.245  — potential injection

Try the demo:
    python -m kaari.test_cases.run_demo

References:
    Lertola, T.S. (2026). "Intent Vectoring: Black-Box Prompt Injection
    Detection via Semantic Deviation Measurement."
"""

__version__ = "0.97.0"

from kaari.client import Kaari, InjectionDetected
from kaari.core.scoring import KaariError, KaariInputError

__all__ = ["Kaari", "InjectionDetected", "KaariError", "KaariInputError", "__version__"]
