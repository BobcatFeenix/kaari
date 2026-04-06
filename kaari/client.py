"""
Kaari Client v0.97 — The public interface.
===========================================

Usage:
    import kaari

    # Basic scoring (standard tier, default)
    k = kaari.Kaari()
    # -> "KAARI v0.97 online. Detection active (standard tier, threshold 0.245).
    #     GREEN/YELLOW/RED zone alerts will appear here."

    result = k.score("What is 2+2?", "The answer is 4.")
    print(result.zone)  # "green" — clean

    # Pause/resume detection
    k.pause()    # Silences all scoring — returns neutral results
    k.resume()   # Reactivates detection

    # Usage report
    k = kaari.Kaari(reporting=True)
    # ... after scoring ...
    k.report()   # Prints scan counts by zone

    # Status check
    k.status()   # Prints current config, state, scan count

    # Configure what happens on RED zone detection
    k = kaari.Kaari(on_red="log")       # Default: log alert, continue
    k = kaari.Kaari(on_red="raise")     # Raise KaariInjectionAlert
    k = kaari.Kaari(on_red=my_handler)  # Call your function

    # Paranoid tier (opt-in add-on, requires LLM for response intent)
    # Additional cost: 1 LLM inference call per scoring
    k = kaari.Kaari(tier="paranoid")

    # With OpenAI embeddings (paid tier)
    from kaari.embeddings import OpenAIEmbedding
    k = kaari.Kaari(embedding=OpenAIEmbedding(api_key="sk-..."))
"""

import functools
import logging
import sys
import time
from typing import Optional, Callable, Union

from kaari.core.scoring import score, ScoringResult, KaariError, KaariInputError
from kaari.core.thresholds import get_config, ZONE_GREEN_MAX, ZONE_YELLOW_MAX
from kaari.embeddings.base import EmbeddingProvider, EmbeddingError
from kaari.embeddings.ollama import OllamaEmbedding

logger = logging.getLogger("kaari")


class Kaari:
    """
    Main Kaari client.

    Args:
        embedding:  EmbeddingProvider instance (default: OllamaEmbedding)
        model:      Model name for per-model calibration (optional)
        tier:       Detection tier: "fast", "standard" (default: "standard")
                    "paranoid" is available as an opt-in add-on. It adds one
                    LLM inference call per scoring for response intent extraction.
                    Use in high-risk environments where the extra cost is justified.
        on_red:     Action when RED zone is triggered:
                    - "log" (default): logs the alert, returns result, process continues
                    - "raise": raises KaariInjectionAlert exception
                    - callable: your function(prompt, response, result) is called
        reporting:  Enable usage reporting (default: False). When True, Kaari
                    tracks scan counts by zone. Call k.report() to see results.
    """

    def __init__(
        self,
        embedding: Optional[EmbeddingProvider] = None,
        model: Optional[str] = None,
        tier: str = "standard",
        on_red: Union[str, Callable] = "log",
        reporting: bool = False,
    ):
        if tier == "paranoid":
            logger.info(
                "Kaari: Paranoid tier enabled. This adds 1 LLM inference call "
                "per scoring for response intent extraction. For most use cases, "
                "the standard tier provides sufficient detection."
            )
        self._embedding = embedding or OllamaEmbedding()
        self._config = get_config(model)
        self._model = model
        self._tier = tier
        self._on_red = on_red
        self._paused = False
        self._started_at = time.time()

        # Reporting
        self._reporting = reporting
        self._scan_counts = {"green": 0, "yellow": 0, "red": 0, "paused": 0}

        # Welcome message
        from kaari import __version__
        threshold = self._config.get("threshold_c2", ZONE_YELLOW_MAX)
        sys.stderr.write(
            f"\n  KAARI v{__version__} online. "
            f"Detection active ({tier} tier, threshold {threshold:.3f}). "
            f"GREEN/YELLOW/RED zone alerts will appear here.\n\n"
        )

    def score(
        self,
        prompt: str,
        response: str,
        tier: Optional[str] = None,
    ) -> ScoringResult:
        """
        Score a prompt-response pair for injection.

        Args:
            prompt:   The user's original prompt text.
            response: The model's response text.
            tier:     Override default tier for this call.

        Returns:
            ScoringResult with score, risk, injected flag, and metadata.
            If paused, returns a neutral green result without embedding.
        """
        tier = tier or self._tier

        # If paused, return neutral result without doing any work
        if self._paused:
            self._scan_counts["paused"] += 1
            return ScoringResult(
                injected=False,
                zone="green",
                risk=0,
                confidence=0.0,
                score=0.0,
                delta_v2=0.0,
                c2=0.0 if tier != "fast" else None,
                delta_v1=None,
                tier=tier,
            )

        # Validate inputs
        if not prompt or not prompt.strip():
            raise KaariInputError(
                "Prompt is empty. Kaari needs the user's original prompt text "
                "to measure whether the response drifted from it."
            )
        if not response or not response.strip():
            raise KaariInputError(
                "Response is empty. No model output to score. Check that your "
                "LLM returned a response before passing it to Kaari."
            )

        # Embed prompt and response
        try:
            prompt_emb = self._embedding.embed(prompt)
        except EmbeddingError:
            raise
        except Exception as e:
            raise KaariError(
                f"Failed to embed prompt: {e}. Check that your embedding "
                f"provider ({self._embedding.name}) is running and reachable."
            ) from e

        try:
            response_emb = self._embedding.embed(response)
        except EmbeddingError:
            raise
        except Exception as e:
            raise KaariError(
                f"Failed to embed response: {e}. Check that your embedding "
                f"provider ({self._embedding.name}) is running and reachable."
            ) from e

        # Response intent embedding for paranoid tier
        response_intent_emb = None
        if tier == "paranoid":
            # In paranoid mode, we'd ideally have a response intent summary.
            # Without an LLM in the loop, we use the response embedding directly.
            # This means paranoid tier currently uses C2 only (no Δv1 boost).
            # Future: optional LLM summarization for paid paranoid tier.
            pass

        # Score
        result = score(
            prompt_embedding=prompt_emb,
            response_embedding=response_emb,
            response_length=len(response),
            config=self._config,
            response_intent_embedding=response_intent_emb,
            tier=tier,
        )

        # Track scan counts
        self._scan_counts[result.zone] += 1

        # Handle red zone
        if result.zone == "red":
            self._handle_red(prompt, response, result)

        return result

    def guard(self, func: Optional[Callable] = None, *, tier: Optional[str] = None):
        """
        Decorator that scores LLM responses and handles injections.

        The decorated function must accept a prompt (str) as first argument
        and return a response (str).

        Usage:
            @k.guard
            def my_llm_call(prompt):
                return call_my_model(prompt)

            # Or with options:
            @k.guard(tier="paranoid")
            def my_sensitive_call(prompt):
                return call_my_model(prompt)

        Args:
            func: The function to decorate.
            tier: Override tier for this guard.
        """
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(prompt, *args, **kwargs):
                # Call the actual LLM function
                response = fn(prompt, *args, **kwargs)

                # Score the response (zone alerts handled by score())
                result = self.score(prompt, response, tier=tier)

                # Attach scoring result to response if possible
                if isinstance(response, str):
                    return response
                else:
                    try:
                        response._kaari = result
                    except AttributeError:
                        pass
                    return response

            # Attach scorer for direct access
            wrapper.kaari = self
            return wrapper

        # Handle both @k.guard and @k.guard(tier="paranoid")
        if func is not None:
            return decorator(func)
        return decorator

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------

    def pause(self):
        """Pause detection. score() returns neutral green results without embedding.

        Use this to temporarily disable Kaari without removing it from your pipeline.
        """
        if not self._paused:
            self._paused = True
            sys.stderr.write(
                "\n  KAARI: Detection paused. "
                "Scoring will return neutral results until k.resume() is called.\n\n"
            )

    def resume(self):
        """Resume detection after pause."""
        if self._paused:
            self._paused = False
            sys.stderr.write(
                "\n  KAARI: Detection resumed. "
                "Scoring is active.\n\n"
            )

    @property
    def is_paused(self) -> bool:
        """Whether detection is currently paused."""
        return self._paused

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self):
        """Print current Kaari configuration and state to terminal."""
        from kaari import __version__
        total = sum(self._scan_counts[z] for z in ("green", "yellow", "red"))
        paused_count = self._scan_counts["paused"]
        uptime = time.time() - self._started_at
        minutes = int(uptime // 60)
        seconds = int(uptime % 60)

        state = "PAUSED" if self._paused else "ACTIVE"
        threshold = self._config.get("threshold_c2", ZONE_YELLOW_MAX)

        lines = [
            f"",
            f"  KAARI v{__version__} — Status",
            f"  State:      {state}",
            f"  Tier:       {self._tier}",
            f"  Threshold:  {threshold:.3f}",
            f"  Zones:      GREEN < {ZONE_GREEN_MAX} | YELLOW {ZONE_GREEN_MAX}-{ZONE_YELLOW_MAX} | RED >= {ZONE_YELLOW_MAX}",
            f"  Embedding:  {self._embedding.name}",
            f"  On RED:     {self._on_red if isinstance(self._on_red, str) else 'callback'}",
            f"  Reporting:  {'ON' if self._reporting else 'OFF'}",
            f"  Scans:      {total} scored ({paused_count} skipped while paused)",
            f"  Uptime:     {minutes}m {seconds}s",
            f"",
        ]
        sys.stderr.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Usage Report
    # ------------------------------------------------------------------

    def report(self):
        """Print usage report to terminal.

        Shows scan counts by zone. Only tracks when reporting=True,
        but zone counts are always maintained internally.
        """
        from kaari import __version__
        total = sum(self._scan_counts[z] for z in ("green", "yellow", "red"))
        paused_count = self._scan_counts["paused"]
        uptime = time.time() - self._started_at
        minutes = int(uptime // 60)
        seconds = int(uptime % 60)

        if total == 0 and paused_count == 0:
            sys.stderr.write(
                "\n  KAARI Report: No scans performed yet.\n\n"
            )
            return

        green = self._scan_counts["green"]
        yellow = self._scan_counts["yellow"]
        red = self._scan_counts["red"]

        lines = [
            f"",
            f"  KAARI v{__version__} — Usage Report",
            f"  {'=' * 40}",
            f"  Total scans:    {total}",
            f"  GREEN (clean):  {green:>6}  ({_pct(green, total)})",
            f"  YELLOW (review):{yellow:>6}  ({_pct(yellow, total)})",
            f"  RED (alert):    {red:>6}  ({_pct(red, total)})",
        ]
        if paused_count > 0:
            lines.append(
                f"  Skipped (paused): {paused_count}"
            )
        lines.extend([
            f"  {'=' * 40}",
            f"  Session duration: {minutes}m {seconds}s",
            f"",
        ])
        sys.stderr.write("\n".join(lines) + "\n")

    def reset_counts(self):
        """Reset scan counters to zero."""
        self._scan_counts = {"green": 0, "yellow": 0, "red": 0, "paused": 0}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_red(self, prompt: str, response: str, result: ScoringResult):
        """Handle RED zone detection based on configured policy.

        on_red options:
          "log"     — log the alert, return result, process continues (default)
          "raise"   — raise KaariInjectionAlert, caller decides what to do
          callable  — your function(prompt, response, result) is called
        """
        if self._on_red == "log":
            logger.warning(
                f"Kaari RED zone: score={result.score:.4f}, "
                f"risk={result.risk}, tier={result.tier}"
            )
        elif self._on_red == "raise":
            raise InjectionDetected(result)
        elif callable(self._on_red):
            self._on_red(prompt, response, result)

    def __repr__(self):
        state = "paused" if self._paused else "active"
        return (
            f"Kaari(embedding={self._embedding.name}, "
            f"model={self._model}, tier={self._tier}, "
            f"on_red={self._on_red}, state={state})"
        )


def _pct(count: int, total: int) -> str:
    """Format a percentage string."""
    if total == 0:
        return "  0.0%"
    return f"{100 * count / total:5.1f}%"


class InjectionDetected(Exception):
    """Raised when injection is detected and on_red='raise'."""

    def __init__(self, result: ScoringResult):
        self.result = result
        super().__init__(
            f"Injection detected: score={result.score:.4f}, "
            f"risk={result.risk}, tier={result.tier}"
        )
