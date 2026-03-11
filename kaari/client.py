"""
Kaari Client — The public interface.
=====================================
This is what users interact with after `pip install kaari`.

Usage:
    import kaari

    # Basic scoring
    k = kaari.Kaari()
    result = k.score("What is 2+2?", "The answer is 4.")

    # With specific model calibration
    k = kaari.Kaari(model="mistral-7b")
    result = k.score(prompt, response)

    # As a decorator/guard
    @k.guard
    def my_llm_call(prompt):
        return call_my_model(prompt)

    # With OpenAI embeddings (paid tier)
    from kaari.embeddings import OpenAIEmbedding
    k = kaari.Kaari(embedding=OpenAIEmbedding(api_key="sk-..."))
"""

import functools
import logging
from typing import Optional, Callable

from kaari.core.scoring import score, ScoringResult, KaariError, KaariInputError
from kaari.core.thresholds import get_config
from kaari.embeddings.base import EmbeddingProvider, EmbeddingError
from kaari.embeddings.ollama import OllamaEmbedding

logger = logging.getLogger("kaari")


class Kaari:
    """
    Main Kaari client.

    Args:
        embedding:  EmbeddingProvider instance (default: OllamaEmbedding)
        model:      Model name for per-model calibration (optional)
        tier:       Detection tier: "fast", "standard", "paranoid" (default: "standard")
        on_inject:  Callback when injection detected: "warn", "raise", or callable
    """

    def __init__(
        self,
        embedding: Optional[EmbeddingProvider] = None,
        model: Optional[str] = None,
        tier: str = "standard",
        on_inject: str = "warn",
    ):
        self._embedding = embedding or OllamaEmbedding()
        self._config = get_config(model)
        self._model = model
        self._tier = tier
        self._on_inject = on_inject

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
        """
        tier = tier or self._tier

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

                # Score the response
                result = self.score(prompt, response, tier=tier)

                if result.injected:
                    self._handle_injection(prompt, response, result)

                # Attach scoring result to response if possible
                if isinstance(response, str):
                    # Can't attach to str, log instead
                    if result.injected:
                        logger.warning(
                            f"Kaari: injection detected (score={result.score:.4f}, "
                            f"risk={result.risk}, tier={result.tier})"
                        )
                    return response
                else:
                    # If response is an object, try to attach
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

    def _handle_injection(self, prompt: str, response: str, result: ScoringResult):
        """Handle detected injection based on configured policy."""
        if self._on_inject == "warn":
            logger.warning(
                f"Kaari injection detected: score={result.score:.4f}, "
                f"risk={result.risk}, tier={result.tier}"
            )
        elif self._on_inject == "raise":
            raise InjectionDetected(result)
        elif callable(self._on_inject):
            self._on_inject(prompt, response, result)

    def __repr__(self):
        return (
            f"Kaari(embedding={self._embedding.name}, "
            f"model={self._model}, tier={self._tier})"
        )


class InjectionDetected(Exception):
    """Raised when injection is detected and on_inject='raise'."""

    def __init__(self, result: ScoringResult):
        self.result = result
        super().__init__(
            f"Injection detected: score={result.score:.4f}, "
            f"risk={result.risk}, tier={result.tier}"
        )
