# Kaari

**Black-box prompt injection detection for AI agent pipelines via semantic deviation measurement.**

[![Paper](https://img.shields.io/badge/Paper-SSRN%20Preprint-blue)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6280858)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Beta%202026-orange)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()

Kaari detects when an AI agent has been redirected from what the user 
actually asked for. No access to model internals required. Works with 
any text-in / text-out API. Built on **Intent Vectoring** — measuring cosine distance between 
embedded user intent and model response in vector space. If the model 
went somewhere the user didn't send it, Kaari sees it.

---
Kaari scores prompt-response pairs by measuring how far a model's response deviates from the user's original intent in embedding space. No access to model internals needed — works with any LLM.

Based on the Intent Vectoring research (N=1,944, AUC 0.883).

## Quick Start

```bash
pip install kaari
```

Requires [Ollama](https://ollama.com) running locally with `nomic-embed-text`:

```bash
ollama pull nomic-embed-text
```

### Score a response

```python
import kaari

k = kaari.Kaari()

result = k.score(
    prompt="What is 2+2?",
    response="The answer is 4."
)

print(result.injected)   # False
print(result.risk)       # 8
print(result.delta_v2)   # 0.0812
```

### Guard an LLM call

```python
@k.guard
def ask(prompt):
    # Your LLM call here
    return my_model.generate(prompt)

response = ask("Explain quantum computing")
# Kaari scores the response automatically.
# Logs a warning if injection is detected.
```

### Raise on injection

```python
k = kaari.Kaari(on_inject="raise")

try:
    result = k.score(prompt, suspicious_response)
except kaari.InjectionDetected as e:
    print(f"Blocked: risk={e.result.risk}, score={e.result.score:.4f}")
```

## How It Works

Kaari embeds the user's prompt and the model's response into the same vector space, then measures their cosine distance.

A clean response stays close to the prompt's intent. An injected response deviates — the model talks about something the user never asked for.

```
prompt  ──→  embed(prompt)    ─┐
                                ├──→  Δv2 = 1 - cos_sim  ──→  score
response ──→  embed(response) ─┘
```

The **C2 metric** normalizes for response verbosity:

```
C2 = Δv2 × (1 + 0.5 × log(response_length / μ_clean_length))
```

### Detection Tiers

| Tier | Metric | AUC | Embed Calls | Use Case |
|------|--------|-----|-------------|----------|
| `fast` | Δv2 only | 0.870 | 2 | High-throughput, latency-sensitive |
| `standard` | C2 | 0.883 | 2 | Default. Best accuracy/cost balance |
| `paranoid` | C2 + Δv1 | ~0.89+ | 3 | High-security applications |

```python
# Choose tier
result = k.score(prompt, response, tier="fast")
result = k.score(prompt, response, tier="paranoid")
```

## Embedding Providers

Kaari works with any embedding model. Swap providers without changing your scoring code.

### Ollama (default, free, local)

```python
k = kaari.Kaari()  # Uses Ollama + nomic-embed-text by default
```

### OpenAI (paid, server-side)

```python
from kaari.embeddings import OpenAIEmbedding

k = kaari.Kaari(
    embedding=OpenAIEmbedding(api_key="sk-...")
)
```

### Custom provider

```python
from kaari.embeddings.base import EmbeddingProvider

class MyEmbedding(EmbeddingProvider):
    def embed(self, text: str) -> np.ndarray:
        return my_embed_function(text)

    @property
    def dimension(self) -> int:
        return 768

    @property
    def name(self) -> str:
        return "my-embedding"

k = kaari.Kaari(embedding=MyEmbedding())
```

## Integration

Kaari sits between your LLM call and your user. Score the response *before* showing it.

### Where it goes in your pipeline

```
User prompt → Your LLM → response text → Kaari.score() → decision → show or block
```

### FastAPI middleware pattern

```python
import kaari

k = kaari.Kaari(tier="standard")

@app.post("/chat")
async def chat(prompt: str):
    response = await my_llm(prompt)
    result = k.score(prompt, response)

    if result.injected:
        # Option A: Block and return safe fallback
        return {"text": "I couldn't generate a safe response.", "blocked": True}
        # Option B: Return with warning flag
        # return {"text": response, "warning": f"risk={result.risk}"}

    return {"text": response}
```

### Deciding what to do with the result

```python
result = k.score(prompt, response)

if not result.injected:
    # Clean — show to user
    pass
elif result.risk < 50:
    # Low-confidence detection — log and show, or show with warning
    logger.info(f"Marginal detection: risk={result.risk}, dv2={result.delta_v2:.3f}")
elif result.risk >= 50:
    # High-confidence detection — block or escalate
    logger.warning(f"Injection blocked: risk={result.risk}, score={result.score:.4f}")
```

### Error handling

Kaari raises `KaariInputError` for bad inputs and `KaariError` for provider failures. Catch these so your pipeline doesn't crash when the detector has a problem:

```python
from kaari import KaariError, KaariInputError

try:
    result = k.score(prompt, response)
except KaariInputError as e:
    # Bad input — log, skip scoring, show response anyway
    logger.error(f"Kaari input error: {e}")
    return {"text": response, "scored": False}
except KaariError as e:
    # Provider down — degrade gracefully
    logger.error(f"Kaari unavailable: {e}")
    return {"text": response, "scored": False}
```

A security tool that crashes your pipeline is worse than no security tool. Always have a fallback path.

## What It Detects

Kaari detects semantic drift between a user's prompt and the model's response — regardless of what caused it. If the response talks about something the user never asked for, Kaari flags it.

The detection is **family-agnostic**: it doesn't classify *what kind* of injection occurred, just *whether* the response drifted. This makes it robust to novel attack types.

```python
result = k.score(prompt, response)
if result.injected:
    print(f"Risk: {result.risk}, Score: {result.score:.4f}")
```

## Limitations

Kaari is a **post-hoc monitoring tool**, not a firewall. It scores responses after they're generated. For real-time blocking, check the response before showing it to the user.

Known limitations:

- **Subtle injections** are harder to detect (AUC drops from 0.92 for obvious to 0.67 for file-based)
- **Single embedding model** validated so far (nomic-embed-text). Other models may need recalibration
- **Simple test prompts** in research. Long-document and multi-turn scenarios need further validation

See [SECURITY.md](SECURITY.md) for responsible use guidance.

## Research

This implementation is based on:

> Lertola, T.S. (2026). "Intent Vectoring: Black-Box Prompt Injection Detection via Semantic Deviation Measurement." Sol Lucid Labs. *arXiv preprint.*

Key findings from the paper (N=1,944 across 4 models):

- Combined Cohen's d = 1.72 (large effect size)
- AUC-ROC = 0.870 (Δv2), 0.883 (C2 with length normalization)
- Cross-model generalization confirmed

## Contributing

Contributions welcome. The core scoring engine is intentionally minimal (~140 lines). If you're adding features, consider whether they belong in core or in a pipeline-specific module.

```bash
git clone https://github.com/BobcatFeenix/kaari.git
cd kaari
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT. See [LICENSE](LICENSE).

---

Built by [Sol Lucid Labs](https://BobcatFeenix.com) in Helsinki.
