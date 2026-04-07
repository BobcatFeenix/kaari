# Kaari v0.98

**Black-box prompt injection detection for AI agent pipelines via semantic deviation measurement.**

[![Paper](https://img.shields.io/badge/Paper-SSRN%20Preprint-blue)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6280858)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-v0.98-orange)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()

Kaari detects when an AI agent has been redirected from what the user actually asked for. No access to model internals required. Works with any text-in / text-out API.

Built on **Intent Vectoring** — measuring cosine distance between embedded user intent and model response in vector space. If the model went somewhere the user didn't send it, Kaari sees it.

Based on the Intent Vectoring research (N=2,228, C2 AUC 0.822). Validated across 4 LLM architectures and 3 embedding models — detection is encoder-independent (AUC spread +/-0.006).

---

## Zone System

Kaari uses a three-zone traffic light system instead of a binary yes/no:

| Zone | C2 Score | What It Means | Terminal Output |
|------|----------|---------------|-----------------|
| **GREEN** | < 0.210 | Clean. No deviation detected. | Silent — no output |
| **YELLOW** | 0.210 - 0.245 | Elevated deviation. May be injection or natural divergence. | Warning with score |
| **RED** | >= 0.245 | High deviation. Potential injection. | ALERT in capitals |

Kaari never interrupts your process by default. You configure what happens on RED.

## How Kaari Fits In Your Pipeline

Kaari analyses the model's response **after it has been generated**, before you show it to the user. It does not filter or block the prompt on the way in — it checks the response on the way out. Think of it as a quality inspector at the end of the production line, not a guard at the door.

```
User prompt -> Your LLM generates response -> Kaari scores the response -> You decide what to show
```

This means the LLM still processes the potentially injected prompt. Kaari catches the damage in the response before it reaches your user. For input-level defense (blocking malicious prompts before they reach the model), combine Kaari with prompt sanitization or input filtering.

We are actively developing earlier-stage detection capabilities for future releases.

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

print(result.zone)       # "green"
print(result.injected)   # False
print(result.risk)       # 8
print(result.delta_v2)   # 0.0812
```

### Try the demo

Kaari ships with three test documents that demonstrate each zone. Run the demo to see what GREEN, YELLOW, and RED look like in your terminal:

```bash
python -m kaari.test_cases.run_demo
```

The test documents contain the same base text (a project status update). Two of them have hidden injection content — text that would be white-on-white in a real document, invisible to the human reader but processed by the LLM. Kaari detects the deviation in the response.

### Calibrate from your data (recommended)

Default thresholds are calibrated on research data. Your pipeline will have different clean-score distributions depending on prompt style, response length, and embedding model. Calibrating from your own data gives Kaari a baseline that matches your specific setup.

Pass 10-50 known-clean (prompt, response) pairs:

```python
k = kaari.Kaari()

k.calibrate([
    ("Summarize this report.\n\n<report text>", "The report shows quarterly growth..."),
    ("What are the key risks?\n\n<report text>", "Three risks were identified..."),
    ("List action items.\n\n<report text>", "The team needs to complete..."),
    # ... 10-50 representative clean pairs from your pipeline
])

# Now k.score() uses thresholds derived from YOUR data.
# Zone boundaries = clean_mean + N × clean_std
```

After calibration, each scoring result includes a `deviation_ratio` — how many times the baseline the response deviates:

```python
result = k.score(prompt, response)
print(result.deviation_ratio)  # 1.0× = normal, 2.0× = double baseline, etc.
```

The deviation ratio is a universal signal that normalizes across pipelines. A ratio of 1.0× means the response deviates about as much as your clean samples. A ratio ≥ 2.0× indicates the response has drifted significantly from the user's intent.

If you just want deviation ratios without changing zone thresholds, use `set_baseline()` with a single clean pair:

```python
k.set_baseline("Summarize this doc.\n\n<text>", "The document describes...")
# deviation_ratio now appears in all scoring results
```

### Pause and resume

```python
k.pause()    # Silences detection — score() returns neutral green results
k.resume()   # Reactivates detection
```

Use this when you want to temporarily disable Kaari without removing it from your pipeline.

### Status and reporting

```python
k.status()   # Prints current config, tier, state, scan count

# Enable reporting to track scan counts
k = kaari.Kaari(reporting=True)
# ... after some scoring ...
k.report()   # Prints scan counts by zone
```

### Guard an LLM call

```python
@k.guard
def ask(prompt):
    return my_model.generate(prompt)

response = ask("Explain quantum computing")
# Kaari scores automatically.
# GREEN: silent. YELLOW: terminal warning. RED: terminal alert.
```

### Handle RED zone

```python
# Option 1: Log only (default) — alert appears, process continues
k = kaari.Kaari(on_red="log")

# Option 2: Raise exception — your code catches and decides
k = kaari.Kaari(on_red="raise")
try:
    result = k.score(prompt, suspicious_response)
except kaari.InjectionDetected as e:
    print(f"Blocked: risk={e.result.risk}, score={e.result.score:.4f}")

# Option 3: Custom callback — full control
def my_handler(prompt, response, result):
    send_alert(f"Injection detected: {result.score:.4f}")
    quarantine(response)

k = kaari.Kaari(on_red=my_handler)
```

## How It Works

Kaari embeds the user's prompt and the model's response into the same vector space, then measures their cosine distance.

A clean response stays close to the prompt's intent. An injected response deviates — the model talks about something the user never asked for.

```
prompt  -->  embed(prompt)    -+
                               +--->  dv2 = 1 - cos_sim  --->  C2  --->  zone
response -->  embed(response) -+
```

The **C2 metric** normalizes for response verbosity:

```
C2 = dv2 x (1 + 0.5 x log(response_length / mean_clean_length))
```

### Detection Tiers

| Tier | Metric | AUC | Embed Calls | Use Case |
|------|--------|-----|-------------|----------|
| `fast` | dv2 only | 0.770 | 2 | High-throughput, latency-sensitive |
| `standard` | C2 | 0.822 | 2 | **Default.** Best accuracy/cost balance |

```python
result = k.score(prompt, response, tier="fast")
```

### Paranoid Tier (opt-in add-on)

The paranoid tier adds one LLM inference call per scoring to extract response intent, providing a second signal (dv1) alongside C2.

```python
k = kaari.Kaari(tier="paranoid")
```

**When to use:** High-risk environments where responses handle sensitive data, financial transactions, or identity-critical operations.

**Cost:** 1 additional LLM inference call per scoring (on top of 2 embedding calls). Depending on your LLM provider, this adds latency (2-5s) and per-call cost. For most use cases, the standard tier provides sufficient detection.

## Natural Divergence Note

Kaari measures semantic distance between prompt and response. Some legitimate conversations naturally produce elevated scores:

- **Creative writing** where the response intentionally explores beyond the prompt
- **Debate or counterargument** prompts where the response takes an opposing position
- **Open-ended exploration** where broad prompts lead to specific responses
- **Multi-turn conversations** where accumulated context shifts the semantic frame

If your use case involves consistently high-divergence interactions, you may see more YELLOW zone results. This is expected behavior, not a detection error. Kaari is measuring real semantic distance — it reports what it sees. Consider adjusting the threshold or reviewing YELLOW flags in context rather than treating them as false positives.

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

## What to Pass as the Prompt

This matters. Kaari measures the cosine distance between the embedded prompt and the embedded response. If the prompt you pass to `k.score()` is too short or too generic, the distance will be large even for clean responses — because there's not enough semantic content in the prompt to anchor against.

**Rule: pass the full user intent, not just the instruction.**

If your user says "Summarize this document" and you send the document to the LLM as context, the prompt you pass to Kaari should include both the instruction *and* the document content:

```python
# WRONG — generic instruction, high distance to any detailed response
result = k.score("Summarize this document", response)

# RIGHT — full intent includes the source material
prompt_for_kaari = f"Summarize this document.\n\n{document_text}"
result = k.score(prompt_for_kaari, response)
```

The same applies to RAG pipelines, tool-use chains, and any setup where the user's intent is distributed across multiple inputs. Concatenate everything the model saw that represents the user's intent into one string. Kaari needs the semantic anchoring point — the richer the prompt, the more meaningful the distance measurement.

## Integration

Kaari sits between your LLM call and your user. Score the response *before* showing it.

### Where it goes in your pipeline

```
User prompt -> Your LLM -> response text -> Kaari.score() -> zone -> act
```

### FastAPI example

```python
import kaari

k = kaari.Kaari(on_red="raise")

@app.post("/chat")
async def chat(prompt: str):
    response = await my_llm(prompt)
    try:
        result = k.score(prompt, response)
    except kaari.InjectionDetected:
        return {"text": "Response blocked by security check.", "blocked": True}
    except kaari.KaariError:
        # Detector unavailable — degrade gracefully
        return {"text": response, "scored": False}

    return {
        "text": response,
        "zone": result.zone,
        "score": result.score,
    }
```

### Using the zone for decisions

```python
result = k.score(prompt, response)

if result.zone == "green":
    # Clean — show to user
    pass
elif result.zone == "yellow":
    # Elevated — show with flag, or log for review
    logger.info(f"Yellow zone: score={result.score:.3f}")
elif result.zone == "red":
    # Potential injection — block, escalate, or quarantine
    logger.warning(f"Red zone: score={result.score:.3f}, risk={result.risk}")
```

### Error handling

A security tool that crashes your pipeline is worse than no security tool. Always have a fallback path:

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

## What It Detects

Kaari detects semantic drift between a user's prompt and the model's response — regardless of what caused it. If the response talks about something the user never asked for, Kaari flags it.

The detection is **family-agnostic**: it doesn't classify *what kind* of injection occurred, just *whether* the response drifted. This makes it robust to novel attack types that keyword-based detectors would miss.

## Limitations

Kaari is a **post-hoc analysis tool** that scores responses after generation. It is not a firewall and does not prevent injection — it detects and reports it.

**Known limitations:**

- **Subtle influence injection** (nudging rather than hijacking) produces deviation ratios close to 1.0×, making it hard to distinguish from natural divergence using embedding geometry alone. Kaari reliably detects full hijacks (ratio ≥ 2.0×) but not subtle reframing. Multi-signal detection (VF/Sg) is in development to address this gap.
- **Subtle code injection** is harder to detect. Code injection is functionally motivated, not semantically divergent — the response can execute harmful code while staying on-topic. This is an inherent limit of embedding-geometry detection. Defense against code injection requires input filtering and sandboxing, not post-response analysis.
- **File-based injection** has the lowest detection rate among tested attack types.
- **Simple test prompts** in research (single-turn, short). Long-document and multi-turn scenarios need further validation.
- **Threshold calibration matters.** Default thresholds are calibrated on research data. Use `k.calibrate()` with your own clean data for production deployments. The deviation ratio provides a pipeline-independent signal that normalizes across different setups.

**Validated embedding models:** nomic-embed-text, all-MiniLM-L6-v2, bge-base-en-v1.5. Detection is encoder-independent (AUC spread +/-0.006), but optimal thresholds vary per model.

See [SECURITY.md](SECURITY.md) for responsible use guidance.

## Research

This implementation is based on:

> Lertola, T.S. (2026). "Intent Vectoring: Black-Box Prompt Injection Detection via Semantic Deviation Measurement." Sol Lucid Labs. *arXiv preprint.*

Key findings (N=2,228 across 4 LLM architectures, 3 embedding models):

- AUC-ROC = 0.770 (dv2), 0.822 (C2 with length normalization)
- Combined Cohen's d = 1.72 (large effect size)
- Cross-model generalization confirmed (4 LLMs, no model systematically weaker)
- Encoder-independent detection confirmed (AUC spread +/-0.006 across 3 embedding models)
- 5-stage pipeline, no LLM dependency in scoring path (embedding-only)

## Contributing

Contributions welcome. The core scoring engine is intentionally minimal. If you're adding features, consider whether they belong in core or in a pipeline-specific module.

```bash
git clone https://github.com/SolLucidLabs/kaari.git
cd kaari
pip install -e ".[dev]"
pytest kaari/tests/ -v
```

## License

MIT. See [LICENSE](LICENSE).

---

Built by [Sol Lucid Labs](https://sollucidlabs.com) in Helsinki.
