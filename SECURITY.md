# Security & Responsible Use

## What Kaari Is

Kaari is a **detection and monitoring tool** for prompt injection in LLM applications. It measures semantic deviation between user intent and model response to flag suspicious behavior.

## What Kaari Is Not

Kaari is **not a firewall, filter, or safety guarantee**.

- It does not prevent injection — it detects it after the response is generated
- It does not guarantee detection of all injection types
- It does not replace human review for high-stakes applications
- It does not validate the factual correctness of responses

## Responsible Use

### DO use Kaari to:
- Monitor LLM responses for deviation from user intent
- Log and alert on suspicious response patterns
- Add a detection layer to existing LLM pipelines
- Research and study prompt injection patterns

### DO NOT rely on Kaari as:
- The sole security measure for safety-critical systems
- A replacement for input sanitization and prompt hardening
- A guarantee against adversarial attacks
- A content moderation or factual accuracy system

## Known Limitations

1. **Adversarial robustness is untested.** An attacker who knows Kaari is in use could craft responses that maintain low cosine distance while still being injected. This is an active research area.

2. **Threshold calibration matters.** The default threshold (0.291) is derived from research data with specific models and prompt types. Your deployment may need different thresholds. Consider running calibration on your own data.

3. **Embedding model dependency.** Detection quality depends on the embedding model. Results validated with nomic-embed-text (768-dim). Other models may produce different separation characteristics.

4. **Simple prompt bias.** Research used short, single-turn prompts. Performance on long documents, multi-turn conversations, or system prompts with extensive context is not yet validated.

5. **Detection is embedding-model-dependent.** Separation quality varies by embedding model. Results are validated with nomic-embed-text; other models may require recalibration.

## Reporting Vulnerabilities

If you find a way to consistently bypass Kaari detection, we'd like to know. This helps us improve the tool for everyone.

Contact: tatu.lertola@gmail.com

Please include:
- The prompt and injection used
- The model and embedding provider
- The Kaari score and expected vs actual classification
- Whether you believe this represents a general bypass or a specific edge case

## Accuracy Statement

As of v0.2.0, Kaari achieves:

| Metric | Value | Conditions |
|--------|-------|------------|
| AUC-ROC (C2) | 0.883 | N=1,944, 4 models |
| Cohen's d | 1.72 | Combined effective |
| Sensitivity | ~0.65 | At Youden-optimal threshold |
| Specificity | ~0.83 | At Youden-optimal threshold |

These numbers are from controlled research conditions. Real-world performance may vary. We encourage users to validate on their own data before deploying to production.

The false positive rate at the default threshold is approximately 17%. For applications where false positives are costly, consider using a higher threshold or the `fast` tier with manual review of flagged responses.

## Citation

If you use Kaari in research, please cite:

```bibtex
@article{lertola2026intent,
  title={Intent Vectoring: Black-Box Prompt Injection Detection via Semantic Deviation Measurement},
  author={Lertola, Tatu Samuli},
  journal={SSRN preprint},
  year={2026}
}
```
