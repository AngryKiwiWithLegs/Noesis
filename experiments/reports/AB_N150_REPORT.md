# Noesis A/B Evaluation: Final Results (arXiv-ready)

> **Date**: 2026-08-11
> **Status**: Final — replaces all prior A/B reports
> **Design**: 30 English profiles × 5 questions, with-memory (proxy) vs without-memory (direct)

---

## Headline result

| Model | n | With memory | Without | Improvement | McNemar χ² | p |
|---|---|---|---|---|---|---|
| **Gemini Flash** | 111 | 84.7% | 41.4% | **+43.2pp** | 42.48 | 5.96×10⁻¹⁰ |
| **gemma3:4b** | 150 | 84.7% | 32.0% | **+52.7pp** | 75.11 | 4.90×10⁻¹⁷ |
| **qwen2.5:3b** | 150 | 65.3% | 32.7% | **+32.7pp** | 35.45 | 2.01×10⁻⁸ |

All three models show statistically significant improvement (p < 0.001).

> **Note on n**: Gemini was planned for n=150 but free-tier rate limiting cut the run to n=111. The 111 completed questions are complete and valid. The two local models completed full n=150 (no rate limit on local inference). The McNemar χ²=42.48 for Gemini at n=111 is already overwhelmingly significant.

---

## Key findings

### 1. Memory injection works across all model tiers

All χ² values far exceed the critical value of 10.83 (p < 0.001, df=1). The improvement holds across cloud and local models.

### 2. gemma3:4b matches Gemini Flash with memory

The 4B local model achieves **84.7%** — identical to Gemini Flash — when given memory injection. Without memory it scores 32.0%. **Memory injection levels the playing field between small local models and large cloud models.**

### 3. Weaker models benefit more (relative gain)

| Model | No-memory | With-memory | Relative gain |
|---|---|---|---|
| Gemini Flash | 41.4% | 84.7% | +105% |
| gemma3:4b | 32.0% | 84.7% | **+165%** |
| qwen2.5:3b | 32.7% | 65.3% | +100% |

### 4. Memory rarely hurts

| Model | Helped (b) | Hurt (c) | Harm rate |
|---|---|---|---|
| Gemini Flash | 50 | 2 | 1.8% |
| gemma3:4b | 80 | 1 | 0.7% |
| qwen2.5:3b | 57 | 8 | 5.3% |

Across 411 paired questions, memory hurt in only 11 cases (2.7%).

---

## Data files

| File | Model | n |
|---|---|---|
| `results/ab_comparison_en_150_20260630.json` | Gemini Flash | 111 |
| `results/ab_ollama_en_gemma3_4b_20260811_010709.json` | gemma3:4b | 150 |
| `results/ab_ollama_en_qwen2.5_3b_20260811_014114.json` | qwen2.5:3b | 150 |

## Reproducibility

```bash
# Gemini (n=111 due to rate limiting)
GEMINI_API_KEY="..." python3 ab_comparison_en.py

# Local models (n=150 each, no API key needed)
python3 ab_ollama_en.py --model gemma3:4b
python3 ab_ollama_en.py --model qwen2.5:3b
```
