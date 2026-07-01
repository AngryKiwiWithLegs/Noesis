# Noesis Evaluation: Consolidated Results (Scaled, arXiv-ready)

> **Date**: 2026-06-30
> **Language**: English
> **Status**: arXiv v1 candidate

---

## Headline result

At the enlarged sample (n=111 paired questions, Gemini), Noesis's memory injection improves answer accuracy from **41.4% → 84.7% (+43.2pp)**. McNemar's test: χ²=42.48, **p ≈ 7×10⁻¹¹** — overwhelmingly significant.

---

## Scaled-up experiments

This document supersedes the earlier reports with larger, more credible sample sizes:

| Experiment | Old n | New n | Why scaled |
|---|---|---|---|
| A/B headline (Gemini, end-to-end) | 50 | **111** (of 150 planned) | Statistical power |
| Cross-tool consistency | 12 | **40** | Defensible claim |
| Contradiction/supersession | 8 | **30** | Reproducible negative |
| Ablation (retrieval-layer) | 50 | 50 (unchanged) | Already sufficient |
| B-e2e strategy (end-to-end) | 50 | 50 (unchanged) | Already sufficient |
| Scale degradation | 10×5 | 10×5 (unchanged) | Already sufficient |

> **Note on A/B**: the planned n=150 run was interrupted by Gemini free-tier rate limiting after n=111. The 111 completed questions are complete and valid; the partial-run status is reported transparently. n=111 is already more than double the original and yields a far stronger significance result.

---

## [1] A/B: Memory injection (n=111, end-to-end)

| Metric | Value |
|---|---|
| With-memory hit rate | **94/111 (84.7%)** |
| Without-memory hit rate | 46/111 (41.4%) |
| Improvement | **+43.2pp** |
| McNemar χ² | **42.48** (p ≈ 7×10⁻¹¹) |
| Discordant pairs | b=50 (memory helped), c=2 (memory hurt) |

The significance is now overwhelming (vs the n=50 run's χ²=7.68). Only 2 of 111 cases showed memory "hurting" — both retrieval-precision edge cases.

**Data**: `results/ab_comparison_en_150_20260630.json`

---

## [2] Cross-tool consistency (n=40, retrieval-layer)

| Metric | Noesis (shared) | Isolated (per-tool) |
|---|---|---|
| Total hits | **20/40 (50%)** | 3/40 (7.5%) |
| Cross-tool-only (source ≠ query tool) | 10/27 (37%) | **0/27 (0%)** |

The 0/27 isolated result on pure cross-tool scenarios confirms the structural argument: **per-tool memory systems cannot share across tools by design.** Noesis's 50% (limited by retrieval precision, not architecture) is the only option.

The Noesis hit rate dropped from the n=12 run's 67% to 50% — at larger n, retrieval-precision limitations surface honestly. The headline contrast (50% vs 7.5%) is still decisive.

**Data**: `results/cross_tool_20260630_205622.json`

---

## [3] Contradiction / supersession (n=30, retrieval-layer)

| Metric | Value |
|---|---|
| Noesis surfaces NEW stance | 12/30 (40%) |
| Noesis suppresses OLD (ideal) | **1/30 (3%)** |
| Noesis LEAKS stale OLD stance | **28/30 (93%)** |
| naive-RAG ideal | 1/30 (3%) |

At n=30 the supersession failure is confirmed beyond doubt: **Noesis leaks the stale old stance 93% of the time, identical to naive-RAG.** This is a reproducible, named limitation (supersession is dead code; contradiction detection only matches literal negation).

Poisoning/robustness (unchanged): Noesis 5 poison leaks vs naive-RAG 13 — gating reduces noise 62%.

**Data**: `results/contradiction_poisoning_20260630_205805.json`

---

## [4–6] Unchanged experiments (already at scale)

- **Ablation** (n=50): Noesis-full 100% recall, semantic=engine (-51pp), gating=precision (leak 10 vs 29)
- **B-e2e strategy** (n=50): Noesis 82% / naive-RAG 80% / recent-window 82% / no-memory 44%
- **Scale degradation** (10→500): Noesis tokens flat (~207) vs naive-RAG exploding (7192 at n=500, 35×)

---

## Verification

All numbers above were independently recomputed from raw JSON by `verify_reports.py`: **23/23 checks pass.** Run it yourself:

```bash
cd ~/ZCodeProject/noesis_experiment && python3 verify_reports.py
```

---

## Honest limitations for arXiv v1

1. **n=111, not 150** — Gemini rate-limited the run. Reported transparently; χ² still overwhelming.
2. **Keyword-hit scoring** — coarse; no LLM-as-judge yet.
3. **Single primary model (Gemini)** — gemma3/qwen runs help but a Claude/GPT-4 replication would strengthen.
4. **Supersession fails (93% leak)** — named, reproducible limitation with a clear fix path.
5. **Cross-tool recall limited by retrieval precision** (50%), not architecture — honest gap.
6. **Retrieval-layer experiments don't measure final-answer quality** — they measure whether the right memory reaches the prompt.

These limitations are real and stated plainly. v1 publishes with them; v2 addresses them.
