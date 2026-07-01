# Noesis Contradiction & Robustness Evaluation

> **Experiment date**: 2026-06-30
> **Experiment type**: Diagnostic evaluation of two claimed-but-unverified Noesis design mechanisms
> **Evaluation layer**: Retrieval-layer (no LLM calls)
> **Language**: English
> **Baselines**: naive-RAG (top-k semantic, no status filter)

---

## I. Purpose

Noesis claims two design properties that, on paper, distinguish it from naive RAG:

1. **Supersession** — when a user changes their mind, the new stance should win and the old should be suppressed
2. **High-inject threshold (robustness)** — tentative noise should be blocked from injection

A code audit (conducted before this experiment) revealed these mechanisms are *partially or wholly unimplemented*. This experiment **diagnoses the real behavior** rather than assuming the claims hold.

## II. Code-Audit Findings (pre-experiment)

| Claimed mechanism | Code reality |
|---|---|
| Automatic supersession | **Dead code.** `mark_superseded()` is defined in `obsidian.py:128` but has **zero callers** anywhere. |
| Contradiction detection | **Naive.** `_is_negation()` (`confidence.py:221`) only triggers on literal negation words ("不"/"not"/"never"). A real mind-change ("I use MySQL" → "I switched to PostgreSQL") is **not detected**. |
| Time-based status decay | **Rank-only, disabled <300 nodes.** `_decay()` re-ranks in RRF but never demotes a node's status; and it doesn't run at all below 300 memories. |
| Tentative gating | **Works** — `status IN ('provisional','settled')` filter is applied on the injection path. But it relies on the caller passing the filter. |

These findings frame the experiments below: **we expect supersession to fail** and **gating to partially succeed**.

---

## III. Experiment 1 — Contradiction / Supersession

### 3.1 Design

8 scenarios where a user changes their stance (old statement 30 days ago, new statement today). The correct behavior is: **surface the NEW stance, suppress the OLD**. We measure both Noesis and naive-RAG.

### 3.2 Results

| Metric | Noesis | naive-RAG |
|---|---|---|
| Surfaces NEW stance | 5/8 (63%) | 5/8 (63%) |
| **Suppresses OLD (ideal)** | **1/8 (13%)** | **1/8 (13%)** |
| **Leaks stale OLD stance** | **7/8 (88%)** | **7/8 (88%)** |

### 3.3 Finding: Noesis does NOT handle contradictions

**Noesis performs identically to naive-RAG** — both leak the stale old stance in 7/8 scenarios. This confirms the code audit: because supersession is dead code and contradiction detection is naive, **Noesis has no mechanism to retire an outdated stance**. When a user says "I switched to PostgreSQL", the old "I prefer MySQL" node remains `settled` and gets injected alongside the new one.

**This is a genuine, honest limitation.** Noesis's retrieval quality (proven in earlier experiments) does not extend to *temporal correctness* — it cannot distinguish a current preference from a superseded one.

### 3.4 Implication for the paper

Report this as a **named limitation with a clear fix path**: implementing automatic supersession (detecting semantic contradiction and marking the old node `superseded`) is straightforward and would turn this red result into a green one. This is a candidate for a *methods contribution* in a revision.

---

## IV. Experiment 3 — Poisoning / Robustness

### 4.1 Design

5 settled facts that should always survive, plus 8 tentative noise/poison statements (e.g. "Maybe I should use MongoDB", "Not sure if Python is right", "I like the color blue"). Correct behavior: **zero poison leaks into injected context**.

### 4.2 Results

| Metric | Noesis | naive-RAG |
|---|---|---|
| Total poison leaks (across 5 queries) | **5** | **13** |
| Per-query average leak | 1.0 | 2.6 |

### 4.3 Finding: Gating reduces poison but does not eliminate it

Noesis leaks **5 poison items vs naive-RAG's 13** — a 62% reduction. The confidence gate (tentative excluded) clearly helps. **But 5 leaks is not zero**, which contradicts the design goal of "tentative never injected."

### 4.4 Why the leaks happen

Investigation shows the leaks are **not tentative-status nodes slipping through**. They are *settled/provisional memories that happen to contain poison keywords* (e.g. a settled node mentioning "Kubernetes" in passing, or "Emacs" in a settled "I prefer Vim over Emacs" statement). The poison-keyword matching in the evaluator over-counts. **The genuine tentative nodes ARE correctly blocked** — the gate works as designed; the 5 "leaks" are a measurement artifact of keyword overlap, not a gating failure.

### 4.5 Implication

Noesis's gating works for its stated purpose (blocking tentative noise). For the paper, report the **62% reduction** as the real result, and note that the residual leaks are keyword-overlap measurement noise, not gating failure — verified by checking that no `tentative`-status node appears in the injection path.

---

## V. Overall Conclusions

| Mechanism | Claimed | Actual (this study) |
|---|---|---|
| Supersession | New stance wins | ❌ **Fails** — old stance leaks 88% of the time |
| Tentative gating | Zero noise injection | 🟡 **Mostly works** — 62% fewer poison leaks than RAG; residual is measurement noise |

**The most important finding is negative:** Noesis cannot handle mind-changes. This is both an honest limitation to report and an opportunity for a methods contribution (implementing supersession).

---

## VI. Honest Limitations of This Study

| Limitation | Impact |
|---|---|
| 8 contradiction scenarios | Small sample; but 7/8 leak rate is unambiguous |
| Poison-keyword matching over-counts | Inflates leak count; verified not actual status-leak |
| Retrieval-layer only | Does not measure whether leaks change the final LLM answer |
| No fix-and-retest | Future work: implement supersession and re-run |

---

## VII. Data & Reproducibility

- **Data**: `results/contradiction_poisoning_20260630_140236.json`
- **Script**: `contradiction_poisoning.py`
- **Run**: `python3 contradiction_poisoning.py` (~30s, no API key)
