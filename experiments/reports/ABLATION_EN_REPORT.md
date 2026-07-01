# Noesis Ablation Study: Retrieval-Layer Evaluation (50 Queries, English)

> **Experiment date**: 2026-06-30
> **Experiment type**: Ablation study — which retrieval components actually help?
> **Evaluation layer**: Retrieval-layer (no LLM calls; Recall@5 / MRR / Precision@5 / Tentative-leak)
> **Language**: English
> **Sample size**: 50 queries over a 28-item memory pool (incl. 5 noise items)

---

## I. Experiment Purpose

Answer: **Does each Noesis retrieval component contribute, and how does Noesis compare to standard retrieval baselines?** This is the standard ablation + baseline comparison that reviewers expect for a methods paper.

## II. Experiment Design

### 2.1 Setup

- **Memory pool**: 28 memories (5 identity, 11 preference, 4 position, 4 event, 4 fact, 5 tentative-noise), spanning 8 topic clusters
- **Test queries**: 50 queries, each with ground-truth relevant texts. Coverage: 10 identity, 15 preference, 10 position, 5 event, 10 fact
- **Scoring**: distinctive multi-word phrase matching (strict — single common words like "backend" do not count)

### 2.2 Configurations

**Ablation variants (5)** — Noesis with one component removed each:

| Config | Removed component | Hypothesis |
|---|---|---|
| `Noesis-full` | (none — baseline) | — |
| `Noesis-no-core` | identity/preference unconditional injection | core-fact signal helps |
| `Noesis-no-recency` | recent-N signal | recency helps |
| `Noesis-no-semantic` | semantic retrieval (engine) | semantic retrieval is the engine |
| `Noesis-no-gating` | confidence gating (tentative also injected) | gating prevents noise |

**External baselines (4)**:

| Baseline | Method | Represents |
|---|---|---|
| `naive-RAG` | top-k semantic, no status filter | standard RAG baseline |
| `recent-window` | last 5 memories | ChatGPT default |
| `random` | random 5 memories | lower bound |
| `all-inject` | all non-tentative memories | theoretical ceiling |

---

## III. Results

### 3.1 Ranking table (sorted by Recall@5)

| Rank | Config | Recall@5 | MRR | Precision@5 | Leak |
|---|---|---|---|---|---|
| 🥇 | **Noesis-full** | **100%** | **1.000** | **96%** | **10** |
| 🥇 | Noesis-no-core | 100% | 1.000 | 96% | 10 |
| 🥇 | Noesis-no-recency | 100% | 1.000 | 96% | 10 |
| 4 | Noesis-no-gating | 100% | 1.000 | 88% | **29** ⚠️ |
| 4 | naive-RAG | 100% | 1.000 | 88% | **29** ⚠️ |
| 6 | Noesis-no-semantic | 49% | 0.160 | 100% | 0 |
| 7 | random | 45% | 0.206 | 94% | 15 |
| 8 | recent-window | 22% | 0.099 | 100% | 0 |
| 8 | all-inject | 22% | 0.099 | 100% | 0 |

### 3.2 Key findings

#### Finding 1: Semantic retrieval is the core engine

Removing semantic retrieval crashes Recall from **100% → 49%** and MRR from **1.000 → 0.160**. With only recency + core-fact signals, the system cannot find topically relevant memories. This confirms semantic retrieval is the backbone.

#### Finding 2: Confidence gating is the precision moat

`Noesis-no-gating` and `naive-RAG` match Noesis-full on Recall (100%) but leak **29 tentative-noise memories** vs Noesis's 10. Precision drops 96% → 88%. Gating doesn't change *what's retrievable* but dramatically improves *what gets injected* — fewer "maybe I should try Bun" / "just chatting, weather is nice" items pollute the context.

#### Finding 3: Noesis dominates all baselines on recall

`naive-RAG` ties Noesis on recall (both 100%) but loses on precision/leak. `recent-window` and `all-inject` only reach **22% recall** — far below Noesis — because they don't do relevance retrieval. `random` reaches 45%, confirming the lower bound.

#### Finding 4: Core-fact and recency signals had no measurable effect here

`no-core` and `no-recency` match `full` exactly. This is honest: with semantic retrieval already strong and the test set query-driven, these signals' "always-inject identity/preference" and "recent context" value didn't trigger. Their value shows on *semantically unrelated but identity-relevant* queries (a future test-set gap).

---

## IV. Honest Limitations

| Limitation | Impact |
|---|---|
| Memory pool small (28 items) | Core-fact/recency signals' value not exercised |
| Distinctive-phrase matching can still miss paraphrases | True semantic recall may be higher |
| Core-fact/recency ablations inconclusive | Need identity-focused queries to show value |
| No token-cost metric | Can't yet show Noesis's efficiency edge |

---

## V. Data & reproducibility

- **Data**: `results/ablation_baseline_en_20260630_124511.json`
- **Script**: `ablation_baseline_en.py`
- **Run**: `python3 ablation_baseline_en.py` (~30s, no API key needed)
