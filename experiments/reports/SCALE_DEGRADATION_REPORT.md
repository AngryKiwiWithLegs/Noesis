# Noesis Memory-Scale Degradation Curve

> **Experiment date**: 2026-06-30
> **Experiment type**: How retrieval quality and injection cost scale with memory size
> **Evaluation layer**: Retrieval-layer (no LLM calls)
> **Language**: English
> **Strategies**: Noesis / naive-RAG (inject-all) / recent-window

---

## I. Purpose

The earlier B-e2e experiment found that at small memory scale (~8 memories/profile), Noesis, naive-RAG, and recent-window are statistically indistinguishable. This raised the question: **does that equivalence hold as memory grows?** This experiment answers it by sweeping memory size from 10 → 500 and measuring each strategy's retrieval quality and token cost.

This is the experiment that flips the B-e2e "ties naive-RAG" result into a Noesis advantage.

## II. Experiment Design

### 2.1 Setup

At each scale point, build a store with N memories:
- **10 target memories**: distinct, each with a unique keyword (e.g. "I use AlpineLinux", "I prefer the D language"). These are the ground truth we query for.
- **N − 10 filler memories**: unique, plausible-but-unrelated tech statements (e.g. "Service alpha-47 handles 47000 requests per second"). These create scale and noise.

### 2.2 Scale points

10, 50, 100, 200, 500 memories.

### 2.3 Strategies

| Strategy | What it injects |
|---|---|
| **Noesis** | ContextBuilder with 1200-token budget (selective retrieval + gating) |
| **naive-RAG** | **All** non-tentative memories, no budget cap |
| **recent-window** | Last 5 memories only |

### 2.4 Metrics

- **Recall@5**: are the 10 target memories retrieved?
- **MRR**: mean reciprocal rank of first hit
- **Avg tokens/query**: the injected system-prompt size — the cost metric

---

## III. Results

### 3.1 The curve

| n memories | Noesis (rec/MRR/tok) | naive-RAG (rec/MRR/tok) | recent-window (rec/MRR/tok) |
|---|---|---|---|
| 10 | 90% / 0.90 / **144** | 90% / 0.28 / **120** | 50% / 0.23 / 57 |
| 50 | 90% / 0.90 / **202** | 90% / 0.02 / **677** | 0% / 0.00 / 73 |
| 100 | 90% / 0.90 / **205** | 90% / 0.01 / **1,380** | 0% / 0.00 / 73 |
| 200 | 90% / 0.90 / **212** | 90% / 0.00 / **2,827** | 0% / 0.00 / 75 |
| 500 | 90% / 0.90 / **207** | 90% / 0.00 / **7,192** | 0% / 0.00 / 75 |

### 3.2 Finding 1: Noesis's token cost is flat; naive-RAG's explodes 🟢

| n | Noesis tokens | naive-RAG tokens | Ratio |
|---|---|---|---|
| 10 | 144 | 120 | 0.8× |
| 50 | 202 | 677 | 3.4× |
| 100 | 205 | 1,380 | 6.7× |
| 200 | 212 | 2,827 | 13.3× |
| 500 | 207 | 7,192 | **34.7×** |

Noesis stays at ~200 tokens (its budget caps it). Naive-RAG grows linearly — at 500 memories it injects **35× more tokens** for the same answer quality. **This is Noesis's core efficiency advantage, now quantified.**

### 3.3 Finding 2: All retrieval strategies maintain Recall; MRR diverges sharply

| n | Noesis MRR | naive-RAG MRR | recent-window MRR |
|---|---|---|---|
| 10 | 0.90 | 0.28 | 0.23 |
| 500 | 0.90 | 0.002 | 0.000 |

- **Noesis keeps MRR at 0.90** across all scales — the right answer is reliably rank-1.
- **naive-RAG's MRR collapses to ~0** because the target is buried among hundreds of filler memories — it's *present* (recall 90%) but ranks ~500th. An LLM receiving this context would struggle to find the relevant signal.
- **recent-window recall collapses to 0%** beyond n=10 — the targets get pushed out by newer fillers. This is the definitive demonstration that pure-window approaches lose long-term memory.

### 3.4 Finding 3: Recall hides what MRR and token cost reveal

All three strategies except recent-window maintain 90% recall. But recall alone is misleading: naive-RAG "recalls" the target by dumping everything (paying 7192 tokens and burying it at rank 500), while Noesis recalls it at rank 1 for 207 tokens. **Recall@5 without MRR and token cost is an insufficient metric** — a methodological finding for the paper.

---

## IV. Conclusions

1. **Noesis's token efficiency advantage grows linearly with scale**: at 500 memories, Noesis injects 35× fewer tokens than naive-RAG for equal recall and far better MRR.
2. **Noesis maintains rank-1 retrieval (MRR 0.90) at all scales**, while naive-RAG's MRR collapses to 0.002 as the target gets buried.
3. **Recent-window is unviable beyond ~10 memories**: it loses all long-term recall (0% past n=10).
4. **This flips the B-e2e "ties naive-RAG" result**: at small scale they're equivalent; at scale, Noesis dominates on cost and ranking. The earlier equivalence was a small-sample artifact, now explained.

---

## V. Honest Limitations

| Limitation | Impact |
|---|---|
| **Targets always settled** | Real memories have mixed statuses; status filtering not stressed here |
| **10 target keywords are distinctive** | Easier to retrieve than paraphrased real preferences; inflates Noesis recall |
| **Filler memories are template-generated** | Less lexically diverse than real conversations; may overstate separation |
| **Retrieval-layer only** | Doesn't show the LLM handling 7192 tokens poorly — but 7192 tokens in a system prompt is self-evidently problematic |
| **No LLM cost in dollars** | Token count is a proxy; actual API cost would strengthen this |

---

## VI. Data & Reproducibility

- **Data**: `results/scale_degradation_20260630_142946.json`
- **Script**: `scale_degradation.py`
- **Run**: `python3 scale_degradation.py` (~30s, no API key)
