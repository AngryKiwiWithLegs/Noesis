# Noesis Experiments

Evaluation harness, raw results, and derived reports for the Noesis
memory-injection research. All experiments run against the live Noesis
proxy (`localhost:8080`) with **synthetic personas** as test subjects —
no real user data.

## Layout

```
experiments/
├── reports/      Derived Markdown reports (the publishable narrative)
├── results/      Raw JSON output from each experiment run
├── *.py          Experiment scripts (each runnable standalone)
└── conversations_*.json/.html   Sample conversation export (synthetic)
```

## Key experiments

| Script | What it measures |
|---|---|
| `ab_comparison_en.py` / `ab_comparison.py` | A/B: with-Noesis vs without-memory, EN + ZH |
| `ablation_baseline_en.py` | Component ablation (semantic/recency/core-fact/gating off) |
| `B_e2e_strategy_en.py` | End-to-end vs naive-RAG vs recent-window vs no-memory |
| `cross_tool.py` | Cross-tool consistency (same user, different AI tools) |
| `contradiction_poisoning.py` | Robustness to contradictory/stale stances |
| `scale_degradation.py` | Token-budget behavior as memory grows |
| `supersession_fix_validation.py` | Before/after the supersession fix (93%→7% leak) |
| `verify_reports.py` | Re-derives every report number from raw JSON (25 checks) |

## Reproducibility

```bash
cd ~/Noesis && source .venv/bin/activate
noesis start --ws                    # proxy on :8080

# Most experiments need a model key in the environment, e.g.:
GEMINI_API_KEY="..." python3 experiments/ab_comparison_en.py

# Verify every headline number recomputes from raw data:
python3 experiments/verify_reports.py
```

`verify_reports.py` is the integrity check: it recomputes all 25 report
metrics from first principles against `results/*.json` and prints PASS/FAIL
for each. As of the last run: **25/25 PASS**.

## Re-runs for the paper (October 2026)

`ablation_baseline_en_20261009_190520.json`, `poisoning_exact_20261009_190546.json`
and `scale_degradation_20261009_190657.json` were produced with the June
software versions: Python 3.12.11 (x86-64), torch 2.2.2, transformers 4.44.2,
sentence-transformers 3.0.1, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.2,
all-MiniLM-L6-v2. Before the changes below, the re-runs reproduced the June
ablation and scale numbers exactly.

- Ablation and poisoning: a leak now counts only when the retrieved item *is*
  a tentative/poison statement. The old keyword metric also counted settled
  memories that shared a word with one ("Kafka", "Emacs").
- Scale: adds a standard top-5 dense-retrieval RAG baseline (`topk-RAG`).
- Ablation, second correction (`ablation_baseline_en_20261017_102429.json`,
  same software versions; two runs gave identical results): scoring uses exact
  memory identity instead of phrase matching (which credited any memory sharing
  wording such as "I prefer"); "Precision@5", which only counted non-tentative
  results, is replaced by the recall, size and leak of the full injected
  context; every Noesis variant runs the real `ContextBuilder` with one signal
  replaced, so the old core-fact and recency rows can now differ from the full
  system; memory ages are written to the store (`update()` had dropped them);
  and the recent-window and random baselines draw from every stored memory,
  tentative ones included.

`python3 experiments/verify_paper.py` recomputes every number printed in the
paper from these files and the June result files.
