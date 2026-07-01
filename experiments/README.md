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
| `ablation_baseline_en.py` | Component ablation (semantic/gating/graph off) |
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
