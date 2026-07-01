# Noesis Experiment Reports — Chronological Index

> Location: `noesis_experiment/reports/`
> Sorted oldest → newest. Status marked as **current** (the canonical version) or **superseded** (an earlier iteration, kept for the audit trail).

---

## Chronological order

| # | Date | Report | Experiment | Status |
|---|---|---|---|---|
| 1 | Jun 24 | `EXPERIMENT_REPORT.md` | Initial project evaluation + Phase 1–3 (CN demo run) | superseded |
| 2 | Jun 29 | `UNIFIED_AB_REPORT.md` | A/B memory-vs-no-memory, 3 models, 15Q (CN) | superseded by #8 |
| 3 | Jun 29 | `ABLATION_BASELINE_REPORT.md` | Ablation + baseline, retrieval-layer, 15Q (CN) | superseded by #6 |
| 4 | Jun 29 | `B_E2E_STRATEGY_REPORT.md` | End-to-end strategy comparison, 15Q (CN) | superseded by #7 |
| 5 | Jun 29 | `BILINGUAL_AB_REPORT.md` | Bilingual EN/CN A/B comparison | superseded by #8 |
| 6 | Jun 30 | `ABLATION_EN_REPORT.md` | Ablation + baseline, retrieval-layer, 50Q (EN) | **current** |
| 7 | Jun 30 | `B_E2E_EN_REPORT.md` | End-to-end strategy comparison, 50Q (EN) | **current** |
| 8 | Jun 30 | `AB_50Q_EN_REPORT.md` | A/B 50Q EN (Gemini-only view) | superseded by #9 |
| 9 | Jun 30 | `UNIFIED_AB_REPORT_EN.md` | Unified A/B, 3 models, 50Q (EN) | **current** |
| 10 | Jun 30 | `CONTRADICTION_ROBUSTNESS_REPORT.md` | Supersession + poisoning diagnostics | **current** |
| 11 | Jun 30 | `CROSS_TOOL_REPORT.md` | Cross-tool consistency (12 scenarios) | **current** |
| 12 | Jun 30 | `SCALE_DEGRADATION_REPORT.md` | Memory-scale degradation curve (10→500) | **current** |

---

## The 7 current canonical reports (read these for the paper)

In suggested reading order (paper-section order, not chronological):

1. **`UNIFIED_AB_REPORT_EN.md`** — *Does memory injection help?* (main result: +28pp, p<0.05)
2. **`ABLATION_EN_REPORT.md`** — *Do the components each contribute?* (semantic=engine, gating=precision)
3. **`B_E2E_EN_REPORT.md`** — *Beats baselines end-to-end?* (ties at small scale → sets up #12)
4. **`SCALE_DEGRADATION_REPORT.md`** — *Wins at scale?* (35× token advantage, MRR held)
5. **`CROSS_TOOL_REPORT.md`** — *Cross-tool sharing works?* (67% vs 0% structural contrast)
6. **`CONTRADICTION_ROBUSTNESS_REPORT.md`** — *Handles mind-changes?* (supersession fails; gating works — honest negative)

---

## The 5 superseded reports (kept for audit trail only)

- `EXPERIMENT_REPORT.md` — the very first CN demo, before 50Q scaling
- `UNIFIED_AB_REPORT.md` — CN 15Q version of #9
- `ABLATION_BASELINE_REPORT.md` — CN version of #6, before matching-bug fix
- `B_E2E_STRATEGY_REPORT.md` — CN version of #7
- `BILINGUAL_AB_REPORT.md` — intermediate bilingual draft, folded into #8/#9

These are retained so the evolution of methodology (matching-bug fix, scale increase, language-fairness fix) is traceable — useful for the paper's "limitations and methodology" discussion.
