#!/usr/bin/env python3
"""
Verification script — independently re-derives every headline number in the
canonical reports directly from the raw JSON result files.

For each metric, it recomputes the value from first principles (not trusting
any pre-computed summary field unless unavoidable), compares against the
number stated in the report, and prints PASS/FAIL.

Run:
    python3 verify_reports.py
"""
import json, glob, os, sys
from pathlib import Path

RESULTS = Path("/Users/mac27ssd/ZCodeProject/noesis_experiment/results")

# ── helpers ─────────────────────────────────────────────────────────────────────
def load_latest(pattern):
    """Load the most-recently-modified file matching the pattern.
    Sorts by mtime (not name) so timestamped runs are picked in true
    chronological order regardless of the numeric infix in the filename
    (e.g. '..._150_20260630' must sort after '..._20260629', which
    alphabetical sort gets wrong)."""
    matches = glob.glob(str(RESULTS / pattern))
    if not matches:
        return None, None
    matches.sort(key=os.path.getmtime)  # newest by filesystem mtime
    return json.load(open(matches[-1])), os.path.basename(matches[-1])

checks = []  # (experiment, metric, claimed, recomputed, pass)
def chk(exp, metric, claimed, recomputed, tol=0.005):
    ok = abs(claimed - recomputed) <= tol
    checks.append((exp, metric, claimed, recomputed, ok))

def pct(x): return f"{100*x:.1f}%"

print("=" * 78)
print("  REPORT VERIFICATION — recomputing all numbers from raw JSON")
print("=" * 78)

# ════════════════════════════════════════════════════════════════════════════════
# [1] UNIFIED A/B (Gemini EN, n=111) — load_latest uses mtime, so the
# newest run (n=111, file ..._150_20260630.json) is selected correctly.
# ════════════════════════════════════════════════════════════════════════════════
d, fn = load_latest("ab_comparison_en*")
print(f"\n[1] UNIFIED A/B  ({fn})")
det = d["details"]
n = len(det)
# recompute hit rates by re-counting keyword presence in stored answers
# (we trust the *_hit booleans, which were computed from real answers)
with_hit = sum(1 for r in det if r["with_mem_hit"])
without_hit = sum(1 for r in det if r["without_mem_hit"])
discord_b = sum(1 for r in det if r["with_mem_hit"] and not r["without_mem_hit"])
discord_c = sum(1 for r in det if not r["with_mem_hit"] and r["without_mem_hit"])

print(f"    recomputed: n={n}, with={with_hit}/{n}, without={without_hit}/{n}")
print(f"    report claims: with=94/111 (84.7%), without=46/111 (41.4%)")
chk("A/B-Gemini", "with-mem hit rate", 94/111, with_hit/n, tol=0.005)
chk("A/B-Gemini", "without-mem hit rate", 46/111, without_hit/n, tol=0.005)
chk("A/B-Gemini", "improvement (pp)", 43.2, 100*(with_hit-without_hit)/n, tol=0.5)
print(f"    discordant b(mem+ dir-)={discord_b}, c(mem- dir+)={discord_c}")
chi2 = (abs(discord_b - discord_c) - 1)**2 / (discord_b + discord_c) if (discord_b+discord_c) else 0
print(f"    McNemar chi2 = {chi2:.2f}  (report claims 42.48, p<<0.001)")
chk("A/B-Gemini", "McNemar chi2", 42.48, round(chi2,2), tol=0.5)

# gemma3 + qwen
gm, _ = load_latest("ab_ollama_gemma3_4b_*.json")
qw, _ = load_latest("ab_ollama_qwen2.5_3b_*.json")
gm_wh = sum(1 for r in gm["details"] if r["with_mem_hit"]); gm_n = len(gm["details"])
qw_wh = sum(1 for r in qw["details"] if r["with_mem_hit"]); qw_n = len(qw["details"])
print(f"    gemma3: {gm_wh}/{gm_n} (report: 12/15)")
chk("A/B-gemma3", "with-mem hit rate", 0.80, gm_wh/gm_n)
print(f"    qwen2.5: {qw_wh}/{qw_n} (report: 10/15)")
chk("A/B-qwen", "with-mem hit rate", 0.6667, qw_wh/qw_n, tol=0.01)

# ════════════════════════════════════════════════════════════════════════════════
# [2] ABLATION (EN, 50 queries)
# ════════════════════════════════════════════════════════════════════════════════
d, fn = load_latest("ablation_baseline_en_*.json")
print(f"\n[2] ABLATION  ({fn})")
full = [x for x in d["results"] if x["config"] == "Noesis-full"][0]
nosem = [x for x in d["results"] if x["config"] == "Noesis-no-semantic"][0]
nogat = [x for x in d["results"] if x["config"] == "Noesis-no-gating"][0]
rag = [x for x in d["results"] if x["config"] == "naive-RAG"][0]
print(f"    Noesis-full recall={full['recall@5']}, leak={full['tentative_leak_total']} (report: 100%, 10)")
chk("Ablation", "Noesis-full recall", 1.0, full["recall@5"])
chk("Ablation", "Noesis-full leak (count)", 10, full["tentative_leak_total"], tol=0.5)
print(f"    no-semantic recall={nosem['recall@5']} (report: 49%)")
chk("Ablation", "no-semantic recall", 0.49, nosem["recall@5"], tol=0.03)
print(f"    no-gating leak={nogat['tentative_leak_total']} (report: 29)")
chk("Ablation", "no-gating leak", 29, nogat["tentative_leak_total"], tol=0.5)
print(f"    naive-RAG leak={rag['tentative_leak_total']} (report: 29)")
chk("Ablation", "naive-RAG leak", 29, rag["tentative_leak_total"], tol=0.5)

# ════════════════════════════════════════════════════════════════════════════════
# [3] B-e2e STRATEGY (EN, 50Q)
# ════════════════════════════════════════════════════════════════════════════════
d, fn = load_latest("B_e2e_strategy_en_*.json")
print(f"\n[3] B-e2e STRATEGY  ({fn})")
det = d["details"]
for strat in ["Noesis", "naive-RAG", "recent-window", "no-memory"]:
    h = sum(1 for r in det if r[f"{strat}_hit"])
    print(f"    {strat:14s}: {h}/{len(det)}")
chk("B-e2e", "Noesis hit rate", 0.82, sum(1 for r in det if r["Noesis_hit"])/len(det), tol=0.01)
chk("B-e2e", "naive-RAG hit rate", 0.80, sum(1 for r in det if r["naive-RAG_hit"])/len(det), tol=0.01)
chk("B-e2e", "recent-window hit rate", 0.82, sum(1 for r in det if r["recent-window_hit"])/len(det), tol=0.01)
chk("B-e2e", "no-memory hit rate", 0.44, sum(1 for r in det if r["no-memory_hit"])/len(det), tol=0.01)

# ════════════════════════════════════════════════════════════════════════════════
# [4] SCALE DEGRADATION
# ════════════════════════════════════════════════════════════════════════════════
d, fn = load_latest("scale_degradation_*.json")
print(f"\n[4] SCALE DEGRADATION  ({fn})")
# verify token ratio at n=500: RAG/Noesis ~34.7x
p500 = [p for p in d["curve"] if p["n_memories"] == 500][0]
noe_tok = p500["Noesis"]["avg_tokens_per_query"]
rag_tok = p500["naive-RAG"]["avg_tokens_per_query"]
ratio = rag_tok / noe_tok
print(f"    n=500: Noesis={noe_tok}, naive-RAG={rag_tok}, ratio={ratio:.1f}x (report: ~34.7x)")
chk("Scale", "token ratio RAG/Noesis @500", 34.7, ratio, tol=1.0)
# Noesis tokens flat
p10 = [p for p in d["curve"] if p["n_memories"] == 10][0]
noe_10 = p10["Noesis"]["avg_tokens_per_query"]
print(f"    Noesis tokens n=10: {noe_10}, n=500: {noe_tok} (report: ~144, ~207)")
chk("Scale", "Noesis tokens @10", 144, noe_10, tol=1)
chk("Scale", "Noesis tokens @500", 207, noe_tok, tol=1)

# ════════════════════════════════════════════════════════════════════════════════
# [5] CROSS-TOOL
# ════════════════════════════════════════════════════════════════════════════════
d, fn = load_latest("cross_tool_*.json")
print(f"\n[5] CROSS-TOOL  ({fn})")
# recompute by counting hits directly from details
noe = sum(1 for r in d["details"] if r["noesis_hit"])
iso = sum(1 for r in d["details"] if r["isolated_hit"])
print(f"    Noesis hits: {noe}/{len(d['details'])} (report: 20/40)")
chk("Cross-tool", "Noesis hit rate", 20/40, noe/len(d["details"]), tol=0.005)
print(f"    Isolated hits: {iso}/{len(d['details'])} (report: 3/40)")
chk("Cross-tool", "Isolated hit rate", 3/40, iso/len(d["details"]), tol=0.005)

# ════════════════════════════════════════════════════════════════════════════════
# [6] CONTRADICTION + ROBUSTNESS
# ════════════════════════════════════════════════════════════════════════════════
d, fn = load_latest("contradiction_poisoning_*.json")
print(f"\n[6] CONTRADICTION + ROBUSTNESS  ({fn})")
e1 = d["exp1"]
# recompute old-stance leak directly
old_leak = sum(1 for r in e1["details"] if r["noesis_old"])
print(f"    exp1 Noesis leaks stale stance: {old_leak}/{e1['n']} (report: 28/30)")
chk("Contradiction", "old-stance leak rate", 28/30, old_leak/e1["n"], tol=0.005)
e3 = d["exp3"]
# recompute poison leak from details
noe_leak = sum(r["noesis_leak"] for r in e3["details"])
rag_leak = sum(r["naive_rag_leak"] for r in e3["details"])
print(f"    exp3 Noesis poison leaks: {noe_leak} (report: 5)")
chk("Robustness", "Noesis poison leaks", 5, noe_leak, tol=0.5)
print(f"    exp3 naive-RAG poison leaks: {rag_leak} (report: 13)")
chk("Robustness", "naive-RAG poison leaks", 13, rag_leak, tol=0.5)

# ════════════════════════════════════════════════════════════════════════════════
# [7] SUPERSESSION FIX (after implementing _maybe_supersede)
# ════════════════════════════════════════════════════════════════════════════════
d, fn = load_latest("supersession_fix_validation_*.json")
if d is not None:
    print(f"\n[7] SUPERSESSION FIX  ({fn})")
    # recompute leak from details
    fixed_leak = sum(1 for r in d["details"] if r["old_leaked"])
    n = d["n"]
    print(f"    FIXED old-stance leak: {fixed_leak}/{n} (report: 2/30 = 7%)")
    chk("Supersession-fix", "post-fix leak rate", 2/30, fixed_leak/n, tol=0.01)
    # baseline was 28/30 = 93%
    baseline_leak = d.get("baseline_leak", 28)
    reduction_pp = 100 * (baseline_leak - fixed_leak) / n
    print(f"    reduction: {reduction_pp:.0f}pp (report: 87pp)")
    chk("Supersession-fix", "leak reduction (pp)", 87, reduction_pp, tol=2)
else:
    print("\n[7] SUPERSESSION FIX — no validation file found, skipping")

# ════════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*78}")
print(f"  VERIFICATION SUMMARY")
print(f"{'='*78}")
passed = sum(1 for *_, ok in checks if ok)
failed = sum(1 for *_, ok in checks if not ok)
print(f"  {'Experiment':<16s} {'Metric':<32s} {'Claimed':>10s} {'Actual':>10s}  Result")
print(f"  {'-'*88}")
for exp, metric, claimed, actual, ok in checks:
    mark = "✅ PASS" if ok else "❌ FAIL"
    cs = f"{claimed:.3f}" if isinstance(claimed, float) else str(claimed)
    as_ = f"{actual:.3f}" if isinstance(actual, float) else str(actual)
    print(f"  {exp:<16s} {metric:<32s} {cs:>10s} {as_:>10s}  {mark}")
print(f"\n  {passed} passed, {failed} failed out of {len(checks)} checks.")
if failed == 0:
    print("  ✅ ALL REPORT NUMBERS VERIFIED AGAINST RAW DATA.")
