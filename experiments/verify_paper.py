#!/usr/bin/env python3
"""
verify_paper.py — recompute every experimental number printed in the paper
(paper/main.tex and the Pages/Word versions) from the raw JSON results in
experiments/results/, and print PASS/FAIL for each.

Each check names the exact raw file it reads, so the paper's numbers are tied
to specific runs rather than to "the latest file". Lifecycle constants are
read from the Noesis source. Run from anywhere:

    python3 experiments/verify_paper.py
"""
import ast, json, math, os, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
sys.path.insert(0, str(ROOT))

rows = []
def chk(section, what, paper, actual, ok=None):
    if ok is None:
        ok = paper == actual
    rows.append((section, what, paper, actual, ok))

def load(name):
    return json.load(open(RES / name))

def r1(x):            # one-decimal percentage, as printed
    return round(100 * x, 1)

def mcnemar_cc(b, c):
    return (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0

def exact_mcnemar_p(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

# ── RQ1: A/B, n = 110 per model (Table 3, abstract, contributions) ──────────────
ab = {}
for model, fname, w, wo, d, x2 in [
        ("Gemini Flash-Lite", "ab_n110_gemini_flash.json", 85.5, 41.8, 43.6, 42.48),
        ("gemma3:4b",         "ab_n110_gemma3_4b.json",    82.7, 34.5, 48.2, 51.02),
        ("qwen2.5:3b",        "ab_n110_qwen2.5_3b.json",   62.7, 37.3, 25.5, 16.57)]:
    det = load(fname)["details"]; n = len(det)
    wh = sum(r["with_mem_hit"] for r in det); woh = sum(r["without_mem_hit"] for r in det)
    b = sum(r["with_mem_hit"] and not r["without_mem_hit"] for r in det)
    c = sum(r["without_mem_hit"] and not r["with_mem_hit"] for r in det)
    ab[model] = (det, wh, woh, b, c, n)
    chk("RQ1", f"{model}: n", 110, n)
    chk("RQ1", f"{model}: with memory %", w, r1(wh / n))
    chk("RQ1", f"{model}: without memory %", wo, r1(woh / n))
    chk("RQ1", f"{model}: gain pp", d, round(100 * (wh - woh) / n, 1))
    chk("RQ1", f"{model}: McNemar chi2", x2, round(mcnemar_cc(b, c), 2))
    chk("RQ1", f"{model}: p < 0.001 (chi2 > 10.83)", True, mcnemar_cc(b, c) > 10.83)
chk("RQ1", "memory hurt in 10 of 330 answers", (10, 330), (sum(v[4] for v in ab.values()), sum(v[5] for v in ab.values())))
_, gw, gwo, *_ = ab["gemma3:4b"]
chk("Discussion", "gemma3:4b relative gain %", 139, round(100 * (gw - gwo) / gwo))
same_q = all(a["question"].startswith(b_["question"][:36]) or b_["question"].startswith(a["question"][:36])
             for a, b_ in zip(ab["Gemini Flash-Lite"][0], ab["gemma3:4b"][0]))
chk("RQ1", "identical question order across models", True, same_q)
gb = sum(a["with_mem_hit"] and not g["with_mem_hit"] for a, g in zip(ab["Gemini Flash-Lite"][0], ab["gemma3:4b"][0]))
gc = sum(g["with_mem_hit"] and not a["with_mem_hit"] for a, g in zip(ab["Gemini Flash-Lite"][0], ab["gemma3:4b"][0]))
chk("Conclusion", "gemma vs Gemini with memory: no significant difference (p > 0.05)", True, exact_mcnemar_p(gb, gc) > 0.05)
src = load("ab_comparison_en_150_20260630.json")
chk("Models", "cloud model id recorded for the 30 June run", "gemini-flash-lite-latest", src.get("model"))
chk("Limitations", "30 June run stopped after 111 questions", 111, len(src["details"]))
chk("Limitations", "n=110 rows are the first 110 of the June run", True,
    all((x["question"], x["with_mem_hit"], x["without_mem_hit"]) == (y["question"], y["with_mem_hit"], y["without_mem_hit"])
        for x, y in zip(ab["Gemini Flash-Lite"][0], src["details"][:110])))
chk("Limitations", "a later session added 35 more (146 total)", 146, len(load("ab_comparison_en_n146_20260811_041949.json")["details"]))

# ── Supersession (Table 2) ──────────────────────────────────────────────────────
before = load("contradiction_poisoning_20260630_205805.json")["exp1"]["details"]
after = load("supersession_fix_validation_20260701_101518.json")["details"]
chk("Table 2", "same 30 scenarios before and after", True, [r["query"] for r in before] == [r["query"] for r in after])
bl = sum(r["noesis_old"] for r in before); bn = sum(r["noesis_new"] for r in before)
bi = sum(r["noesis_new"] and not r["noesis_old"] for r in before)
al = sum(r["old_leaked"] for r in after); an = sum(r["new_surfaced"] for r in after)
ai = sum(r["new_surfaced"] and not r["old_leaked"] for r in after)
for what, paper, actual in [("old stance leaked before", 28, bl), ("old stance leaked after", 2, al),
                            ("new stance surfaced before", 12, bn), ("new stance surfaced after", 28, an),
                            ("ideal before", 1, bi), ("ideal after", 27, ai),
                            ("leak change pp", -87, round(100 * (al - bl) / 30)),
                            ("surfaced change pp", 53, round(100 * (an - bn) / 30)),
                            ("ideal change pp", 87, round(100 * (ai - bi) / 30)),
                            ("leak before % (93%)", 93, round(100 * bl / 30)), ("leak after % (7%)", 7, round(100 * al / 30))]:
    chk("Table 2", what, paper, actual)

# ── RQ2: ablation (Table 4) — corrected leak metric ─────────────────────────────
abl = {r["config"]: r for r in load("ablation_baseline_en_20261009_190520.json")["results"]}
for cfg, rec, mrr, prec, leak in [("Noesis-full", 1.00, 1.000, 1.000, 0), ("Noesis-no-core", 1.00, 1.000, 1.000, 0),
                                  ("Noesis-no-recency", 1.00, 1.000, 1.000, 0), ("Noesis-no-gating", 1.00, 1.000, 0.924, 19),
                                  ("Noesis-no-semantic", 0.49, 0.160, 1.000, 0), ("naive-RAG", 1.00, 1.000, 0.924, 19),
                                  ("recent-window", 0.22, 0.099, 1.000, 0), ("random", 0.45, 0.206, 1.000, 0)]:
    r = abl[cfg]
    chk("Table 4", f"{cfg}", (rec, mrr, prec, leak),
        (round(r["recall@5"], 2), round(r["mrr"], 3), round(r["precision@5"], 3), r["tentative_leak_total"]))

# ── RQ2: poisoning (exact poison-item matching) ─────────────────────────────────
po = load("poisoning_exact_20261009_190546.json")["exp3"]
chk("RQ2", "poisoning: 8 hedged statements, 5 queries", (8, 5), (po["n_poison"], po["n_queries"]))
chk("RQ2", "poisoning: Noesis injects 0, ungated top-5 returns 13 of 25 slots", (0, 13, 25),
    (po["noesis_total_leak"], po["naive_rag_total_leak"], sum(d["rag_retrieved"] for d in po["details"])))

# ── RQ3: scale (Figure 4) ───────────────────────────────────────────────────────
new = {p["n_memories"]: p for p in load("scale_degradation_20261009_190657.json")["curve"]}
old = {p["n_memories"]: p for p in load("scale_degradation_20260630_142946.json")["curve"]}
chk("RQ3", "re-run reproduces the June run for Noesis / naive / recent", True,
    all(new[n][s] == old[n][s] for n in old for s in ("Noesis", "naive-RAG", "recent-window")))
chk("RQ3", "Noesis tokens 144 -> 207", (144, 207), (new[10]["Noesis"]["avg_tokens_per_query"], new[500]["Noesis"]["avg_tokens_per_query"]))
chk("RQ3", "inject-all tokens at 500", 7192, new[500]["naive-RAG"]["avg_tokens_per_query"])
chk("RQ3", "inject-all / Noesis token ratio (35)", 35, round(new[500]["naive-RAG"]["avg_tokens_per_query"] / new[500]["Noesis"]["avg_tokens_per_query"]))
chk("RQ3", "Noesis recall 0.9 and MRR 0.90 at every size (9/10 at rank 1)", True,
    all(new[n]["Noesis"]["recall"] == 0.9 and new[n]["Noesis"]["mrr"] == 0.9 for n in new))
chk("RQ3", "inject-all MRR at 500 (0.002)", 0.002, round(new[500]["naive-RAG"]["mrr"], 3))
chk("RQ3", "recent-window recall 0 beyond 10", True, all(new[n]["recent-window"]["recall"] == 0 for n in new if n > 10))
chk("RQ3", "top-5 RAG recall 0.9 and MRR 0.90 at every size", True,
    all(new[n]["topk-RAG"]["recall"] == 0.9 and new[n]["topk-RAG"]["mrr"] == 0.9 for n in new))
chk("RQ3", "top-5 RAG tokens 66-68", (66, 68), (min(new[n]["topk-RAG"]["avg_tokens_per_query"] for n in new), max(new[n]["topk-RAG"]["avg_tokens_per_query"] for n in new)))
chk("RQ3", "budget of 1,200 never reached by Noesis", True, max(new[n]["Noesis"]["avg_tokens_per_query"] for n in new) < 1200)

# ── RQ4: cross-tool (Figure 5) ──────────────────────────────────────────────────
det = load("cross_tool_20260630_205622.json")["details"]
strict = [r for r in det if len(r["source_tools"]) == 1 and r["query_tool"] not in r["source_tools"]]
same = [r for r in det if r["query_tool"] in r["source_tools"]]
third = [r for r in det if len(r["source_tools"]) == 2 and r["query_tool"] not in r["source_tools"]]
cnt = lambda rs, k: sum(r[k] for r in rs)
chk("RQ4", "scenario split 27 / 7 / 6 of 40", (27, 7, 6, 40), (len(strict), len(same), len(third), len(det)))
chk("RQ4", "strictly cross-tool: Noesis 10/27, isolated 0/27", (10, 0), (cnt(strict, "noesis_hit"), cnt(strict, "isolated_hit")))
chk("RQ4", "37% strictly cross-tool", 37, round(100 * cnt(strict, "noesis_hit") / len(strict)))
chk("RQ4", "asking tool saw it: Noesis 6/7, isolated 3/7", (6, 3), (cnt(same, "noesis_hit"), cnt(same, "isolated_hit")))
chk("RQ4", "overall: Noesis 50%, isolated 7.5%", (50.0, 7.5), (r1(cnt(det, "noesis_hit") / 40), r1(cnt(det, "isolated_hit") / 40)))

# ── End-to-end strategy comparison ──────────────────────────────────────────────
e2e = load("B_e2e_strategy_en_20260630_130707.json")
chk("RQ1", "end-to-end: Noesis / inject-all / recent / none of 50", (41, 40, 41, 22, 50),
    tuple(e2e["summary"][s]["hit"] for s in ("Noesis", "naive-RAG", "recent-window", "no-memory")) + (e2e["summary"]["Noesis"]["total"],))
chk("RQ1", "end-to-end model", "gemini-flash-lite-latest", e2e.get("model"))

# ── Latency (§3.1) ──────────────────────────────────────────────────────────────
lat = load("latency_20260629_143306.json")
m = lat["metrics"]
chk("Latency", "Phase 1 (Memory.add) median / p99 ms", (21, 684), (round(m["add_p50_ms"]), round(m["add_p99_ms"])))
chk("Latency", "context build median / p99 ms", (21, 55), (round(m["ctx_p50_ms"]), round(m["ctx_p99_ms"])))
chk("Latency", "100 adds, 50 context builds", (100, 50), (m["num_adds"], m["num_queries"]))

# ── Lifecycle constants (§3.2), read from source ────────────────────────────────
def const(path, name):
    tree = ast.parse(open(ROOT / path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
weights = None
for node in ast.walk(ast.parse(open(ROOT / "noesis/thoughts/confidence.py").read())):
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "WEIGHTS" for t in node.targets):
        weights = ast.literal_eval(node.value)
chk("§3.2", "weights 0.35/0.25/0.25/0.15", (0.35, 0.25, 0.25, 0.15),
    (weights["repetition"], weights["assertion"], weights["no_contradiction"], weights["cross_tool"]))
chk("§3.2", "thresholds provisional 0.45 / settled 0.75", (0.45, 0.75),
    (const("noesis/thoughts/types.py", "THRESHOLD_PROVISIONAL"), const("noesis/thoughts/types.py", "THRESHOLD_SETTLED")))
score = lambda rep, a, nc, ct: weights["repetition"] * rep + weights["assertion"] * a + weights["no_contradiction"] * nc + weights["cross_tool"] * ct
chk("§3.2", "single firm statement scores exactly 0.45", 0.45, round(score(0, 0.8, 1, 0), 3))
need = lambda a, ct: next(k for k in range(1, 20) if score(min(0.15 * k, 1), a, 1, ct) >= 0.75 - 1e-9)
chk("§3.2", "settling needs >=3 similar thoughts with cross-tool, >=6 without", (3, 6),
    (max(need(0.8, 1), need(0.9, 1)), max(need(0.8, 0), need(0.9, 0))))
chk("§3.2", "never settles without repetition", True, score(0, 0.9, 1, 1) < 0.75)

# ── Star map counts (§6) — only when the local store is present ────────────────
db = Path.home() / ".noesis" / "hot.db"
if db.exists():
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    q = lambda s: con.execute(s).fetchone()[0]
    chk("§6", "store shown: 601 thoughts, 18 settled", (601, 18),
        (q("select count(*) from items"), q("select count(*) from items where status='settled'")))
    chk("§6", "stored 'general' topic (+3 inferred by the viewer = 355)", 352, q("select count(*) from items where topic_cluster='general'"))

# ── report ───────────────────────────────────────────────────────────────────────
w = max(len(r[1]) for r in rows)
for sec, what, paper, actual, ok in rows:
    print(f"{'PASS' if ok else 'FAIL'}  {sec:<11s} {what:<{w}s}  paper={paper}  data={actual}")
fails = sum(not r[4] for r in rows)
print(f"\n{len(rows) - fails} passed, {fails} failed out of {len(rows)} checks.")
sys.exit(1 if fails else 0)
