#!/usr/bin/env python3
"""
Experiment 4: Memory-Scale Degradation Curve (retrieval-layer, no API key)
===========================================================================
Core question: As memory grows from 10 → 500, how do retrieval strategies
change in (a) retrieval quality and (b) token cost?

The hypothesis (from B-e2e's "ties naive-RAG at small scale" finding):
  - naive-RAG's injected-token cost grows linearly → explodes at scale
  - Noesis stays flat (fixed ~1200-token budget)
  - recent-window stays flat (fixed N=5) but recall degrades (older answers lost)

Strategies: Noesis / naive-RAG / recent-window
Scale points: 10, 50, 100, 200, 500
"""
from __future__ import annotations
import os, sys, json, time, random, tempfile, re
from datetime import datetime
from pathlib import Path

NOESIS_DIR = "/Users/mac27ssd/Noesis"
RESULTS_DIR = Path("/Users/mac27ssd/ZCodeProject/noesis_experiment/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, NOESIS_DIR)

SCALE_POINTS = [10, 50, 100, 200, 500]
NUM_TARGETS = 10  # "ground truth" memories we'll query for, embedded among noise
NUM_QUERIES = 10


def _kw_match(retrieved_text: str, target: str) -> bool:
    return target.lower() in retrieved_text.lower()


def _est_tokens(text: str) -> int:
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cjk // 2 + (len(text) - cjk) // 4 + 1


def build_store(n_memories: int, tmp_path: Path):
    """Build a store with n_memories items: NUM_TARGETS 'answer' memories
    (settled, queryable) + (n - NUM_TARGETS) filler/noise memories."""
    import logging; logging.disable(logging.CRITICAL)
    from noesis.memory.main import Memory
    m = Memory.from_config({
        "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        "cold_store":   {"config": {"vault_path": str(tmp_path / "vault")}},
    })
    now = time.time()

    # TARGET memories — distinct, each with a distinctive keyword we'll query for
    targets = [
        ("I use AlpineLinux as my preferred operating system", "alpinelinux", "os"),
        ("I prefer the D programming language for systems work", "d language", "language"),
        ("I chose NetBSD for my home server deployment", "netbsd", "os"),
        ("I use the Zig programming language for new projects", "zig", "language"),
        ("I deploy with Nomad instead of Kubernetes", "nomad", "devops"),
        ("I prefer the Helix editor over Vim", "helix", "editor"),
        ("I use DuckDB for analytical SQL queries", "duckdb", "database"),
        ("I chose Caddy as my web server", "caddy", "devops"),
        ("I prefer the Elixir language for concurrent systems", "elixir", "language"),
        ("I use Tailscale for mesh networking", "tailscale", "networking"),
    ]
    # FILLER memories — plausible but unrelated tech statements.
    # Generate UNIQUE fillers at every scale so naive-RAG's token cost
    # grows linearly (otherwise cycling text hides the explosion).
    def _make_filler(i):
        templates = [
            f"Service alpha-{i} handles {i} thousand requests per second",
            f"The microservice beta-{i} was refactored in sprint {i}",
            f"Database gamma-{i} stores approximately {i*1000} user records",
            f"Team delta-{i} owns the checkout flow module version {i}",
            f"API endpoint epsilon-{i} has a p99 latency of {i*5} milliseconds",
            f"Library zeta-{i} was upgraded to version 2.{i}.0 last week",
            f"Component eta-{i} runs on a node pool of size {i}",
            f"Cache theta-{i} has a hit ratio of {0.6 + i*0.001:.3f}",
            f"Queue iota-{i} processes {i*10} messages per minute",
            f"Worker kappa-{i} lambda-{i} handles scheduled jobs every {i} hours",
        ]
        return templates[i % len(templates)]

    # Insert targets (settled, so injectable)
    for text, _, cluster in targets[:NUM_TARGETS]:
        r = m.add(text, user_id="scale", type="preference", source_tool="eval", topic_cluster=cluster)
        m.vector_store.update(r["results"][0]["id"], {"status":"settled","confidence":0.85,"created_at":now})

    # Insert fillers to reach n_memories (mix of settled/provisional), all UNIQUE
    inserted = NUM_TARGETS
    fi = 0
    while inserted < n_memories:
        text = _make_filler(fi)
        r = m.add(text, user_id="scale", type="fact", source_tool="eval", topic_cluster="general")
        status = "settled" if inserted % 3 else "provisional"
        m.vector_store.update(r["results"][0]["id"], {"status":status,"confidence":0.6,"created_at":now-inserted*10})
        inserted += 1; fi += 1

    # Build queries: ask for each target's keyword
    queries = [
        ("What operating system do I prefer?", "alpinelinux"),
        ("What language do I use for systems work?", "d language"),
        ("What OS is my home server?", "netbsd"),
        ("What language for new projects?", "zig"),
        ("What do I use instead of Kubernetes?", "nomad"),
        ("Which editor do I prefer?", "helix"),
        ("What do I use for analytical SQL?", "duckdb"),
        ("What is my web server?", "caddy"),
        ("What language for concurrent systems?", "elixir"),
        ("What do I use for networking?", "tailscale"),
    ]
    return m, queries


def eval_strategies(m, queries):
    from noesis.context.signals import semantic_signal, recency_signal, core_fact_signal
    results = {"Noesis": [], "naive-RAG": [], "recent-window": []}
    token_costs = {"Noesis": 0, "naive-RAG": 0, "recent-window": 0}

    for q_text, expect_kw in queries:
        # --- Noesis: ContextBuilder.build (budget 1200) ---
        ctx = m._ctx_builder.build(q_text, user_id="scale", budget_tokens=1200)
        noe_texts = []
        if ctx:
            for line in ctx.split("\n"):
                if line.startswith("- ["):
                    try:
                        be = line.index("]"); noe_texts.append(line[be+2:].strip())
                    except: pass
        noe_hit = any(_kw_match(t, expect_kw) for t in noe_texts)
        # rank-based MRR
        noe_mrr = 0.0
        for rank, t in enumerate(noe_texts, 1):
            if _kw_match(t, expect_kw): noe_mrr = 1.0/rank; break
        results["Noesis"].append({"hit":noe_hit, "mrr":noe_mrr})
        token_costs["Noesis"] += _est_tokens(ctx or "")

        # --- naive-RAG: inject ALL non-tentative memories (no budget cap, no selectivity) ---
        # This is the key baseline: it pays O(n) tokens because it injects everything.
        all_mems = m.vector_store.get_recent("scale", n=10000,
                    filter={"status": {"$in": ["provisional", "settled"]}})
        rag_texts = [r.get("text","") for r in all_mems]
        rag_hit = any(_kw_match(t, expect_kw) for t in rag_texts)
        rag_mrr = 0.0
        for rank, t in enumerate(rag_texts, 1):
            if _kw_match(t, expect_kw): rag_mrr = 1.0/rank; break
        results["naive-RAG"].append({"hit":rag_hit, "mrr":rag_mrr})
        rag_sys = "Here is what is known:\n" + "\n".join(f"- {t}" for t in rag_texts)
        token_costs["naive-RAG"] += _est_tokens(rag_sys)

        # --- recent-window: last 5, no budget cap ---
        rec = m.vector_store.get_recent("scale", n=5)
        rec_texts = [r.get("text","") for r in rec]
        rec_hit = any(_kw_match(t, expect_kw) for t in rec_texts)
        rec_mrr = 0.0
        for rank, t in enumerate(rec_texts, 1):
            if _kw_match(t, expect_kw): rec_mrr = 1.0/rank; break
        results["recent-window"].append({"hit":rec_hit, "mrr":rec_mrr})
        rec_sys = "Here is what is known:\n" + "\n".join(f"- {t}" for t in rec_texts)
        token_costs["recent-window"] += _est_tokens(rec_sys)

    # average over queries
    summary = {}
    for s in results:
        hits = sum(1 for r in results[s] if r["hit"])
        avg_mrr = sum(r["mrr"] for r in results[s]) / len(results[s])
        summary[s] = {
            "recall": round(hits/len(queries), 4),
            "mrr": round(avg_mrr, 4),
            "avg_tokens_per_query": round(token_costs[s]/len(queries)),
        }
    return summary


def main():
    random.seed(42)
    print("="*72)
    print("  EXP 4: Memory-Scale Degradation Curve")
    print(f"  Scale: {SCALE_POINTS} | Strategies: Noesis / naive-RAG / recent-window")
    print(f"  Targets: {NUM_TARGETS} (constant) | Queries: {NUM_QUERIES}")
    print("="*72)

    curve = []
    for n in SCALE_POINTS:
        print(f"\n--- n={n} memories ---")
        tmp = Path(tempfile.mkdtemp(prefix=f"scale_{n}_"))
        print(f"  building store...", end=" ", flush=True)
        t0 = time.time()
        m, queries = build_store(n, tmp)
        print(f"done ({time.time()-t0:.1f}s)")
        print(f"  evaluating...")
        summary = eval_strategies(m, queries)
        point = {"n_memories": n, **summary}
        curve.append(point)
        for s in ["Noesis","naive-RAG","recent-window"]:
            d = summary[s]
            print(f"    {s:16s} recall={d['recall']:.0%}  mrr={d['mrr']:.3f}  tokens/q={d['avg_tokens_per_query']}")

    print(f"\n{'='*72}")
    print("  Curve summary")
    print(f"{'='*72}")
    print(f"  {'n':>5s} | {'Noesis':>22s} | {'naive-RAG':>22s} | {'recent-window':>22s}")
    print(f"  {'':5s} | {'rec/mrr/tok':>22s} | {'rec/mrr/tok':>22s} | {'rec/mrr/tok':>22s}")
    print("  " + "-"*78)
    for p in curve:
        n = p["n_memories"]
        noe = p["Noesis"]; rag = p["naive-RAG"]; rec = p["recent-window"]
        print(f"  {n:>5d} | {noe['recall']:>4.0%}/{noe['mrr']:.2f}/{noe['avg_tokens_per_query']:>5d} | "
              f"{rag['recall']:>4.0%}/{rag['mrr']:.2f}/{rag['avg_tokens_per_query']:>5d} | "
              f"{rec['recall']:>4.0%}/{rec['mrr']:.2f}/{rec['avg_tokens_per_query']:>5d}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"scale_degradation_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"experiment":"exp4_scale_degradation","timestamp":ts,"scale_points":SCALE_POINTS,"curve":curve},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {path}")

if __name__ == "__main__":
    main()
