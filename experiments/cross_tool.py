#!/usr/bin/env python3
"""
Experiment 2: Cross-Tool Memory Consistency
=============================================
Noesis's headline claim: a stance expressed in tool A is available in tool B.
This is the most distinctive and previously-unquantified feature.

Two mechanisms to evaluate:
  (a) Cross-tool RETRIEVABILITY: is tool-A's stance retrievable when querying
      under tool B? (storage is per-user, so should work by construction)
  (b) Cross-tool CONFIDENCE BOOST: does the cross_tool signal promote a node
      to provisional when two tools corroborate it?

Comparison:
  - Noesis (shared memory across tools) vs per-tool isolation (each tool only
    sees memories it created itself)

Retrieval-layer, no LLM calls.
"""
from __future__ import annotations
import os, sys, json, time, random, tempfile, sqlite3, yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

NOESIS_DIR = "/Users/mac27ssd/Noesis"
RESULTS_DIR = Path("/Users/mac27ssd/ZCodeProject/noesis_experiment/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, NOESIS_DIR)

TOOLS = ["claude-desktop", "chatgpt-web", "gemini-web", "perplexity-web"]


def build_memory(pool_items, tmp_path, with_pipeline=True):
    """pool_items: list of (text, type, source_tool, cluster, user_id)"""
    import logging; logging.disable(logging.CRITICAL)
    from noesis.memory.main import Memory
    if with_pipeline:
        cfg = yaml.safe_load(open(os.path.expanduser("~/.noesis/config.yaml")))
        cfg["vector_store"]["config"]["db_path"] = str(tmp_path / "hot.db")
        cfg["cold_store"]["config"]["vault_path"] = str(tmp_path / "vault")
        cfg_path = tmp_path / "c.yaml"; yaml.safe_dump(cfg, open(cfg_path, "w"))
        m = Memory.from_config_file(str(cfg_path))
    else:
        m = Memory.from_config({
            "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
            "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
            "cold_store":   {"config": {"vault_path": str(tmp_path / "vault")}},
        })
    for text, typ, tool, cluster, uid in pool_items:
        m.add(text, user_id=uid, type=typ, source_tool=tool, topic_cluster=cluster)
        time.sleep(0.3)
    time.sleep(2)  # let pipeline settle
    return m


def get_stored_nodes(m, uid):
    db = sqlite3.connect(str(Path(m.vector_store._path).parent / "hot.db") if hasattr(m.vector_store, "_path") else "")
    return []


def noesis_context(m, query, uid):
    """Full Noesis retrieval path."""
    return m.build_context(query, user_id=uid)


def isolated_context(m, query, uid, query_tool):
    """Per-tool isolation baseline: only memories created by query_tool.
    Simulates ChatGPT-only-sees-ChatGPT-memory. We filter the injected nodes
    to those whose source_tool == query_tool."""
    ctx = m.build_context(query, user_id=uid)
    if not ctx: return ""
    # parse and filter by source_tool — but build_context doesn't expose source_tool.
    # So we reimplement: retrieve and filter.
    from noesis.context.signals import semantic_signal, recency_signal, core_fact_signal
    cands = []
    try: cands += semantic_signal(query, uid, m._retriever, top_k=8)
    except: pass
    try: cands += recency_signal(uid, m.vector_store, n=3)
    except: pass
    try: cands += core_fact_signal(uid, m.vector_store, top_k=3)
    except: pass
    # filter to query_tool only
    filtered = [c for c in cands if c.get("source_tool") == query_tool]
    if not filtered: return ""
    # dedup
    seen, out = set(), []
    for c in filtered:
        cid = c.get("id") or c.get("hash_id", "")
        if cid in seen: continue
        seen.add(cid)
        out.append(c)
    lines = [f"- [{c.get('type','?')}·{c.get('status','?')}] {c.get('text','')}" for c in out]
    return "Here is what is known about the user:\n" + "\n".join(lines) if out else ""


# ── Scenario design (40 scenarios) ──────────────────────────────────────────────
# Each scenario: a stance expressed under tool A, queried under a DIFFERENT tool B.
# Mix of single-tool-source and two-tool-corroboration, across many topics/tools.
# Generated to cover diverse (source_tool, query_tool) pairs and niche keywords.
def _build_scenarios():
    TOOLS = ["claude-desktop", "chatgpt-web", "gemini-web", "perplexity-web"]
    # (stance_text, expect_kw, cluster, query)
    base = [
        ("I decided to use sqlite-vec for vector search, it is lightweight", "sqlite", "vector-store", "What vector database should I use?"),
        ("I prefer PostgreSQL over MySQL for JSON support", "postgres", "database", "Which database do I use?"),
        ("I am learning Rust because it is memory-safe", "rust", "language", "What language am I learning?"),
        ("I think microservices scale better than monoliths", "microservice", "architecture", "What is my architecture preference?"),
        ("I use Docker for all my dev environments", "docker", "tools", "How do I manage dev environments?"),
        ("I prefer Tailwind CSS over handwritten CSS", "tailwind", "frontend", "What CSS framework do I use?"),
        ("I like Vim keybindings, they are efficient", "vim", "editor", "Which editor keybindings do I use?"),
        ("I prefer GraphQL over REST for complex APIs", "graphql", "api", "What API style do I prefer?"),
        ("I use Kafka for high-throughput messaging", "kafka", "messaging", "What message queue do I use?"),
        ("I deploy on GCP, it fits my needs better than AWS", "gcp", "cloud", "Which cloud do I deploy on?"),
        ("I use Helm for Kubernetes package management", "helm", "devops", "How do I manage K8s packages?"),
        ("I chose Snowflake over BigQuery for warehousing", "snowflake", "data", "What data warehouse do I use?"),
        ("I prefer Airflow over Luigi for orchestration", "airflow", "data", "What orchestration tool do I use?"),
        ("I use Terraform for infrastructure as code", "terraform", "devops", "What do I use for infrastructure?"),
        ("I chose ArgoCD over Flux for GitOps", "argocd", "devops", "What is my GitOps tool?"),
        ("I prefer dbt for data transformations", "dbt", "data", "What do I use for transformations?"),
        ("I use Grafana for monitoring dashboards", "grafana", "devops", "What monitoring tool do I prefer?"),
        ("I chose Solidity for smart contract development", "solidity", "blockchain", "What language do I use for smart contracts?"),
        ("I prefer Hardhat over Truffle for dev workflow", "hardhat", "blockchain", "What dev tool do I prefer?"),
        ("I use Fastlane for mobile CI/CD releases", "fastlane", "mobile", "What do I use for mobile CI/CD?"),
        ("I prefer SwiftUI over UIKit for new projects", "swiftui", "mobile", "What UI framework do I use for new projects?"),
        ("I chose Combine over RxSwift for reactive code", "combine", "mobile", "What is my reactive framework?"),
        ("I use PyTorch for all my ML research", "pytorch", "ml", "What ML framework do I prefer?"),
        ("I prefer LoRA fine-tuning over full fine-tuning for cost", "lora", "ml", "What is my fine-tuning approach?"),
        ("I use Weights and Biases for experiment tracking", "weights", "ml", "What do I use for experiment tracking?"),
        ("I chose Hugging Face Transformers for model deployment", "hugging", "ml", "What do I use for model deployment?"),
        ("I prefer Blender for 3D modeling work", "blender", "game", "What do I use for 3D modeling?"),
        ("I think ECS architecture is better than OOP for games", "ecs", "game", "What is my game architecture view?"),
        ("I use Unity for game development", "unity", "game", "What game engine do I use?"),
        ("I prefer C# over C++ for game scripting", "c#", "game", "What language do I prefer for scripting?"),
        ("I use Vault for secrets management", "vault", "security", "What do I use for secrets management?"),
        ("I prefer SentinelOne for endpoint detection", "sentinelone", "security", "What endpoint tool do I prefer?"),
        ("I think zero-trust architecture is essential", "zero-trust", "security", "What is my security philosophy?"),
        ("I use Falco for runtime threat detection", "falco", "security", "What do I use for runtime threat detection?"),
        ("I chose Bun over Node.js for performance", "bun", "language", "What JS runtime do I prefer?"),
        ("I use DuckDB for analytical SQL queries", "duckdb", "data", "What do I use for analytical SQL?"),
        ("I prefer Tailscale for mesh networking", "tailscale", "networking", "What do I use for networking?"),
        ("I use Caddy as my web server", "caddy", "devops", "What is my web server?"),
        ("I prefer NetBSD for my home server deployment", "netbsd", "os", "What OS is my home server?"),
        ("I chose the Zig language for new systems projects", "zig", "language", "What language for new projects?"),
    ]
    import itertools
    sc = []
    tool_cycle = itertools.cycle(TOOLS)
    # alternate single-tool and two-tool (every 3rd is two-tool)
    for i, (text, kw, cluster, query) in enumerate(base):
        src_a = next(tool_cycle)
        qt = next(tool_cycle)
        # ensure query tool differs from source
        if qt == src_a:
            qt = TOOLS[(TOOLS.index(qt)+1) % 4]
        if i % 3 == 2 and i + 1 < len(base):
            # two-tool corroboration: use next base item's reworded stance
            text2 = base[i+1][0].replace("I prefer", "I chose").replace("I use", "I rely on")
            sc.append({"stmts": [(text, src_a), (text2, qt)],
                       "query_tool": TOOLS[(i+2) % 4], "query": query,
                       "expect": kw, "cluster": cluster, "user": f"u{i+1}"})
        else:
            sc.append({"stmts": [(text, src_a)],
                       "query_tool": qt, "query": query,
                       "expect": kw, "cluster": cluster, "user": f"u{i+1}"})
    return sc

SCENARIOS = _build_scenarios()


def main():
    random.seed(42)
    print("=" * 72)
    print("  EXP 2: Cross-Tool Memory Consistency")
    print("=" * 72)
    print(f"  Scenarios: {len(SCENARIOS)}")
    print(f"  Single-tool: {sum(1 for s in SCENARIOS if len(s['stmts'])==1)}")
    print(f"  Two-tool (corroboration): {sum(1 for s in SCENARIOS if len(s['stmts'])==2)}")
    print("=" * 72)

    # Build a single store with ALL scenarios (each under its own user_id)
    tmp = Path(tempfile.mkdtemp(prefix="exp2_"))
    print(f"\nBuilding shared store ({tmp})...")
    import logging; logging.disable(logging.CRITICAL)
    from noesis.memory.main import Memory
    cfg = yaml.safe_load(open(os.path.expanduser("~/.noesis/config.yaml")))
    cfg["vector_store"]["config"]["db_path"] = str(tmp / "hot.db")
    cfg["cold_store"]["config"]["vault_path"] = str(tmp / "vault")
    cfg_path = tmp / "c.yaml"; yaml.safe_dump(cfg, open(cfg_path, "w"))
    m = Memory.from_config_file(str(cfg_path))

    for sc in SCENARIOS:
        for text, tool in sc["stmts"]:
            m.add(text, user_id=sc["user"], type="position",
                  source_tool=tool, topic_cluster=sc["cluster"])
            time.sleep(0.3)
    time.sleep(3)
    print("Store ready.\n")

    results = []
    for sc in SCENARIOS:
        uid, qt, q, exp = sc["user"], sc["query_tool"], sc["query"], sc["expect"]
        is_corroborated = len(sc["stmts"]) > 1

        # Noesis: shared memory (cross-tool)
        ctx_noe = noesis_context(m, q, uid)
        noe_hit = exp.lower() in ctx_noe.lower() if ctx_noe else False

        # Isolated: only query_tool's own memories
        ctx_iso = isolated_context(m, q, uid, qt)
        iso_hit = exp.lower() in ctx_iso.lower() if ctx_iso else False

        results.append({
            "user": uid, "query": q, "query_tool": qt,
            "source_tools": [t for _, t in sc["stmts"]],
            "expect": exp, "corroborated": is_corroborated,
            "noesis_hit": noe_hit, "isolated_hit": iso_hit,
        })
        tag = "2-tool" if is_corroborated else "1-tool"
        n_mark = "✅" if noe_hit else "❌"
        i_mark = "✅" if iso_hit else "❌"
        print(f"  [{tag}] {q[:40]:42s} Noesis{n_mark} Isolated{i_mark}  ({'+'.join(t.split('-')[0] for _,t in sc['stmts'])}→{qt.split('-')[0]})")

    # Summary
    noe_hits = sum(1 for r in results if r["noesis_hit"])
    iso_hits = sum(1 for r in results if r["isolated_hit"])
    # For single-tool-source scenarios queried from a DIFFERENT tool, isolated should FAIL
    cross_only = [r for r in results if not r["corroborated"]]
    cross_noe = sum(1 for r in cross_only if r["noesis_hit"])
    cross_iso = sum(1 for r in cross_only if r["isolated_hit"])

    print(f"\n{'='*72}")
    print("  Summary")
    print(f"{'='*72}")
    print(f"  Noesis (shared/cross-tool) hits:   {noe_hits}/{len(results)}")
    print(f"  Isolated (per-tool) hits:          {iso_hits}/{len(results)}")
    print(f"  --- Cross-tool-only scenarios (source tool != query tool) ---")
    print(f"  Noesis retrieves cross-tool:       {cross_noe}/{len(cross_only)}")
    print(f"  Isolated retrieves cross-tool:     {cross_iso}/{len(cross_only)}  ← should be 0")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"cross_tool_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"experiment": "exp2_cross_tool", "timestamp": ts,
                   "summary": {"noesis_hits": noe_hits, "isolated_hits": iso_hits,
                               "cross_tool_noesis": cross_noe, "cross_tool_isolated": cross_iso,
                               "total": len(results)},
                   "details": results}, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
