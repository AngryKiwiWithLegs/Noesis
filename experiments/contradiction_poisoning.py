#!/usr/bin/env python3
"""
Experiments 1 & 3: Contradiction/Supersession + Poisoning/Robustness
=====================================================================
Two retrieval-layer experiments that test Noesis's CLAIMED but UNVERIFIED
design mechanisms, against naive-RAG baseline. No LLM calls.

Experiment 1 — Contradiction/Supersession:
  User states X, later states NOT-X (changed mind). Does Noesis surface the
  NEW stance and suppress the OLD? Tests supersession + no_contradiction.

Experiment 3 — Poisoning/Robustness:
  Settled facts exist. Inject tentative noise/poison statements. Does the
  high-inject threshold keep them out? Tests confidence gating.

IMPORTANT: Code exploration found supersession is dead code and contradiction
detection is naive (literal negation only). These experiments are designed
to DIAGNOSE that honestly, not to confirm it works.
"""
from __future__ import annotations
import os, sys, json, time, random, tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

NOESIS_DIR = "/Users/mac27ssd/Noesis"
RESULTS_DIR = Path("/Users/mac27ssd/ZCodeProject/noesis_experiment/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, NOESIS_DIR)


def _kw_match(retrieved_text: str, target: str) -> bool:
    """Strict: target's distinctive phrase must appear verbatim (lowercased)."""
    rt = retrieved_text.lower()
    return target.lower() in rt


def build_memory(pool_items, tmp_path):
    """Build a Memory store from a list of (text, type, status, confidence, age_days, cluster)."""
    import logging; logging.disable(logging.CRITICAL)
    from noesis.memory.main import Memory
    m = Memory.from_config({
        "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        "cold_store":   {"config": {"vault_path": str(tmp_path / "vault")}},
    })
    now = time.time()
    for text, typ, status, conf, age, cluster in pool_items:
        r = m.add(text, user_id="exp", type=typ, source_tool="eval", topic_cluster=cluster)
        hid = r["results"][0]["id"]
        m.vector_store.update(hid, {
            "status": status, "confidence": conf,
            "created_at": now - age * 86400,
        })
    return m


def noesis_retrieve(memory, query, k=5):
    """Noesis full path: ContextBuilder.build → parse injected nodes."""
    ctx = memory._ctx_builder.build(query, user_id="exp", budget_tokens=2000)
    if not ctx: return []
    nodes = []
    for line in ctx.split("\n"):
        if line.startswith("- ["):
            try:
                be = line.index("]"); meta = line[3:be]; text = line[be+2:].strip()
                nodes.append({"text": text})
            except: pass
    return nodes[:k]


def naive_rag(memory, query, k=5):
    """Baseline: top-k semantic, NO status filter (everything injectable)."""
    return memory._retriever.search(query, user_id="exp", top_k=k)


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Contradiction / Supersession
# ════════════════════════════════════════════════════════════════════════════════
def run_exp1():
    print("="*72)
    print("  EXP 1: Contradiction / Supersession")
    print("="*72)

    # Scenario pairs: (old stance, new stance, query, correct answer keyword)
    # 30 scenarios — user "changes their mind" between old and new.
    scenarios = [
        ("I prefer MySQL for my database", 30, "I switched to PostgreSQL, it has better JSON support", 0, "What database do I use now?", "postgresql", "mysql", "database"),
        ("I use Python for everything", 30, "I have moved to Rust for systems programming", 0, "What language do I program in now?", "rust", "python", "language"),
        ("I think monoliths are the way to go", 30, "I now believe microservices are better for scaling", 0, "What is my current view on architecture?", "microservice", "monolith", "architecture"),
        ("I deploy on AWS", 30, "I migrated everything to GCP last month", 0, "Which cloud do I deploy on now?", "gcp", "aws", "cloud"),
        ("I like JavaScript", 30, "I switched to TypeScript, the type safety is worth it", 0, "What do I code in now?", "typescript", "javascript", "language"),
        ("I use REST APIs everywhere", 30, "I have moved to GraphQL for complex apps", 0, "What API style do I prefer now?", "graphql", "rest", "api"),
        ("I use Vim", 30, "I switched to Emacs recently", 0, "Which editor do I use now?", "emacs", "vim", "editor"),
        ("I prefer Redis for caching", 30, "I moved to Memcached, simpler for my needs", 0, "What cache do I use now?", "memcached", "redis", "cache"),
        ("I use RabbitMQ for messaging", 30, "I migrated to Kafka for higher throughput", 0, "What message queue do I use now?", "kafka", "rabbitmq", "messaging"),
        ("I prefer MongoDB", 30, "I switched to PostgreSQL for relational integrity", 0, "What database do I use now?", "postgresql", "mongodb", "database"),
        ("I use Java for backend", 30, "I have moved to Go for concurrency", 0, "What backend language do I use now?", "go", "java", "language"),
        ("I deploy on Heroku", 30, "I migrated to Kubernetes on AWS", 0, "Where do I deploy now?", "kubernetes", "heroku", "cloud"),
        ("I use SVN for version control", 30, "I switched to Git years ago", 0, "What version control do I use now?", "git", "svn", "tools"),
        ("I prefer SOAP web services", 30, "I have moved to REST for simplicity", 0, "What web service style do I use now?", "rest", "soap", "api"),
        ("I use Memcached for sessions", 30, "I migrated to Redis for persistence", 0, "What do I use for sessions now?", "redis", "memcached", "cache"),
        ("I prefer PHP for web", 30, "I switched to Node.js for async I/O", 0, "What web runtime do I use now?", "node", "php", "language"),
        ("I use Apache as my web server", 30, "I moved to Nginx for performance", 0, "What web server do I use now?", "nginx", "apache", "devops"),
        ("I prefer Bash scripting", 30, "I have switched to Python for scripting", 0, "What do I script in now?", "python", "bash", "language"),
        ("I use Firebase for backend", 30, "I migrated to Supabase for open-source", 0, "What BaaS do I use now?", "supabase", "firebase", "backend"),
        ("I prefer XML for data", 30, "I switched to JSON for lighter payloads", 0, "What data format do I use now?", "json", "xml", "api"),
        ("I use Jenkins for CI", 30, "I migrated to GitHub Actions last quarter", 0, "What CI tool do I use now?", "github actions", "jenkins", "devops"),
        ("I prefer Docker Swarm", 30, "I have moved to Kubernetes for orchestration", 0, "What orchestrator do I use now?", "kubernetes", "swarm", "devops"),
        ("I use MySQL for analytics", 30, "I switched to ClickHouse for columnar queries", 0, "What analytics database do I use now?", "clickhouse", "mysql", "data"),
        ("I prefer Grunt for builds", 30, "I moved to Webpack for module bundling", 0, "What build tool do I use now?", "webpack", "grunt", "frontend"),
        ("I use Vue for frontend", 30, "I have switched to Svelte for smaller bundles", 0, "What frontend framework do I use now?", "svelte", "vue", "frontend"),
        ("I prefer Stripe for payments", 30, "I migrated to Adyen for global reach", 0, "What payment processor do I use now?", "adyen", "stripe", "business"),
        ("I use ElasticSearch for search", 30, "I moved to Meilisearch for simplicity", 0, "What search engine do I use now?", "meilisearch", "elastic", "data"),
        ("I prefer Webpack for bundling", 30, "I switched to Vite for faster builds", 0, "What bundler do I use now?", "vite", "webpack", "frontend"),
        ("I use Vercel for hosting", 30, "I migrated to Cloudflare Pages for edge", 0, "Where do I host now?", "cloudflare", "vercel", "cloud"),
        ("I prefer SendGrid for email", 30, "I moved to Postmark for deliverability", 0, "What email service do I use now?", "postmark", "sendgrid", "business"),
    ]

    # Pool: each scenario contributes 2 nodes (old settled + new settled) + filler
    pool = []
    for old_t, old_age, new_t, new_age, *_, cluster in scenarios:
        pool.append((old_t, "preference", "settled", 0.80, old_age, cluster))
        pool.append((new_t, "preference", "settled", 0.85, new_age, cluster))
    # filler to make the store non-trivial
    fillers = [
        ("My name is John, I am a backend engineer", "identity", "settled", 0.90, 0, "identity"),
        ("Our team has 7 engineers", "fact", "settled", 0.85, 60, "team"),
        ("Production runs on Kubernetes", "fact", "settled", 0.80, 45, "infra"),
    ]
    pool.extend(fillers)

    tmp = Path(tempfile.mkdtemp(prefix="exp1_"))
    print(f"Building store ({len(pool)} items)...")
    memory = build_memory(pool, tmp)

    results = []
    for old_t, old_age, new_t, new_age, query, exp_new, exp_old, cluster in scenarios:
        noe = noesis_retrieve(memory, query, k=5)
        rag = naive_rag(memory, query, k=5)
        noe_texts = [n["text"] for n in noe]
        rag_texts = [r.get("text","") for r in rag]

        # Does each method surface the NEW stance and/or the OLD stale stance?
        noe_new = any(_kw_match(t, exp_new) for t in noe_texts)
        noe_old = any(_kw_match(t, exp_old) for t in noe_texts)
        rag_new = any(_kw_match(t, exp_new) for t in rag_texts)
        rag_old = any(_kw_match(t, exp_old) for t in rag_texts)

        # Correct behavior: surface NEW, suppress OLD
        noe_correct = noe_new and not noe_old
        rag_correct = rag_new and not rag_old

        results.append({
            "query": query, "old": old_t, "new": new_t,
            "exp_new": exp_new, "exp_old": exp_old,
            "noesis_new": noe_new, "noesis_old": noe_old, "noesis_correct": noe_correct,
            "rag_new": rag_new, "rag_old": rag_old, "rag_correct": rag_correct,
        })
        n_mark = "✅" if noe_correct else "❌"
        r_mark = "✅" if rag_correct else "❌"
        print(f"  {query[:42]:44s} Noesis{n_mark}(new={noe_new},old={noe_old}) RAG{r_mark}(new={rag_new},old={rag_old})")

    noe_correct_n = sum(1 for r in results if r["noesis_correct"])
    rag_correct_n = sum(1 for r in results if r["rag_correct"])
    noe_new_n = sum(1 for r in results if r["noesis_new"])
    noe_old_leak = sum(1 for r in results if r["noesis_old"])

    print(f"\n  Summary:")
    print(f"    Noesis surfaces NEW stance:    {noe_new_n}/{len(results)}")
    print(f"    Noesis suppresses OLD (ideal): {noe_correct_n}/{len(results)}")
    print(f"    Noesis LEAKS stale OLD stance: {noe_old_leak}/{len(results)}  ← lower is better")
    print(f"    naive-RAG ideal:               {rag_correct_n}/{len(results)}")
    return {"experiment":"exp1_contradiction","n":len(results),
            "noesis_new_surfaced":noe_new_n,"noesis_ideal_suppress_old":noe_correct_n,
            "noesis_old_leak":noe_old_leak,"naive_rag_ideal":rag_correct_n,
            "details":results}


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Poisoning / Robustness
# ════════════════════════════════════════════════════════════════════════════════
def run_exp3():
    print("\n" + "="*72)
    print("  EXP 3: Poisoning / Robustness")
    print("="*72)

    # Settled facts (should always survive), and tentative noise/poison (should be blocked)
    settled = [
        ("I prefer PostgreSQL for databases", "preference", "settled", 0.85, 5, "database"),
        ("I am a Python developer", "identity", "settled", 0.90, 0, "language"),
        ("I think microservices scale better", "position", "settled", 0.80, 10, "architecture"),
        ("I use Docker for everything", "preference", "settled", 0.82, 3, "tools"),
        ("I prefer Vim over Emacs", "preference", "settled", 0.78, 7, "editor"),
    ]
    poison = [
        # tentative noise — should NEVER be injected
        ("Maybe I should use MongoDB instead", "preference", "tentative", 0.25, 0, "database"),
        ("Not sure if Python is right for me", "position", "tentative", 0.20, 1, "language"),
        ("Perhaps monoliths are actually fine", "position", "tentative", 0.15, 2, "architecture"),
        ("Might switch away from Docker", "preference", "tentative", 0.20, 0, "tools"),
        ("Just thinking, Emacs might be better", "preference", "tentative", 0.10, 1, "editor"),
        ("Wondering if I should try Go", "position", "tentative", 0.18, 0, "language"),
        ("Could Kubernetes be overkill?", "position", "tentative", 0.12, 1, "infra"),
        ("Randomly, I like the color blue", "fact", "tentative", 0.05, 0, "general"),
    ]

    pool = settled + poison
    tmp = Path(tempfile.mkdtemp(prefix="exp3_"))
    print(f"Building store ({len(settled)} settled + {len(poison)} poison)...")
    memory = build_memory(pool, tmp)

    poison_kws = ["mongodb", "monolith", "emacs", "kubernetes", "blue", "go"]
    queries = [
        "What database do I prefer?",
        "What language do I develop in?",
        "What is my view on architecture?",
        "What tools do I use?",
        "Which editor do I prefer?",
    ]

    results = []
    total_noe_leak = 0
    total_rag_leak = 0
    for q in queries:
        noe = noesis_retrieve(memory, q, k=5)
        rag = naive_rag(memory, q, k=5)
        noe_texts = [n["text"] for n in noe]
        rag_texts = [r.get("text","") for r in rag]

        noe_leak = sum(1 for t in noe_texts if any(kw in t.lower() for kw in poison_kws))
        rag_leak = sum(1 for t in rag_texts if any(kw in t.lower() for kw in poison_kws))
        total_noe_leak += noe_leak
        total_rag_leak += rag_leak

        results.append({
            "query": q, "noesis_leak": noe_leak, "naive_rag_leak": rag_leak,
            "noesis_retrieved": len(noe_texts), "rag_retrieved": len(rag_texts),
        })
        print(f"  {q[:36]:38s} Noesis leak={noe_leak}  RAG leak={rag_leak}")

    print(f"\n  Summary:")
    print(f"    Noesis total poison leaks:   {total_noe_leak}  ← should be ~0")
    print(f"    naive-RAG total poison leaks:{total_rag_leak}  ← higher = worse")
    return {"experiment":"exp3_poisoning","n_queries":len(queries),"n_poison":len(poison),
            "noesis_total_leak":total_noe_leak,"naive_rag_total_leak":total_rag_leak,
            "details":results}


def main():
    random.seed(42)
    exp1 = run_exp1()
    exp3 = run_exp3()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"contradiction_poisoning_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "exp1": exp1, "exp3": exp3}, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*72}\n  Saved: {path}\n{'='*72}")

if __name__ == "__main__":
    main()
