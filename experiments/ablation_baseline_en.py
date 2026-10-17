#!/usr/bin/env python3
"""
English Ablation + Baseline experiment (50 queries, retrieval-layer)
=====================================================================
Scales the ablation study to 50 queries across a richer memory pool,
entirely in English. No LLM calls — purely retrieval-layer evaluation.

Configs evaluated (same as before, English test set):
  Noesis-full / no-core / no-recency / no-semantic / no-gating  (ablation)
  naive-RAG / recent-window / random                           (baselines)

Every Noesis variant runs the real ContextBuilder with exactly one component
switched off, so signal order, dedup and the token budget stay the same.
Baselines draw from every stored memory, tentative ones included, because
they stand for systems without a gate.

Scoring (October 2026 revision): a retrieved memory counts as relevant only if
it IS a ground-truth memory (exact text identity; every pool text is unique).
Two views of each config:
  @5      the first 5 memories, in injection order (Recall@5, MRR)
  context everything the config would inject (recall, items, tentative leak)
Queries that return fewer than 5 memories are scored on what they return.
Status and confidence are assigned when the store is built, so this suite
tests whether retrieval respects the labels, not whether the lifecycle
derives them from conversation.
"""
from __future__ import annotations
import os, sys, json, time, random, tempfile
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable
from unittest.mock import patch

NOESIS_DIR = "/Users/mac27ssd/Noesis"
RESULTS_DIR = Path("/Users/mac27ssd/ZCodeProject/noesis_experiment/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, NOESIS_DIR)


@dataclass
class MemoryItem:
    text: str
    type: str
    status: str
    confidence: float
    age_days: int = 0
    topic_cluster: str = "general"

@dataclass
class TestCase:
    query: str
    relevant_texts: list[str]
    description: str

# ── Richer memory pool (28 items incl. 5 noise) ─────────────────────────────────
MEMORY_POOL = [
    # identity
    MemoryItem("My name is John, I am a backend engineer", "identity", "settled", 0.90, 0, "identity"),
    MemoryItem("I am a native English speaker, working in Berlin", "identity", "settled", 0.85, 30, "identity"),
    MemoryItem("I have 8 years of experience in distributed systems", "identity", "settled", 0.80, 40, "identity"),
    # preference
    MemoryItem("I prefer PostgreSQL over MySQL for its JSON support", "preference", "settled", 0.88, 5, "database"),
    MemoryItem("I decided to use sqlite-vec for vector search, lighter than FAISS", "preference", "settled", 0.82, 2, "vector"),
    MemoryItem("I prefer Python for backend, cleaner than Java", "preference", "provisional", 0.65, 1, "language"),
    MemoryItem("I like using Vim keybindings, faster than a mouse", "preference", "provisional", 0.60, 10, "tools"),
    MemoryItem("I prefer Grafana for monitoring, better than Prometheus UI alone", "preference", "settled", 0.75, 8, "infra"),
    MemoryItem("I decided to use Tailwind CSS for styling, faster than handwritten CSS", "preference", "provisional", 0.55, 3, "frontend"),
    MemoryItem("I prefer TypeScript over JavaScript", "preference", "settled", 0.70, 12, "frontend"),
    MemoryItem("I like using Docker for dev environments, I dislike local installs", "preference", "settled", 0.78, 4, "tools"),
    # position
    MemoryItem("I think microservices are better than monoliths for fast iteration", "position", "settled", 0.75, 7, "architecture"),
    MemoryItem("I think Rust's memory safety is superior to C++", "position", "provisional", 0.55, 3, "language"),
    MemoryItem("I believe GraphQL is better than REST for complex frontends", "position", "settled", 0.72, 9, "frontend"),
    MemoryItem("I think Kafka is better than RabbitMQ for high throughput", "position", "provisional", 0.50, 6, "infra"),
    # event
    MemoryItem("Yesterday I fixed a concurrency bug that had plagued the team for two weeks", "event", "provisional", 0.50, 0, "work"),
    MemoryItem("Last week I migrated the database from MySQL to PostgreSQL", "event", "settled", 0.80, 5, "database"),
    MemoryItem("Last month I attended the QCon tech conference", "event", "provisional", 0.45, 35, "general"),
    MemoryItem("I just deployed a new payment system to production", "event", "settled", 0.70, 1, "work"),
    # fact
    MemoryItem("Our team has 5 backend and 2 frontend engineers", "fact", "settled", 0.90, 60, "team"),
    MemoryItem("Production runs on AWS us-east-1", "fact", "settled", 0.85, 90, "infra"),
    MemoryItem("Our daily active users are about 500 thousand", "fact", "settled", 0.80, 45, "team"),
    MemoryItem("The codebase has over 1 million lines of code", "fact", "settled", 0.75, 70, "team"),
    # noise (tentative — must NOT be injected)
    MemoryItem("Maybe I should try Bun instead of Node", "preference", "tentative", 0.25, 1, "language"),
    MemoryItem("Not sure if we should adopt Kafka", "position", "tentative", 0.20, 2, "infra"),
    MemoryItem("Just chatting, the weather is nice today", "fact", "tentative", 0.10, 0, "general"),
    MemoryItem("Perhaps React Native could work for the mobile app", "preference", "tentative", 0.15, 3, "frontend"),
    MemoryItem("I might be interested in learning Go", "position", "tentative", 0.20, 1, "language"),
]

# ── 50 test queries with ground truth ──────────────────────────────────────────
TEST_CASES = [
    # Identity (10)
    TestCase("What is my name?", ["My name is John, I am a backend engineer"], "identity-name"),
    TestCase("Who am I?", ["My name is John, I am a backend engineer"], "identity-who"),
    TestCase("What is my job title?", ["My name is John, I am a backend engineer"], "identity-job"),
    TestCase("Where do I work?", ["I am a native English speaker, working in Berlin"], "identity-location"),
    TestCase("How many years of experience do I have?", ["I have 8 years of experience in distributed systems"], "identity-exp"),
    TestCase("What is my specialty?", ["I have 8 years of experience in distributed systems"], "identity-specialty"),
    TestCase("What language do I speak natively?", ["I am a native English speaker, working in Berlin"], "identity-lang"),
    TestCase("Tell me about my background", ["My name is John, I am a backend engineer", "I have 8 years of experience in distributed systems"], "identity-bg"),
    TestCase("Am I a frontend or backend engineer?", ["My name is John, I am a backend engineer"], "identity-role"),
    TestCase("What city do I live in?", ["I am a native English speaker, working in Berlin"], "identity-city"),
    # Preference (15)
    TestCase("Which database do I prefer?", ["I prefer PostgreSQL over MySQL for its JSON support", "Last week I migrated the database from MySQL to PostgreSQL"], "pref-db"),
    TestCase("PostgreSQL or MySQL for me?", ["I prefer PostgreSQL over MySQL for its JSON support"], "pref-db-kw"),
    TestCase("What do I use for vector search?", ["I decided to use sqlite-vec for vector search, lighter than FAISS"], "pref-vector"),
    TestCase("sqlite-vec or FAISS?", ["I decided to use sqlite-vec for vector search, lighter than FAISS"], "pref-vector-kw"),
    TestCase("What programming language do I prefer?", ["I prefer Python for backend, cleaner than Java"], "pref-lang"),
    TestCase("Python or Java for me?", ["I prefer Python for backend, cleaner than Java"], "pref-lang-kw"),
    TestCase("What editor keybindings do I use?", ["I like using Vim keybindings, faster than a mouse"], "pref-editor"),
    TestCase("What monitoring tool do I like?", ["I prefer Grafana for monitoring, better than Prometheus UI alone"], "pref-monitor"),
    TestCase("What CSS framework do I use?", ["I decided to use Tailwind CSS for styling, faster than handwritten CSS"], "pref-css"),
    TestCase("TypeScript or JavaScript?", ["I prefer TypeScript over JavaScript"], "pref-ts"),
    TestCase("How do I handle dev environments?", ["I like using Docker for dev environments, I dislike local installs"], "pref-docker"),
    TestCase("Do I like local dependency installs?", ["I like using Docker for dev environments, I dislike local installs"], "pref-docker-neg"),
    TestCase("Grafana or Prometheus?", ["I prefer Grafana for monitoring, better than Prometheus UI alone"], "pref-monitor-kw"),
    TestCase("What do I think of handwritten CSS?", ["I decided to use Tailwind CSS for styling, faster than handwritten CSS"], "pref-css-kw"),
    TestCase("Vim or mouse?", ["I like using Vim keybindings, faster than a mouse"], "pref-editor-kw"),
    # Position (10)
    TestCase("What is my view on microservices?", ["I think microservices are better than monoliths for fast iteration"], "pos-micro"),
    TestCase("Microservices or monoliths?", ["I think microservices are better than monoliths for fast iteration"], "pos-micro-kw"),
    TestCase("What do I think about Rust vs C++?", ["I think Rust's memory safety is superior to C++"], "pos-rust"),
    TestCase("Rust or C++ in my opinion?", ["I think Rust's memory safety is superior to C++"], "pos-rust-kw"),
    TestCase("GraphQL or REST for me?", ["I believe GraphQL is better than REST for complex frontends"], "pos-graphql"),
    TestCase("What is my view on GraphQL?", ["I believe GraphQL is better than REST for complex frontends"], "pos-graphql-kw"),
    TestCase("Kafka or RabbitMQ?", ["I think Kafka is better than RabbitMQ for high throughput"], "pos-kafka"),
    TestCase("What do I think about message queues?", ["I think Kafka is better than RabbitMQ for high throughput"], "pos-mq"),
    TestCase("Do I prefer monoliths?", ["I think microservices are better than monoliths for fast iteration"], "pos-mono-neg"),
    TestCase("How do I feel about type safety?", ["I prefer TypeScript over JavaScript"], "pos-type"),
    # Event (5)
    TestCase("What bug did I fix recently?", ["Yesterday I fixed a concurrency bug that had plagued the team for two weeks"], "evt-bug"),
    TestCase("What did I migrate last week?", ["Last week I migrated the database from MySQL to PostgreSQL"], "evt-migrate"),
    TestCase("Did I attend any conferences?", ["Last month I attended the QCon tech conference"], "evt-conf"),
    TestCase("What did I just deploy?", ["I just deployed a new payment system to production"], "evt-deploy"),
    TestCase("Database migration", ["Last week I migrated the database from MySQL to PostgreSQL"], "evt-migrate-kw"),
    # Fact (10)
    TestCase("How big is our team?", ["Our team has 5 backend and 2 frontend engineers"], "fact-team"),
    TestCase("How many engineers are on the backend?", ["Our team has 5 backend and 2 frontend engineers"], "fact-backend"),
    TestCase("Where does production run?", ["Production runs on AWS us-east-1"], "fact-aws"),
    TestCase("Which AWS region do we use?", ["Production runs on AWS us-east-1"], "fact-aws-region"),
    TestCase("How many daily active users do we have?", ["Our daily active users are about 500 thousand"], "fact-dau"),
    TestCase("What is our user base size?", ["Our daily active users are about 500 thousand"], "fact-users"),
    TestCase("How large is the codebase?", ["The codebase has over 1 million lines of code"], "fact-loc"),
    TestCase("AWS deployment region", ["Production runs on AWS us-east-1"], "fact-aws-kw"),
    TestCase("Team composition", ["Our team has 5 backend and 2 frontend engineers"], "fact-team-kw"),
    TestCase("Lines of code", ["The codebase has over 1 million lines of code"], "fact-loc-kw"),
]


def _build_store(pool, tmp_path):
    import logging; logging.disable(logging.CRITICAL)
    from noesis.memory.main import Memory
    m = Memory.from_config({
        "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        "cold_store":   {"config": {"vault_path": str(tmp_path / "vault")}},
    })
    now = time.time()
    for item in pool:
        r = m.add(item.text, user_id="abl_en", type=item.type,
                  source_tool="eval", topic_cluster=item.topic_cluster)
        hid = r["results"][0]["id"]
        m.vector_store.update(hid, {
            "status": item.status, "confidence": item.confidence,
        })
        # update() does not accept created_at, so set the age directly
        # (before October 2026 every age was silently dropped).
        m.vector_store._con.execute("UPDATE items SET created_at=? WHERE hash_id=?",
                                    [now - item.age_days * 86400, hid])
    m.vector_store._con.commit()
    return m


# Exact identity: every pool text is unique, and the store returns it verbatim.
POOL_INDEX = {m.text: i for i, m in enumerate(MEMORY_POOL)}
TENTATIVE = {i for i, m in enumerate(MEMORY_POOL) if m.status == "tentative"}


def _ids(memories):
    """Pool index of each retrieved memory (None if the text is not in the pool)."""
    return [POOL_INDEX.get(m.get("text", "").strip()) for m in memories]


def evaluate_config(name, retrieve_fn, cases, k=5):
    results = []
    for tc in cases:
        context = _ids(retrieve_fn(tc.query))
        top = context[:k]
        truth = {POOL_INDEX[t] for t in tc.relevant_texts}
        rank = next((r for r, i in enumerate(top, 1) if i in truth), None)
        results.append({"query": tc.query,
                        "recall": len(truth & set(top)) / len(truth),
                        "mrr": 1.0 / rank if rank else 0.0,
                        "context_recall": len(truth & set(context)) / len(truth),
                        "context_items": len(context),
                        "tentative_leak": sum(1 for i in context if i in TENTATIVE),
                        "unknown_items": sum(1 for i in context if i is None)})
    avg = lambda key: round(sum(r[key] for r in results)/len(results), 4)
    return {"config": name, "recall@5": avg("recall"), "mrr": avg("mrr"),
            "context_recall": avg("context_recall"),
            "context_items": avg("context_items"),
            "tentative_leak_total": sum(r["tentative_leak"] for r in results),
            "context_slots_total": sum(r["context_items"] for r in results),
            "unknown_items_total": sum(r["unknown_items"] for r in results),
            "detail": results}


def make_configs(memory):
    import noesis.context.builder as builder_mod
    builder = memory._ctx_builder
    retriever = memory._retriever
    vs = memory.vector_store

    def build(q, **replaced):
        """The real ContextBuilder, with the named signal functions replaced."""
        with patch.multiple(builder_mod, **replaced) if replaced else nullcontext():
            return _parse_ctx(builder.build(q, user_id="abl_en", budget_tokens=2000))

    off = lambda *a, **kw: []
    # The same three signals with their status filters removed.
    ungated = {
        "semantic_signal":  lambda q, u, r, top_k=8: r.search(q, user_id=u, top_k=top_k),
        "recency_signal":   lambda u, store, n=3: _everything(store, u)[:n],
        "core_fact_signal": lambda u, store, top_k=3: sorted(
            (m for m in store.get_all(u) if m["type"] in ("identity", "preference")),
            key=lambda m: (-m["confidence"], -m["created_at"]))[:top_k],
    }
    configs = {}
    configs["Noesis-full"] = lambda q: build(q)
    configs["Noesis-no-core"] = lambda q: build(q, core_fact_signal=off)
    configs["Noesis-no-recency"] = lambda q: build(q, recency_signal=off)
    configs["Noesis-no-semantic"] = lambda q: build(q, semantic_signal=off)
    configs["Noesis-no-gating"] = lambda q: build(q, **ungated)
    configs["naive-RAG"] = lambda q: retriever.search(q, user_id="abl_en", top_k=5)
    configs["recent-window"] = lambda q: _everything(vs, "abl_en")[:5]
    configs["random"] = lambda q: random.sample(_everything(vs, "abl_en"), 5)
    return configs

def _parse_ctx(ctx_str):
    if not ctx_str: return []
    nodes = []
    for line in ctx_str.split("\n"):
        if line.startswith("- ["):
            try:
                be = line.index("]"); meta = line[3:be]; text = line[be+2:].strip()
                t, s = meta.split("·"); nodes.append({"text": text, "type": t, "status": s})
            except: pass
    return nodes

def _everything(vs, user_id):
    """Every stored memory, tentative included, newest first."""
    return sorted(vs.get_all(user_id), key=lambda m: -m["created_at"])


def main():
    random.seed(42)
    print("="*72)
    print("  English Ablation + Baseline (50 queries, retrieval-layer)")
    print("="*72)
    print(f"  Memory pool: {len(MEMORY_POOL)} items (incl. {sum(1 for m in MEMORY_POOL if m.status=='tentative')} noise)")
    print(f"  Test queries: {len(TEST_CASES)}")
    print("="*72)
    tmp = Path(tempfile.mkdtemp(prefix="abl_en_"))
    print(f"\nBuilding memory store...")
    memory = _build_store(MEMORY_POOL, tmp)
    print(f"Ready.\n")
    configs = make_configs(memory)
    all_results = []
    for name, fn in configs.items():
        print(f"> Eval: {name}")
        res = evaluate_config(name, fn, TEST_CASES, k=5)
        all_results.append(res)
        print(f"  Recall@5={res['recall@5']:.1%}  MRR={res['mrr']:.3f}  context recall={res['context_recall']:.1%}  "
              f"items={res['context_items']:.1f}  leak={res['tentative_leak_total']}/{res['context_slots_total']}\n")
        if res["unknown_items_total"]:
            print(f"  ! {res['unknown_items_total']} retrieved texts not found in the pool\n")
    all_results.sort(key=lambda x: x["recall@5"], reverse=True)
    print("="*72)
    print("  Final ranking (by Recall@5)")
    print("="*72)
    print(f"  {'Config':<20s} {'Recall@5':>10s} {'MRR':>8s} {'Ctx rec':>8s} {'Items':>6s} {'Leak':>9s}")
    print(f"  {'-'*66}")
    for r in all_results:
        print(f"  {r['config']:<20s} {r['recall@5']:>9.1%} {r['mrr']:>8.3f} {r['context_recall']:>8.1%} "
              f"{r['context_items']:>6.1f} {r['tentative_leak_total']:>4d}/{r['context_slots_total']:<4d}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"ablation_baseline_en_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"experiment":"ablation_baseline_en","language":"en","timestamp":ts,
                   "memory_pool_size":len(MEMORY_POOL),"num_queries":len(TEST_CASES),
                   "results":all_results}, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {path}")

if __name__ == "__main__":
    main()
