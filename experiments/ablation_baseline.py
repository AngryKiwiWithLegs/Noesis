#!/usr/bin/env python3
"""
检索层消融 + 基线对比实验
==========================
核心问题: Noesis 的每个检索组件 (置信度门控/语义/BM25/图扩展/近因/核心事实)
相比朴素基线 (RAG/窗口/随机) 检索质量如何?

评测路线: 检索层 (Recall@k / Precision / MRR)
  给定 query + ground truth (正确记忆的 id), 看各方法能否检索到。

设计:
  1. 构建标注数据集 (记忆 + query + 正确答案 id)
  2. 实现 11 种检索配置:
     - Noesis 变体 (消融): full / 无门控 / 无衰减 / 纯向量 / 无图扩展 / 无BM25 / 无核心事实 / 无近因
     - 基线: 朴素RAG / 最近窗口 / 随机
  3. 每种配置跑全部 query, 输出对比表
"""
from __future__ import annotations
import os, sys, json, time, random, math, sqlite3, tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

NOESIS_DIR = "/Users/mac27ssd/Noesis"
RESULTS_DIR = Path("/Users/mac27ssd/ZCodeProject/noesis_experiment/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, NOESIS_DIR)


# ════════════════════════════════════════════════════════════════════════════════
# 标注数据集: 记忆池 + query (每个 query 标注正确答案的文本关键词)
# ════════════════════════════════════════════════════════════════════════════════
# 设计原则:
#   - 包含相关记忆 (用户真实偏好) 和噪声记忆 (随口/无关的话)
#   - 不同类型 (identity/preference/position/event/fact)
#   - 不同置信度 (settled/provisional/tentative)
#   - query 用不同表述方式问同一信息 (测语义泛化)

@dataclass
class MemoryItem:
    text: str
    type: str
    status: str        # settled / provisional / tentative
    confidence: float
    age_days: int = 0  # 多少天前存的 (用于测衰减)
    topic_cluster: str = "general"

@dataclass
class TestCase:
    query: str
    relevant_texts: list[str]   # ground truth: 这些文本是正确答案
    description: str

# ── 记忆池 (一个用户的知识库) ────────────────────────────────────────────────────
MEMORY_POOL = [
    # === 身份 (core_fact 候选) ===
    MemoryItem("我叫张伟, 是一名后端工程师", "identity", "settled", 0.90, 0, "identity"),
    MemoryItem("用户母语是中文, 工作语言英语", "identity", "settled", 0.85, 30, "identity"),
    # === 偏好 (core_fact 候选) ===
    MemoryItem("我偏好用 PostgreSQL 而不是 MySQL", "preference", "settled", 0.88, 5, "database"),
    MemoryItem("我决定用 sqlite-vec 做向量检索, 比 FAISS 轻量", "preference", "settled", 0.82, 2, "vector"),
    MemoryItem("我偏好用 Python 写后端, 觉得比 Java 简洁", "preference", "provisional", 0.65, 1, "language"),
    MemoryItem("我喜欢用 Vim 快捷键, 觉得比鼠标高效", "preference", "provisional", 0.60, 10, "tools"),
    # === 观点 ===
    MemoryItem("我认为微服务架构比单体更适合快速迭代", "position", "settled", 0.75, 7, "architecture"),
    MemoryItem("我觉得 Rust 的内存安全优于 C++", "position", "provisional", 0.55, 3, "language"),
    # === 事件 (近因信号 + 衰减测试) ===
    MemoryItem("昨天修复了一个困扰团队两周的并发 bug", "event", "provisional", 0.50, 0, "work"),
    MemoryItem("上周把数据库从 MySQL 迁移到了 PostgreSQL", "event", "settled", 0.80, 5, "database"),
    MemoryItem("上个月参加了 QCon 技术大会", "event", "provisional", 0.45, 35, "general"),
    # === 事实 ===
    MemoryItem("我们团队有 5 个后端和 2 个前端", "fact", "settled", 0.90, 60, "team"),
    MemoryItem("生产环境跑在 AWS 的 us-east-1", "fact", "settled", 0.85, 90, "infra"),
    # === 噪声 (随口的/试探性的, 不应被注入) ===
    MemoryItem("也许可以试试 Bun 替代 Node", "preference", "tentative", 0.25, 1, "language"),
    MemoryItem("不确定是不是该用 Kafka", "position", "tentative", 0.20, 2, "infra"),
    MemoryItem("随便聊聊, 今天天气不错", "fact", "tentative", 0.10, 0, "general"),
]

# ── 测试用例 (query + ground truth) ─────────────────────────────────────────────
# 设计原则: query 措辞贴近真实用户, 含可命中的关键词或语义
TEST_CASES = [
    # 身份类 (语义+核心事实双命中)
    TestCase("我叫什么名字",                ["我叫张伟, 是一名后端工程师"], "直问身份-姓名"),
    TestCase("我是做什么的",                ["我叫张伟, 是一名后端工程师"], "直问身份-职业"),
    TestCase("我的职业是什么",                ["我叫张伟, 是一名后端工程师"], "换措辞问身份"),
    # 偏好类 (关键词+语义双命中)
    TestCase("我用什么数据库",              ["我偏好用 PostgreSQL 而不是 MySQL", "上周把数据库从 MySQL 迁移到了 PostgreSQL"], "数据库偏好"),
    TestCase("PostgreSQL 还是 MySQL",       ["我偏好用 PostgreSQL 而不是 MySQL"], "数据库偏好-带关键词"),
    TestCase("sqlite-vec 相关的",           ["我决定用 sqlite-vec 做向量检索, 比 FAISS 轻量"], "向量检索-带关键词"),
    TestCase("我用什么编程语言",            ["我偏好用 Python 写后端, 觉得比 Java 简洁"], "语言偏好"),
    TestCase("Python 还是 Java",            ["我偏好用 Python 写后端, 觉得比 Java 简洁"], "语言偏好-带关键词"),
    # 观点类
    TestCase("我对微服务的看法",            ["我认为微服务架构比单体更适合快速迭代"], "架构观点-带关键词"),
    TestCase("微服务和单体",                ["我认为微服务架构比单体更适合快速迭代"], "架构观点-带关键词2"),
    # 事件类 (关键词命中)
    TestCase("数据库迁移",                  ["上周把数据库从 MySQL 迁移到了 PostgreSQL"], "事件-带关键词"),
    TestCase("并发 bug",                    ["昨天修复了一个困扰团队两周的并发 bug"], "事件-带关键词"),
    # 事实类
    TestCase("团队有多少人",                ["我们团队有 5 个后端和 2 个前端"], "团队规模"),
    TestCase("AWS 部署",                    ["生产环境跑在 AWS 的 us-east-1"], "云部署-带关键词"),
    # 近因查询
    TestCase("最近修复了什么",              ["昨天修复了一个困扰团队两周的并发 bug"], "近因-带关键词"),
]


# ════════════════════════════════════════════════════════════════════════════════
# 检索配置实现
# ════════════════════════════════════════════════════════════════════════════════

def _build_store(pool: list[MemoryItem], tmp_path: Path):
    """构建带数据的 Memory + retriever。"""
    import logging; logging.disable(logging.CRITICAL)
    from noesis.memory.main import Memory
    m = Memory.from_config({
        "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        "cold_store":   {"config": {"vault_path": str(tmp_path / "vault")}},
    })
    now = time.time()
    for item in pool:
        r = m.add(item.text, user_id="ab_test", type=item.type,
                  source_tool="eval", topic_cluster=item.topic_cluster)
        hid = r["results"][0]["id"]
        m.vector_store.update(hid, {
            "status": item.status,
            "confidence": item.confidence,
            "created_at": now - item.age_days * 86400,  # 模拟时间
        })
    return m


# ── 评测函数 ────────────────────────────────────────────────────────────────────
def evaluate_config(
    name: str,
    retrieve_fn: Callable[[str], list[dict]],
    cases: list[TestCase],
    k: int = 5,
) -> dict:
    """评测单个检索配置, 返回指标。"""
    results = []
    for tc in cases:
        retrieved = retrieve_fn(tc.query)[:k]
        retrieved_texts = [r.get("text", "") for r in retrieved]

        # 提取每个 ground truth 的关键词 (去标点, 取 2-4 字核心词)
        def keywords(text):
            import re
            clean = re.sub(r"[，。、,.!?？\s]+", " ", text)
            toks = [t for t in clean.split() if len(t) >= 2]
            return toks

        # Recall@k: 正确答案有多少出现在 top-k (任一关键词命中即可)
        hits = 0
        for truth in tc.relevant_texts:
            kws = keywords(truth)
            for rt in retrieved_texts:
                if any(kw in rt for kw in kws):
                    hits += 1
                    break
        recall = hits / len(tc.relevant_texts)

        # MRR: 第一个命中的倒数排名
        mrr = 0.0
        for rank, rt in enumerate(retrieved_texts, 1):
            matched = False
            for truth in tc.relevant_texts:
                kws = keywords(truth)
                if any(kw in rt for kw in kws):
                    matched = True
                    break
            if matched:
                mrr = 1.0 / rank
                break

        # Precision@k: top-k 里有多少是相关的 (非噪声)
        noise_texts = [m.text for m in MEMORY_POOL if m.status == "tentative"]
        noise_kws = [kw for nt in noise_texts for kw in keywords(nt)]
        relevant_count = 0
        for rt in retrieved_texts:
            is_noise = any(nkw in rt for nkw in noise_kws)
            if not is_noise:
                relevant_count += 1
        precision = relevant_count / len(retrieved) if retrieved else 0

        # tentative 泄漏
        leak = sum(1 for rt in retrieved_texts
                   if any(nkw in rt for nkw in noise_kws))

        results.append({
            "query": tc.query, "recall": recall, "mrr": mrr,
            "precision": precision, "tentative_leak": leak,
            "retrieved_count": len(retrieved),
        })

    avg = lambda key: round(sum(r[key] for r in results) / len(results), 4)
    return {
        "config": name,
        "recall@5":    avg("recall"),
        "mrr":         avg("mrr"),
        "precision@5": avg("precision"),
        "tentative_leak_total": sum(r["tentative_leak"] for r in results),
        "detail": results,
    }


# ════════════════════════════════════════════════════════════════════════════════
# 配置定义: Noesis 变体 (消融) + 基线
# ════════════════════════════════════════════════════════════════════════════════

def make_configs(memory) -> dict[str, Callable]:
    """返回 {配置名: 检索函数}。"""
    vs = memory.vector_store
    emb = memory.embedding
    builder = memory._ctx_builder  # ContextBuilder
    retriever = memory._retriever  # HybridRetriever

    configs = {}

    # ─── Noesis 消融变体 ────────────────────────────────────────────────────────

    # 1. Noesis 完整版 (ContextBuilder.build)
    configs["Noesis-full"] = lambda q: _parse_ctx(
        builder.build(q, user_id="ab_test", budget_tokens=2000))

    # 2. 无核心事实信号 (关 identity/preference 无条件注入)
    configs["Noesis-无核心事实"] = lambda q: _signals_only(
        q, memory, use_core=False, use_recency=True, use_semantic=True)

    # 3. 无近因信号
    configs["Noesis-无近因"] = lambda q: _signals_only(
        q, memory, use_core=True, use_recency=False, use_semantic=True)

    # 4. 无语义信号 (只用近因 + 核心事实)
    configs["Noesis-无语义"] = lambda q: _signals_only(
        q, memory, use_core=True, use_recency=True, use_semantic=False)

    # 5. 无置信度门控 (tentative 也注入)
    configs["Noesis-无门控"] = lambda q: _no_gating(retriever, q)

    # ─── 基线 ──────────────────────────────────────────────────────────────────

    # 6. 朴素 RAG: top-k 向量相似, 全注入 (不过滤 status)
    configs["朴素RAG"] = lambda q: _naive_rag(retriever, q, k=5)

    # 7. 最近窗口: 最近 N 条 (不检索)
    configs["最近窗口"] = lambda q: _recent_window(vs, n=5)

    # 8. 随机检索
    configs["随机"] = lambda q: _random(vs, k=5)

    # 9. 全量注入 (上界参考)
    configs["全量注入"] = lambda q: _all_inject(vs)

    return configs


# ── 配置实现辅助 ─────────────────────────────────────────────────────────────────
def _parse_ctx(ctx_str: str) -> list[dict]:
    """从 ContextBuilder 输出解析回节点列表 (用于评测)。"""
    if not ctx_str:
        return []
    nodes = []
    for line in ctx_str.split("\n"):
        if line.startswith("- ["):
            try:
                bracket_end = line.index("]")
                meta = line[3:bracket_end]
                text = line[bracket_end+2:].strip()
                t, s = meta.split("·")
                nodes.append({"text": text, "type": t, "status": s})
            except (ValueError, IndexError):
                pass
    return nodes

def _signals_only(query, memory, use_core, use_recency, use_semantic):
    """手动控制各 signal 开关。"""
    candidates = []
    from noesis.context.signals import semantic_signal, recency_signal, core_fact_signal
    if use_semantic:
        try:
            candidates += semantic_signal(query, "ab_test", memory._retriever, top_k=8)
        except Exception as e: pass
    if use_recency:
        try:
            candidates += recency_signal("ab_test", memory.vector_store, n=3)
        except Exception as e: pass
    if use_core:
        try:
            candidates += core_fact_signal("ab_test", memory.vector_store, top_k=3)
        except Exception as e: pass
    return candidates

def _no_gating(retriever, query):
    """无置信度门控: 用 retriever 检索, 但不过滤 status。"""
    return retriever.search(query, user_id="ab_test", top_k=5)

def _naive_rag(retriever, query, k=5):
    """朴素 RAG: 用 retriever top-k, 不过滤 status。"""
    return retriever.search(query, user_id="ab_test", top_k=k)

def _recent_window(vs, n=5):
    """最近窗口: 最近 N 条, 不过滤。"""
    return vs.get_recent("ab_test", n=n)

def _random(vs, k=5):
    """随机检索 k 条。"""
    import random as rnd
    all_items = vs.get_recent("ab_test", n=100)
    return rnd.sample(all_items, min(k, len(all_items)))

def _all_inject(vs):
    """全量注入: 所有非 tentative 记忆。"""
    return vs.get_recent("ab_test", n=100,
                         filter={"status": {"$in": ["provisional", "settled"]}})


# ════════════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  消融 + 基线对比实验 (检索层评测)")
    print("=" * 72)
    print(f"  记忆池: {len(MEMORY_POOL)} 条 (含 {sum(1 for m in MEMORY_POOL if m.status=='tentative')} 条噪声)")
    print(f"  测试用例: {len(TEST_CASES)} 个")
    print(f"  指标: Recall@5 / MRR / Precision@5 / Tentative泄漏")
    print("=" * 72)

    tmp = Path(tempfile.mkdtemp(prefix="ablation_"))
    print(f"\n构建记忆库 ({tmp})...")
    memory = _build_store(MEMORY_POOL, tmp)
    print(f"记忆库就绪 ✓\n")

    configs = make_configs(memory)

    all_results = []
    for name, fn in configs.items():
        print(f"▶ 评测: {name}")
        res = evaluate_config(name, fn, TEST_CASES, k=5)
        all_results.append(res)
        print(f"  Recall@5={res['recall@5']:.2%}  MRR={res['mrr']:.3f}  "
              f"Precision={res['precision@5']:.2%}  泄漏={res['tentative_leak_total']}")
        print()

    # 排序输出
    all_results.sort(key=lambda x: x["recall@5"], reverse=True)

    print("=" * 72)
    print("  最终对比表 (按 Recall@5 降序)")
    print("=" * 72)
    print(f"  {'配置':<22s} {'Recall@5':>10s} {'MRR':>8s} {'Prec@5':>8s} {'泄漏':>6s}")
    print(f"  {'-'*56}")
    for r in all_results:
        print(f"  {r['config']:<22s} {r['recall@5']:>9.1%} {r['mrr']:>8.3f} "
              f"{r['precision@5']:>8.1%} {r['tentative_leak_total']:>6d}")

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"ablation_baseline_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "ablation_baseline",
            "timestamp": ts,
            "memory_pool_size": len(MEMORY_POOL),
            "num_test_cases": len(TEST_CASES),
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 {path}")


if __name__ == "__main__":
    main()
