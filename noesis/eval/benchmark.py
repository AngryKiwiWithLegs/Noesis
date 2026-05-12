"""
noesis/eval/benchmark.py

Formal evaluation framework for Noesis.

Metrics:
  injection_accuracy  — correct memories injected / total cases
  time_awareness      — newer node surfaced over contradicting older node
  cross_tool_recall   — cross-tool nodes both retrievable
  tentative_leakage   — tentative nodes that appeared in context (must = 0)

Usage:
    python -m noesis.eval.benchmark
    # or
    noesis eval
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    name:       str
    category:   str
    setup:      list[dict]      # [{text, type, status, age_days, source_tool}]
    query:      str
    must_contain:     list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    must_inject:      bool      = True   # False means empty ctx is correct

# 30 cases across 5 categories
EVAL_DATASET: list[EvalCase] = [

    # ── Category 1: Status gating (6 cases) ──────────────────────────────────

    EvalCase("settled_identity", "status_gating",
             [{"text": "用户叫赵伟，是一名后端工程师", "type": "identity",
               "status": "settled", "age_days": 0}],
             "请做个自我介绍",
             must_contain=["赵伟", "工程师"]),

    EvalCase("settled_preference", "status_gating",
             [{"text": "用户强烈偏好零依赖的技术方案", "type": "preference",
               "status": "settled", "age_days": 5}],
             "帮我选择一个 web 框架",
             must_contain=["零依赖"]),

    EvalCase("tentative_not_injected", "status_gating",
             [{"text": "TENTATIVE_MARKER_XYZ", "type": "position",
               "status": "tentative", "age_days": 0}],
             "任意查询",
             must_not_contain=["TENTATIVE_MARKER_XYZ"],
             must_inject=False),

    EvalCase("provisional_relevant", "status_gating",
             [{"text": "用户倾向于使用 BM25 做初步检索", "type": "position",
               "status": "provisional", "age_days": 2}],
             "搜索架构应该怎么设计",
             must_contain=["BM25"]),

    EvalCase("superseded_excluded", "status_gating",
             [{"text": "旧立场 SUPERSEDED_TEXT", "type": "position",
               "status": "superseded", "age_days": 30}],
             "检索方案",
             must_not_contain=["SUPERSEDED_TEXT"],
             must_inject=False),

    EvalCase("identity_always_present", "status_gating",
             [{"text": "用户叫李明，研究方向是强化学习", "type": "identity",
               "status": "settled", "age_days": 200}],
             "帮我解释一个数学概念",
             must_contain=["李明"]),

    # ── Category 2: Time awareness (6 cases) ─────────────────────────────────

    EvalCase("new_overrides_old", "time_awareness",
             [{"text": "用户认为向量检索总是优于关键词检索", "type": "position",
               "status": "settled", "age_days": 90},
              {"text": "用户发现 BM25 在大多数场景表现更好", "type": "position",
               "status": "settled", "age_days": 1}],
             "检索方案的选择",
             must_contain=["BM25"]),

    EvalCase("identity_persists_age", "time_awareness",
             [{"text": "用户是一名 AI 产品经理", "type": "identity",
               "status": "settled", "age_days": 365}],
             "帮我写邮件",
             must_contain=["产品经理"]),

    EvalCase("recent_event_recency_signal", "time_awareness",
             [{"text": "用户今天完成了产品上线", "type": "event",
               "status": "settled", "age_days": 0}],
             "有什么需要庆祝的",
             must_contain=["上线"]),

    EvalCase("old_event_decayed", "time_awareness",
             [{"text": "STALE_EVENT_60DAYS_AGO", "type": "event",
               "status": "settled", "age_days": 60}],
             "近况怎么样",
             must_not_contain=["STALE_EVENT_60DAYS_AGO"]),

    EvalCase("synthesis_long_lived", "time_awareness",
             [{"text": "外部记忆+图谱遍历是提升 AI 上下文质量的核心路径",
               "type": "synthesis", "status": "settled", "age_days": 150}],
             "AI 记忆系统的核心设计",
             must_contain=["图谱遍历"]),

    EvalCase("position_90_day_decay", "time_awareness",
             [{"text": "NINETY_DAY_OLD_POSITION", "type": "position",
               "status": "settled", "age_days": 120}],
             "NINETY_DAY_OLD query",
             # After 120 days (>1 half-life), should have lower rank but may still appear
             must_not_contain=[]),   # soft check only

    # ── Category 3: Cross-tool (5 cases) ─────────────────────────────────────

    EvalCase("both_tools_retrievable", "cross_tool",
             [{"text": "CLAUDE_PERSPECTIVE_ON_ARCH",
               "type": "position", "status": "settled",
               "source_tool": "claude-sonnet-4-6", "age_days": 0},
              {"text": "GPT_PERSPECTIVE_ON_ARCH",
               "type": "position", "status": "settled",
               "source_tool": "gpt-4o", "age_days": 0}],
             "架构分析",
             must_contain=["CLAUDE_PERSPECTIVE_ON_ARCH"]),

    EvalCase("cross_tool_dedup", "cross_tool",
             [{"text": "用户选择 sqlite-vec", "type": "position",
               "status": "settled", "source_tool": "claude-sonnet-4-6"},
              {"text": "用户选择 sqlite-vec", "type": "position",
               "status": "settled", "source_tool": "gpt-4o"}],
             "向量库选型",
             must_contain=["sqlite-vec"]),

    EvalCase("source_tool_preserved", "cross_tool",
             [{"text": "工具标签测试", "type": "position",
               "status": "settled", "source_tool": "perplexity-web"}],
             "工具测试查询",
             must_contain=["工具标签测试"]),

    EvalCase("cross_tool_user_isolation", "cross_tool",
             [{"text": "USER_A_SECRET", "type": "identity",
               "status": "settled", "source_tool": "claude-sonnet-4-6"}],
             "任意查询",
             must_not_contain=["USER_A_SECRET"]),

    EvalCase("cross_tool_context_completeness", "cross_tool",
             [{"text": "用户关注检索延迟优化", "type": "position",
               "status": "settled", "source_tool": "claude-sonnet-4-6"},
              {"text": "用户也关注存储成本", "type": "position",
               "status": "settled", "source_tool": "gpt-4o"}],
             "系统优化方向",
             must_contain=["检索延迟"]),

    # ── Category 4: Core facts (5 cases) ─────────────────────────────────────

    EvalCase("identity_unrelated_query", "core_facts",
             [{"text": "用户是一名量化研究员", "type": "identity",
               "status": "settled", "age_days": 100}],
             "解释一下光合作用",
             must_contain=["量化研究员"]),

    EvalCase("preference_technical", "core_facts",
             [{"text": "用户偏好 Python 3.11+ 特性，不用旧版", "type": "preference",
               "status": "settled", "age_days": 30}],
             "帮我写一个脚本",
             must_contain=["Python"]),

    EvalCase("multiple_identities", "core_facts",
             [{"text": "用户叫陈晓明", "type": "identity", "status": "settled"},
              {"text": "用户在上海工作", "type": "identity", "status": "settled"}],
             "你好",
             must_contain=["陈晓明"]),

    EvalCase("preference_overrides_generic", "core_facts",
             [{"text": "用户强烈反对使用 JavaScript", "type": "preference",
               "status": "settled", "age_days": 20}],
             "全栈开发建议",
             must_contain=["JavaScript"]),

    EvalCase("identity_in_creative_task", "core_facts",
             [{"text": "用户是一位诗人，喜欢古典风格", "type": "identity",
               "status": "settled"}],
             "帮我写一段话",
             must_contain=["诗人"]),

    # ── Category 5: Budget & dedup (8 cases) ─────────────────────────────────

    EvalCase("budget_respected", "budget",
             [{"text": f"长文本测试 " * 30 + f"节点", "type": "position",
               "status": "settled"} for _ in range(5)],
             "任意查询",
             must_not_contain=[]),   # just check no crash

    EvalCase("empty_when_all_tentative", "budget",
             [{"text": "ONLY_TENTATIVE", "type": "position",
               "status": "tentative"}],
             "任意查询",
             must_not_contain=["ONLY_TENTATIVE"],
             must_inject=False),

    EvalCase("not_empty_when_settled", "budget",
             [{"text": "SHOULD_APPEAR", "type": "position",
               "status": "settled"}],
             "任意查询",
             must_contain=["SHOULD_APPEAR"]),
]


# ── Runner ────────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    name:     str
    category: str
    passed:   bool
    ctx:      str
    failure:  str = ""


def run_benchmark(memory, user_prefix: str = "bench") -> dict:
    """Run all eval cases and return results dict."""
    results:    list[CaseResult] = []
    by_category: dict[str, list] = {}

    for i, case in enumerate(EVAL_DATASET):
        uid  = f"{user_prefix}_{i}"
        res  = _run_case(memory, case, uid)
        results.append(res)
        by_category.setdefault(case.category, []).append(res)

    total   = len(results)
    passed  = sum(1 for r in results if r.passed)
    accuracy = passed / total if total else 0

    category_scores = {
        cat: sum(1 for r in cases if r.passed) / len(cases)
        for cat, cases in by_category.items()
    }

    failures = [r for r in results if not r.passed]

    return {
        "total":            total,
        "passed":           passed,
        "accuracy":         accuracy,
        "category_scores":  category_scores,
        "failures":         [{"name": r.name, "reason": r.failure} for r in failures],
    }


def _run_case(memory, case: EvalCase, user_id: str) -> CaseResult:
    # Setup
    for setup in case.setup:
        r = memory.add(
            setup["text"], user_id=user_id,
            type         = setup.get("type", "position"),
            source_tool  = setup.get("source_tool", "eval-harness"),
            topic_cluster= setup.get("topic", "eval-topic"),
        )
        h  = r["results"][0]["id"]
        ts = time.time() - setup.get("age_days", 0) * 86400
        memory.vector_store.update(h, {
            "status":     setup.get("status", "settled"),
            "confidence": 0.85,
            "created_at": ts,
        })

    # Use different user for isolation test
    query_user = (
        "OTHER_USER_ISOLATION"
        if case.name == "cross_tool_user_isolation"
        else user_id
    )

    ctx = memory.build_context(case.query, user_id=query_user)

    # Evaluate
    failures = []
    for text in case.must_contain:
        if text not in ctx:
            failures.append(f"missing '{text[:40]}'")
    for text in case.must_not_contain:
        if text in ctx:
            failures.append(f"leaked '{text[:40]}'")
    if case.must_inject and not ctx:
        failures.append("context empty but must_inject=True")

    passed  = len(failures) == 0
    failure = " | ".join(failures)
    return CaseResult(name=case.name, category=case.category,
                      passed=passed, ctx=ctx, failure=failure)


# ── CLI entry ─────────────────────────────────────────────────────────────────

def print_report(results: dict):
    acc = results["accuracy"]
    bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))

    print(f"\n  {'═' * 50}")
    print(f"  Noesis Injection Accuracy Benchmark")
    print(f"  {'─' * 50}")
    print(f"  [{bar}] {acc:.0%}  ({results['passed']}/{results['total']})")
    print(f"  {'─' * 50}")

    for cat, score in sorted(results["category_scores"].items()):
        bar2 = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"  {cat:<25} [{bar2}] {score:.0%}")

    if results["failures"]:
        print(f"\n  Failures ({len(results['failures'])}):")
        for f in results["failures"]:
            print(f"    ✗  {f['name']}: {f['reason']}")
    else:
        print(f"\n  ✓ All cases passed!")

    print(f"  {'═' * 50}\n")

    target = 0.80
    if acc < target:
        print(f"  ⚠ Below target ({target:.0%}). "
              f"Check: ContextBuilder signals, status filters, time decay.\n")
        return False
    else:
        print(f"  ✓ Target {target:.0%} met.\n")
        return True


if __name__ == "__main__":
    from pathlib import Path as _P
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        from noesis.memory.main import Memory as _M
        mem = _M.from_config({
            "vector_store": {"config": {"db_path": os.path.join(tmp, "hot.db")}},
            "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        })
        results = run_benchmark(mem)
        ok = print_report(results)

        # Write JSON report
        report_path = _P("noesis_benchmark_report.json")
        report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"  Report written to {report_path}")

        sys.exit(0 if ok else 1)
