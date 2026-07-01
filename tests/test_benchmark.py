"""
tests/test_benchmark.py

Runs the formal 30-case evaluation benchmark.
This is the Week 6 gate — must pass before publishing.

Marked as slow because it loads the embedding model and runs 30 eval cases.

Run:
    pytest tests/test_benchmark.py -v           # all benchmark tests
    pytest tests/test_benchmark.py -v -m slow   # same thing
    pytest tests/test_benchmark.py -k "not slow"  # skip benchmark in CI

The final accuracy score goes in README.md.
"""
import json
import pytest
import tempfile
import os

from noesis.memory.main import Memory
from noesis.eval.benchmark import run_benchmark, print_report, EVAL_DATASET


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_memory(tmp_path):
    """Create a fresh Memory on an isolated temp DB.

    NOTE: intentionally built WITHOUT the consolidation pipeline
    (from_config, not from_config_file). The benchmark tests retrieval
    and injection behaviour by manually setting node status via
    vector_store.update(), so a background pipeline would race with
    those manual updates. Each test gets its own isolated store, so no
    cross-test pollution.
    """
    return Memory.from_config({
        "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
    })


@pytest.fixture
def bench_mem(tmp_path):
    """Per-test isolated Memory instance (no shared state between tests).

    Previously this was scope='module', which let earlier category tests
    pollute the shared store and made test_full_benchmark_accuracy flaky
    (36% under pytest vs 92% standalone). Isolating per test fixes that.
    """
    return _make_memory(tmp_path)


# ── Dataset sanity ────────────────────────────────────────────────────────────

def test_dataset_not_empty():
    assert len(EVAL_DATASET) >= 20, "Benchmark needs at least 20 cases"


def test_dataset_categories_covered():
    categories = {c.category for c in EVAL_DATASET}
    required   = {"status_gating", "time_awareness", "cross_tool",
                  "core_facts", "budget"}
    missing    = required - categories
    assert not missing, f"Missing categories: {missing}"


def test_all_cases_have_query():
    for case in EVAL_DATASET:
        assert case.query.strip(), f"Case '{case.name}' has empty query"


def test_all_cases_have_setup():
    for case in EVAL_DATASET:
        assert case.setup, f"Case '{case.name}' has empty setup"


# ── Per-category accuracy ─────────────────────────────────────────────────────

@pytest.mark.slow
def test_status_gating_accuracy(bench_mem):
    """Status gating cases: settled injected, tentative never injected."""
    cases   = [c for c in EVAL_DATASET if c.category == "status_gating"]
    results = _run_cases(bench_mem, cases, "sg")
    passed  = sum(1 for r in results if r["passed"])
    acc     = passed / len(results)
    print(f"\nStatus gating: {passed}/{len(results)} = {acc:.0%}")
    assert acc >= 0.80, f"Status gating accuracy {acc:.0%} < 80%"


@pytest.mark.slow
def test_time_awareness_accuracy(bench_mem):
    """Time-aware retrieval: new nodes rank above old contradicting ones."""
    cases   = [c for c in EVAL_DATASET if c.category == "time_awareness"]
    results = _run_cases(bench_mem, cases, "ta")
    passed  = sum(1 for r in results if r["passed"])
    acc     = passed / len(results)
    print(f"\nTime awareness: {passed}/{len(results)} = {acc:.0%}")
    assert acc >= 0.70, f"Time awareness accuracy {acc:.0%} < 70%"


@pytest.mark.slow
def test_core_facts_accuracy(bench_mem):
    """Core facts: identity and preference always present in context."""
    cases   = [c for c in EVAL_DATASET if c.category == "core_facts"]
    results = _run_cases(bench_mem, cases, "cf")
    passed  = sum(1 for r in results if r["passed"])
    acc     = passed / len(results)
    print(f"\nCore facts: {passed}/{len(results)} = {acc:.0%}")
    assert acc >= 0.85, f"Core facts accuracy {acc:.0%} < 85%"


# ── Full benchmark ────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_full_benchmark_accuracy(bench_mem):
    """
    Runs all 30 eval cases.
    Accuracy >= 80% is the Week 6 gate.
    The number printed here goes in README.md.
    """
    results = run_benchmark(bench_mem, user_prefix="full_bench")
    ok      = print_report(results)

    # Write report artifact
    report_path = "noesis_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Report → {report_path}")

    assert ok, (
        f"Benchmark accuracy {results['accuracy']:.0%} below 80% target.\n"
        f"Failures: {[f['name'] for f in results['failures']]}"
    )


# ── Tentative leakage (critical, must be 0%) ──────────────────────────────────

@pytest.mark.slow
def test_zero_tentative_leakage(bench_mem):
    """
    Tentative nodes must NEVER appear in build_context output.
    This is an absolute requirement — any leakage is a critical bug.
    """
    from noesis.eval.benchmark import _run_case, EvalCase
    import time

    leak_cases = [c for c in EVAL_DATASET
                  if "TENTATIVE" in str(c.must_not_contain)
                  or c.name in ("tentative_not_injected", "empty_when_all_tentative")]

    leaks = 0
    for i, case in enumerate(leak_cases):
        uid = f"leak_test_{i}"
        res = _run_case(bench_mem, case, uid)
        if not res.passed:
            leaks += 1
            print(f"  LEAK: {case.name} — {res.failure}")

    assert leaks == 0, f"{leaks} tentative node(s) leaked into context!"


# ── Helper ────────────────────────────────────────────────────────────────────

def _run_cases(memory, cases, prefix: str) -> list[dict]:
    import time
    from noesis.eval.benchmark import _run_case
    results = []
    for i, case in enumerate(cases):
        uid = f"{prefix}_{i}"
        res = _run_case(memory, case, uid)
        results.append({"name": case.name, "passed": res.passed,
                         "failure": res.failure})
    return results
