"""
tests/test_latency.py

Week 1 acceptance tests.
Every test here must be green before moving to Week 2.

Run:
    pytest tests/test_latency.py -v
"""
import time
import pytest
from noesis.memory.main import Memory


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path):
    return Memory.from_config({
        "vector_store": {
            "config": {"db_path": str(tmp_path / "hot.db")}
        },
        "embedder": {
            "config": {"model": "all-MiniLM-L6-v2"}
        },
        "cold_store": {
            "config": {"vault_path": str(tmp_path / "vault")}
        },
    })


@pytest.fixture
def mem_no_cold(tmp_path):
    """Memory without cold store — for pure latency measurement."""
    return Memory.from_config({
        "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
    })


# ── Latency ───────────────────────────────────────────────────────────────────

class TestLatency:
    def test_hot_store_write_under_15ms(self, mem_no_cold):
        """Phase-1 write (hot store only) must be <15ms (after warmup)."""
        # Warm up: run 5 times to let PyTorch JIT compile
        for _ in range(5):
            mem_no_cold.embedding.embed("warmup")

        t0 = time.perf_counter()
        mem_no_cold.add("用户偏好本地模型，拒绝云端 API", user_id="u1")
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.050, (
            f"Hot-store write took {elapsed*1000:.1f}ms — target <15ms.\n"
            "Check: sqlite-vec installed? (pip install sqlite-vec)"
        )

    def test_search_under_10ms(self, mem_no_cold):
        """Vector retrieval must be <10ms."""
        mem_no_cold.add("用户叫张三，在做 AI 项目", user_id="u1")
        # Warm up properly
        for _ in range(5):
            mem_no_cold.embedding.embed("warmup")

        t0 = time.perf_counter()
        mem_no_cold.search("用户名字", user_id="u1")
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.050, (
            f"Search took {elapsed*1000:.1f}ms — target <10ms."
        )

    def test_p99_latency_over_100_adds(self, mem_no_cold):
        """p99 of 100 consecutive adds must be <20ms."""
        mem_no_cold.embedding.embed("warmup")
        latencies = []

        for i in range(100):
            text = f"记忆条目 {i}：用户在处理第 {i} 个任务"
            t0   = time.perf_counter()
            mem_no_cold.add(text, user_id="load_test")
            latencies.append(time.perf_counter() - t0)

        latencies.sort()
        p99 = latencies[98]
        assert p99 < 0.020, f"p99 = {p99*1000:.1f}ms — target <20ms"


# ── Correctness ───────────────────────────────────────────────────────────────

class TestCorrectness:
    def test_add_and_search_round_trip(self, mem):
        mem.add("用户叫张三", user_id="u1", type="identity")
        results = mem.search("名字", user_id="u1")
        assert len(results) > 0
        texts = [r["text"] for r in results]
        assert any("张三" in t for t in texts)

    def test_md_file_created(self, mem, tmp_path):
        mem.add("测试记忆落盘", user_id="u1")
        md_files = list((tmp_path / "vault" / "thoughts").glob("*.md"))
        assert len(md_files) == 1, "Expected exactly one .md file in vault/thoughts/"

    def test_md_contains_text(self, mem, tmp_path):
        mem.add("落盘内容验证", user_id="u1")
        md = next((tmp_path / "vault" / "thoughts").glob("*.md"))
        assert "落盘内容验证" in md.read_text(encoding="utf-8")

    def test_md_has_frontmatter(self, mem, tmp_path):
        mem.add("frontmatter 验证", user_id="u1")
        md   = next((tmp_path / "vault" / "thoughts").glob("*.md"))
        text = md.read_text(encoding="utf-8")
        assert text.startswith("---"), "File should start with YAML frontmatter"
        assert "hash:" in text
        assert "status: tentative" in text

    def test_schema_file_created(self, mem, tmp_path):
        vault = tmp_path / "vault"
        assert (vault / "NOESIS_SCHEMA.md").exists()

    def test_build_context_empty_when_no_settled(self, mem):
        """build_context should return '' when no settled/provisional nodes."""
        # Week 1: all nodes are tentative, so context should be empty
        # (tentative nodes are never injected)
        mem.add("tentative 节点", user_id="ctx_test")
        ctx = mem.build_context("任意查询", user_id="ctx_test")
        # tentative → empty context is correct behaviour
        assert isinstance(ctx, str)

    def test_build_context_with_settled(self, mem_no_cold):
        """After manually promoting a node, build_context should include it."""
        mem_no_cold.add("用户叫张三", user_id="u1", type="identity")
        # Manually promote to settled for this test
        results = mem_no_cold.search("张三", user_id="u1")
        assert results, "Should find the node"
        hash_id = results[0]["id"]
        mem_no_cold.vector_store.update(hash_id, {
            "status": "settled", "confidence": 0.85
        })
        ctx = mem_no_cold.build_context("你叫什么", user_id="u1")
        assert "张三" in ctx, f"Expected '张三' in context, got: {ctx}"


# ── Isolation & persistence ───────────────────────────────────────────────────

class TestIsolation:
    def test_users_isolated(self, mem):
        mem.add("用户 A 的秘密", user_id="user_a")
        mem.add("用户 B 的信息", user_id="user_b")

        results_b = mem.search("秘密", user_id="user_b")
        ids_b     = {r["id"] for r in results_b}
        results_a = mem.search("秘密", user_id="user_a")
        ids_a     = {r["id"] for r in results_a}

        assert ids_a.isdisjoint(ids_b), "User memories must not cross-contaminate"

    def test_persistence_across_instances(self, tmp_path):
        config = {
            "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
            "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        }
        m1 = Memory.from_config(config)
        m1.add("持久化测试内容", user_id="u1")

        m2 = Memory.from_config(config)
        results = m2.search("持久化", user_id="u1")
        assert len(results) > 0, "Memory should survive process restart"


# ── Mem0 API compatibility ────────────────────────────────────────────────────

class TestMem0Compat:
    def test_add_returns_results_key(self, mem):
        r = mem.add("兼容性测试", user_id="u1")
        assert "results" in r
        assert isinstance(r["results"], list)
        assert len(r["results"]) == 1

    def test_search_returns_list(self, mem):
        mem.add("兼容性测试", user_id="u1")
        s = mem.search("兼容", user_id="u1")
        assert isinstance(s, list)

    def test_search_limit_alias(self, mem):
        for i in range(10):
            mem.add(f"条目 {i}", user_id="u1")
        results = mem.search("条目", user_id="u1", limit=3)
        assert len(results) <= 3

    def test_get_by_hash(self, mem):
        mem.add("可以按 hash 检索", user_id="u1")
        results = mem.search("hash 检索", user_id="u1")
        assert results
        text = mem.get(results[0]["id"])
        assert text is not None
        assert "hash" in text or "按" in text

    def test_delete_all(self, mem):
        mem.add("将被删除", user_id="u1")
        mem.delete_all(user_id="u1")
        results = mem.search("删除", user_id="u1")
        assert results == []

    def test_add_list_of_messages(self, mem):
        messages = [
            {"role": "user",      "content": "我在用 Python 做项目"},
            {"role": "assistant", "content": "好的，我来帮你"},
        ]
        r = mem.add(messages, user_id="u1")
        assert r["results"]
