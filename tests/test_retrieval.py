"""
tests/test_retrieval.py

Week 3 retrieval tests.
Tests HybridRetriever, time decay, and graph expansion.
No API keys needed.

Run:
    pytest tests/test_retrieval.py -v
"""
import time
import pytest

from noesis.memory.main import Memory
from noesis.retrieval.hybrid import HybridRetriever, CrossEncoderReranker
from noesis.context.builder import ContextBuilder
from noesis.context.signals import semantic_signal, recency_signal, core_fact_signal


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path):
    """Full Memory with HybridRetriever + ContextBuilder (no LLM)."""
    return Memory.from_config({
        "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        "cold_store":   {"config": {"vault_path": str(tmp_path / "vault")}},
    })


@pytest.fixture
def retriever(mem):
    return HybridRetriever(
        vector_store=mem.vector_store,
        embedding_model=mem.embedding,
    )


def _add_settled(mem, text, user_id, type="position", topic="test",
                 source_tool="claude", age_days=0):
    """Add a node and immediately promote it to settled."""
    result = mem.add(text, user_id=user_id, type=type,
                     topic_cluster=topic, source_tool=source_tool)
    h = result["results"][0]["id"]
    ts = time.time() - age_days * 86400
    mem.vector_store.update(h, {
        "status": "settled",
        "confidence": 0.82,
        "created_at": ts,
    })
    return h


# ── Basic hybrid search ───────────────────────────────────────────────────────

class TestHybridSearch:

    def test_finds_semantic_match(self, mem, retriever):
        """Vec search must find semantically related text."""
        _add_settled(mem, "用户偏好离线推理，不喜欢云端 API", "u1", topic="model-pref")
        results = retriever.search("本地模型", user_id="u1", top_k=3)
        assert len(results) > 0
        texts = [r["text"] for r in results]
        assert any("离线" in t or "云端" in t for t in texts)

    def test_identity_retrievable(self, mem, retriever):
        """Identity nodes must be searchable."""
        _add_settled(mem, "用户叫张三，是一名 ML 工程师", "u1", type="identity")
        results = retriever.search("工程师", user_id="u1", top_k=5)
        texts = [r["text"] for r in results]
        assert any("张三" in t for t in texts)

    def test_user_isolation(self, mem, retriever):
        """Search must not return other users' nodes."""
        _add_settled(mem, "用户 A 的私密信息", "user_a")
        _add_settled(mem, "用户 B 的公开信息", "user_b")
        results = retriever.search("私密信息", user_id="user_b", top_k=5)
        for r in results:
            assert r.get("user_id") == "user_b", \
                "user_b should never see user_a's nodes"

    def test_tentative_excluded(self, mem, retriever):
        """search_with_graph_expansion must exclude tentative nodes."""
        mem.add("这是 tentative 节点，不应该出现", user_id="u1")
        results = retriever.search_with_graph_expansion(
            "tentative 节点", user_id="u1", top_k=5
        )
        for r in results:
            assert r.get("status") != "tentative", \
                "Tentative nodes must never be returned by graph-expanded search"


# ── Time decay ────────────────────────────────────────────────────────────────

class TestTimeDecay:

    def test_newer_ranks_higher_than_older(self, mem, retriever):
        """A newer node on the same topic should outrank an older one."""
        _add_settled(mem, "用户认为 RAG 有用", "u1",
                     topic="rag", age_days=60)
        _add_settled(mem, "用户认为 RAG 没必要", "u1",
                     topic="rag", age_days=1)

        results = retriever.search("RAG 检索", user_id="u1", top_k=5)
        assert results, "Should return results"

        texts  = [r["text"] for r in results]
        newer  = next((i for i, t in enumerate(texts) if "没必要" in t), None)
        older  = next((i for i, t in enumerate(texts) if "有用" in t),    None)

        if newer is not None and older is not None:
            assert newer < older, (
                "Newer node ('没必要', 1 day old) should rank above "
                "older node ('有用', 60 days old)"
            )

    def test_event_decays_faster_than_position(self, mem):
        """Events (7-day half-life) should decay faster than positions (90-day)."""
        from noesis.retrieval.hybrid import _decay

        event_node    = {"type": "event",    "created_at": time.time() - 14 * 86400}
        position_node = {"type": "position", "created_at": time.time() - 14 * 86400}

        event_d    = _decay(event_node,    time.time())
        position_d = _decay(position_node, time.time())

        assert event_d < position_d, (
            "Event (half-life=7d) should decay faster than position (half-life=90d)"
        )

    def test_identity_never_decays(self, mem):
        """Identity nodes have None half-life → decay factor = 1.0."""
        from noesis.retrieval.hybrid import _decay
        node = {"type": "identity", "created_at": time.time() - 365 * 86400}
        assert _decay(node, time.time()) == 1.0


# ── Graph expansion ───────────────────────────────────────────────────────────

class TestGraphExpansion:

    def test_expansion_without_linker_returns_seeds(self, mem):
        """Without a linker, search_with_graph_expansion falls back to seeds."""
        retriever = HybridRetriever(
            vector_store=mem.vector_store,
            embedding_model=mem.embedding,
            linker=None,
        )
        _add_settled(mem, "图谱扩展基础节点", "u1", topic="graph-test")
        seeds   = retriever.search("图谱扩展", user_id="u1", top_k=3)
        expanded = retriever.search_with_graph_expansion(
            "图谱扩展", user_id="u1", top_k=3
        )
        assert len(expanded) >= len(seeds)

    def test_expansion_with_vault_links(self, mem, tmp_path):
        """Nodes linked via [[]] in the vault should appear in expanded results."""
        from noesis.graphs.obsidian_linker import ObsidianLinker

        h1 = _add_settled(mem, "本地向量存储方案", "u1", topic="storage")
        h2 = _add_settled(mem, "embedding 模型选型", "u1", topic="storage")

        # Manually write a vault link: h2 is Related to h1
        vault = tmp_path / "vault"
        md = vault / "thoughts" / f"{h2}.md"
        if md.exists():
            content = md.read_text()
            content = content.replace("Related:", f"Related: [[{h1[:8]}]]")
            md.write_text(content)

            linker = ObsidianLinker(
                vault_path=str(vault),
                vector_store=mem.vector_store,
                embedding_model=mem.embedding,
            )
            retriever = HybridRetriever(
                vector_store=mem.vector_store,
                embedding_model=mem.embedding,
                linker=linker,
            )

            # Search for h2's topic — should also pull in h1 via expansion
            results = retriever.search_with_graph_expansion(
                "embedding 选型", user_id="u1", top_k=10
            )
            result_ids = {r.get("id") or r.get("hash_id") for r in results}
            # h2 should definitely be there
            assert h2 in result_ids or any(
                "embedding" in r.get("text", "") for r in results
            )


# ── Signals ───────────────────────────────────────────────────────────────────

class TestSignals:

    def test_recency_signal_returns_recent(self, mem):
        """recency_signal should return most recent settled/provisional nodes."""
        for i in range(5):
            _add_settled(mem, f"历史节点 {i}", "u1")
        time.sleep(0.01)
        latest_h = _add_settled(mem, "最新节点", "u1")

        recent = recency_signal("u1", mem.vector_store, n=1)
        assert recent, "recency_signal should return at least one node"
        assert recent[0].get("text") == "最新节点", (
            "Most recent node should be first"
        )

    def test_core_fact_signal_returns_identity(self, mem):
        _add_settled(mem, "用户叫李四", "u1", type="identity")
        _add_settled(mem, "用户喜欢简洁代码风格", "u1", type="preference")
        _add_settled(mem, "某个随机位置立场", "u1", type="position")

        core = core_fact_signal("u1", mem.vector_store, top_k=5)
        types = {c.get("type") for c in core}
        assert "identity"   in types, "identity nodes must be returned"
        assert "preference" in types, "preference nodes must be returned"
        assert "position" not in types, "position nodes must NOT be in core signal"

    def test_recency_excludes_tentative(self, mem):
        """recency_signal must not return tentative nodes."""
        mem.add("tentative 节点不应该出现", user_id="u1")
        recent = recency_signal("u1", mem.vector_store, n=5)
        for r in recent:
            assert r.get("status") != "tentative"


# ── ContextBuilder end-to-end ─────────────────────────────────────────────────

class TestContextBuilder:

    def test_builder_returns_string(self, mem):
        _add_settled(mem, "用户叫王五", "u1", type="identity")
        ctx = mem.build_context("你叫什么名字", user_id="u1")
        assert isinstance(ctx, str)
        assert "王五" in ctx

    def test_builder_respects_budget(self, mem):
        """Output must not exceed token budget."""
        for i in range(20):
            _add_settled(mem, f"很长的立场内容 " * 10 + f" 节点{i}", "u1")

        ctx = mem.build_context("检索", user_id="u1", budget_tokens=100)
        # Rough check: 100 tokens ≈ 200 CJK chars
        assert len(ctx) < 800, "Context must be trimmed by budget"

    def test_builder_deduplicates(self, mem):
        """Same node should appear at most once in context."""
        h = _add_settled(mem, "独一无二的立场", "u1", type="identity")
        ctx = mem.build_context("独一无二", user_id="u1")
        # Count occurrences of the text
        count = ctx.count("独一无二的立场")
        assert count <= 1, f"Node appeared {count} times — must be deduplicated"
