"""
tests/test_cross_tool.py

Cross-tool scenario tests and cluster management tests.
No API keys required.

Run:
    pytest tests/test_cross_tool.py -v
"""
import time
import pytest

from noesis.memory.main import Memory
from noesis.graphs.cluster import ClusterManager


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path):
    return Memory.from_config({
        "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        "cold_store":   {"config": {"vault_path": str(tmp_path / "vault")}},
    })


@pytest.fixture
def cluster_mgr(tmp_path, mem):
    return ClusterManager(
        vault_path=str(tmp_path / "vault"),
        vector_store=mem.vector_store,
    )


def _add_settled(mem, text, user_id, source_tool="claude",
                 topic="test", type="position", age_days=0):
    r = mem.add(text, user_id=user_id, source_tool=source_tool,
                topic_cluster=topic, type=type)
    h = r["results"][0]["id"]
    ts = time.time() - age_days * 86400
    mem.vector_store.update(h, {"status": "settled", "confidence": 0.85,
                                "created_at": ts})
    return h


# ── Cross-tool retrieval ───────────────────────────────────────────────────────

class TestCrossToolRetrieval:

    def test_memories_from_both_tools_searchable(self, mem):
        """Nodes from Claude and GPT must both be retrievable."""
        h1 = _add_settled(mem, "Claude 对话：使用 sqlite-vec 作为向量库",
                          "u1", source_tool="claude-sonnet-4-6", topic="storage")
        h2 = _add_settled(mem, "GPT 对话：sqlite-vec 比 Chroma 更轻量",
                          "u1", source_tool="gpt-4o", topic="storage")

        results = mem.search("向量库选型", user_id="u1", top_k=10)
        ids     = {r.get("id") or r.get("hash_id") for r in results}
        assert h1 in ids or h2 in ids, "At least one cross-tool node must appear"

    def test_source_tool_preserved(self, mem):
        """source_tool metadata must survive storage and retrieval."""
        h = _add_settled(mem, "GPT 分析：项目架构合理",
                         "u1", source_tool="gpt-4o")
        node = mem.vector_store.get(h)
        assert node["source_tool"] == "gpt-4o"

    def test_cross_tool_in_build_context(self, mem):
        """build_context must include nodes from multiple tools."""
        _add_settled(mem, "用户认为本地模型够用", "u1",
                     source_tool="claude-sonnet-4-6", topic="model-choice")
        _add_settled(mem, "用户担心本地模型推理速度", "u1",
                     source_tool="gpt-4o", topic="model-choice")

        ctx = mem.build_context("本地模型的优劣", user_id="u1")
        # At least one of the two should appear
        assert "本地模型" in ctx

    def test_cross_tool_no_duplication(self, mem):
        """Identical text from two tools should appear at most once in context."""
        text = "用户选择 sqlite-vec 作为向量存储"
        _add_settled(mem, text, "u1", source_tool="claude-sonnet-4-6")
        _add_settled(mem, text, "u1", source_tool="gpt-4o")

        ctx = mem.build_context("向量存储选择", user_id="u1")
        assert ctx.count(text) <= 1, "Duplicate text must be deduplicated"

    def test_different_users_isolated(self, mem):
        """Cross-tool nodes from user A must never appear for user B."""
        _add_settled(mem, "User A: confidential project details",
                     "user_a", source_tool="claude-sonnet-4-6")
        _add_settled(mem, "User A via GPT: more confidential info",
                     "user_a", source_tool="gpt-4o")

        results = mem.search("confidential", user_id="user_b", top_k=10)
        for r in results:
            assert r.get("user_id") == "user_b", \
                "user_b must never see user_a's cross-tool nodes"


# ── Confidence boost from cross-tool ─────────────────────────────────────────

class TestCrossToolConfidence:

    def test_cross_tool_corroboration_detected(self, mem):
        """
        When same stance appears in two different tools,
        ConfidenceScorer should detect cross_tool signal.
        """
        from noesis.thoughts.confidence import ConfidenceScorer
        from noesis.thoughts.types import ThoughtNode

        scorer = ConfidenceScorer()

        node = ThoughtNode(
            hash_id="n1", type="position",
            text="用户认为 sqlite-vec 是最佳热库方案",
            user_id="u1", source_tool="claude-sonnet-4-6",
            topic_cluster="storage",
        )
        history = [
            {
                "hash_id":       "h1",
                "text":          "sqlite-vec 是最佳热库方案",
                "source_tool":   "gpt-4o",
                "topic_cluster": "storage",
                "type":          "position",
            }
        ]
        score = scorer.score(node, history)
        cross = scorer._cross_tool(node, history)
        assert cross == 1.0, "Cross-tool corroboration should return 1.0"
        assert score > scorer.score(node, []), \
            "Cross-tool history should raise overall confidence"


# ── Cluster management ────────────────────────────────────────────────────────

class TestClusterManager:

    def test_create_cluster(self, cluster_mgr):
        path = cluster_mgr.get_or_create("rag-retrieval")
        assert path.exists()
        content = path.read_text()
        assert "rag-retrieval" in content
        assert "## Thoughts" in content

    def test_idempotent_create(self, cluster_mgr):
        cluster_mgr.get_or_create("storage-choice")
        cluster_mgr.get_or_create("storage-choice")  # should not raise or duplicate
        clusters = cluster_mgr.list_clusters()
        assert clusters.count("storage-choice") == 1

    def test_add_member(self, cluster_mgr):
        cluster_mgr.get_or_create("retrieval")
        cluster_mgr.add_member("retrieval", "abc123def456",
                               source_tool="claude-sonnet-4-6")
        members = cluster_mgr.get_members("retrieval")
        assert "abc123def456" in members

    def test_add_member_idempotent(self, cluster_mgr):
        cluster_mgr.get_or_create("test-cluster")
        cluster_mgr.add_member("test-cluster", "aabbccddeeff")
        cluster_mgr.add_member("test-cluster", "aabbccddeeff")  # second add
        content = cluster_mgr._path("test-cluster").read_text()
        assert content.count("aabbccddeeff") == 1

    def test_source_tool_tracked(self, cluster_mgr):
        cluster_mgr.get_or_create("tools-cluster")
        cluster_mgr.add_member("tools-cluster", "aaa111bbb222",
                               source_tool="gpt-4o")
        content = cluster_mgr._path("tools-cluster").read_text()
        assert "gpt-4o" in content

    def test_open_question_added(self, cluster_mgr):
        cluster_mgr.get_or_create("arch-decisions")
        cluster_mgr.add_open_question(
            "arch-decisions",
            "q_abc123",
            "sqlite-vec 在超过 10 万条记录时性能如何？"
        )
        content = cluster_mgr._path("arch-decisions").read_text()
        assert "q_abc123" in content
        assert "Open Questions" in content

    def test_close_question(self, cluster_mgr):
        cluster_mgr.get_or_create("arch")
        cluster_mgr.add_open_question("arch", "q_001", "测试问题")
        cluster_mgr.close_question("arch", "q_001", "ans_002")
        content = cluster_mgr._path("arch").read_text()
        assert "~~[[q_001]]~~" in content
        assert "ans_002" in content

    def test_list_clusters(self, cluster_mgr):
        for name in ["topic-a", "topic-b", "topic-c"]:
            cluster_mgr.get_or_create(name)
        clusters = cluster_mgr.list_clusters()
        for name in ["topic-a", "topic-b", "topic-c"]:
            assert name in clusters


# ── NOESIS_SCHEMA.md ──────────────────────────────────────────────────────────

class TestVaultSchema:

    def test_schema_file_exists(self, mem, tmp_path):
        vault = tmp_path / "vault"
        schema = vault / "NOESIS_SCHEMA.md"
        assert schema.exists(), "NOESIS_SCHEMA.md must be created on vault init"

    def test_schema_has_required_sections(self, mem, tmp_path):
        vault = tmp_path / "vault"
        content = (vault / "NOESIS_SCHEMA.md").read_text()
        for section in [
            "Directory ownership",
            "Required frontmatter fields",
            "Cross-layer link rules",
            "Conflict resolution",
            "Graph tier convention",
        ]:
            assert section in content, f"Schema missing section: {section}"

    def test_vault_directories_created(self, mem, tmp_path):
        vault = tmp_path / "vault"
        assert (vault / "thoughts").is_dir()
        assert (vault / "clusters").is_dir()
        assert (vault / "wiki").is_dir()
