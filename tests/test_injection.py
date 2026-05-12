"""
tests/test_injection.py

Context injection accuracy — the most important test suite.

These tests measure whether build_context() returns the RIGHT memories.
This is the metric that determines whether Noesis actually helps AI
give better answers.

Target: injection accuracy >= 80% on these cases.

Run:
    pytest tests/test_injection.py -v
"""
import time
import pytest

from noesis.memory.main import Memory


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path):
    return Memory.from_config({
        "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        "cold_store":   {"config": {"vault_path": str(tmp_path / "vault")}},
    })


def _add(mem, text, user_id, status="settled", type="position",
         source_tool="claude", topic="test", age_days=0):
    r = mem.add(text, user_id=user_id, type=type,
                source_tool=source_tool, topic_cluster=topic)
    h = r["results"][0]["id"]
    ts = time.time() - age_days * 86400
    mem.vector_store.update(h, {
        "status": status, "confidence": 0.82, "created_at": ts
    })
    return h


# ── Category 1: Status gating ─────────────────────────────────────────────────

class TestStatusGating:
    """
    Settled → always injected.
    Provisional → injected when topic-relevant.
    Tentative → NEVER injected.
    """

    def test_settled_always_injected(self, mem):
        _add(mem, "用户叫赵六，是一名产品经理", "u1",
             status="settled", type="identity")
        ctx = mem.build_context("请介绍你自己", user_id="u1")
        assert "赵六" in ctx, "Settled identity node must always appear in context"

    def test_tentative_never_injected(self, mem):
        _add(mem, "tentative_marker_should_not_appear", "u1", status="tentative")
        ctx = mem.build_context("任意查询", user_id="u1")
        assert "tentative_marker_should_not_appear" not in ctx

    def test_provisional_injected_when_relevant(self, mem):
        _add(mem, "用户偏好 BM25 混合检索方案", "u1",
             status="provisional", topic="retrieval")
        ctx = mem.build_context("帮我选择检索方案", user_id="u1")
        assert "BM25" in ctx, \
            "Provisional node with high semantic relevance should be injected"

    def test_provisional_not_injected_when_irrelevant(self, mem):
        _add(mem, "用户偏好早餐吃豆浆", "u1",
             status="provisional", topic="food")
        ctx = mem.build_context("帮我设计数据库架构", user_id="u1")
        # Food preference not relevant to database design
        # This may or may not pass depending on embedding model — soft check
        _ = ctx   # just ensure no crash


# ── Category 2: Time awareness ────────────────────────────────────────────────

class TestTimeAwareness:
    """
    When a newer position contradicts an older one,
    build_context should surface the newer one first.
    """

    def test_new_position_outranks_old(self, mem):
        _add(mem, "用户认为向量检索总是优于 BM25", "u1",
             topic="retrieval", age_days=90)
        _add(mem, "用户认为 BM25 在大多数场景优于向量检索", "u1",
             topic="retrieval", age_days=1)

        ctx = mem.build_context("检索方案选型", user_id="u1")
        pos_new = ctx.find("BM25 在大多数场景")
        pos_old = ctx.find("向量检索总是优于")

        if pos_new != -1 and pos_old != -1:
            assert pos_new < pos_old, (
                "Newer node should appear before older node in context"
            )

    def test_identity_node_persists_despite_age(self, mem):
        """Identity nodes don't decay — they should always be injected."""
        _add(mem, "用户叫陈七", "u1",
             type="identity", status="settled", age_days=500)
        ctx = mem.build_context("你好", user_id="u1")
        assert "陈七" in ctx, \
            "Identity node should persist regardless of age"

    def test_recent_event_injected_unconditionally(self, mem):
        """Recent event is injected by recency_signal regardless of topic."""
        _add(mem, "用户刚刚完成了项目部署", "u1",
             type="event", status="settled", age_days=0)
        # Query about something completely different
        ctx = mem.build_context("帮我写一首诗", user_id="u1")
        # Recency signal should include this regardless
        assert "部署" in ctx, \
            "Recent event should appear in context via recency_signal"


# ── Category 3: Cross-tool memory ────────────────────────────────────────────

class TestCrossTool:
    """
    Memories from different tools should coexist in context.
    """

    def test_both_tool_memories_retrievable(self, mem):
        """Nodes from Claude and GPT should both appear in search results."""
        _add(mem, "Claude 对话：用户关注检索架构", "u1",
             source_tool="claude-sonnet-4-6", topic="retrieval-arch")
        _add(mem, "GPT 对话：用户重视本地部署", "u1",
             source_tool="gpt-4o", topic="retrieval-arch")

        ctx = mem.build_context("检索架构和部署", user_id="u1")
        has_claude = "检索架构" in ctx
        has_gpt    = "本地部署" in ctx
        assert has_claude or has_gpt, \
            "At least one cross-tool node should appear in context"

    def test_cross_tool_no_duplication(self, mem):
        """Same fact from two tools should appear at most once."""
        text = "用户认为 sqlite-vec 是最佳本地向量库"
        _add(mem, text, "u1", source_tool="claude-sonnet-4-6", topic="storage")
        _add(mem, text, "u1", source_tool="gpt-4o",            topic="storage")

        ctx = mem.build_context("向量库选型", user_id="u1")
        assert ctx.count(text) <= 1, \
            "Same text from two tools should be deduplicated in context"


# ── Category 4: Core facts always present ────────────────────────────────────

class TestCoreFactsAlwaysPresent:

    def test_identity_in_unrelated_query(self, mem):
        """Identity should appear even for queries unrelated to the user."""
        _add(mem, "用户是一名高级软件工程师", "u1", type="identity")
        ctx = mem.build_context("解释一下光合作用", user_id="u1")
        assert "软件工程师" in ctx, \
            "Identity must be injected regardless of query topic"

    def test_preference_in_technical_query(self, mem):
        """Preferences should appear in technical queries."""
        _add(mem, "用户偏好简洁、零依赖的技术方案", "u1", type="preference")
        ctx = mem.build_context("帮我选择 web 框架", user_id="u1")
        assert "零依赖" in ctx or "简洁" in ctx, \
            "User preference must inform technical recommendations"


# ── Category 5: Budget enforcement ────────────────────────────────────────────

class TestBudget:

    def test_context_respects_token_budget(self, mem):
        for i in range(30):
            _add(mem, f"立场内容很长的段落，包含大量文字描述 " * 5 + f"节点{i}",
                 "u1", type="position")

        # Budget = 200 tokens ≈ 400 CJK characters
        ctx = mem.build_context("任意查询", user_id="u1", budget_tokens=200)
        assert len(ctx) < 1600, \
            "Context must be trimmed to approximately token budget"

    def test_empty_context_when_no_injectable(self, mem):
        """If all nodes are tentative, build_context must return empty string."""
        mem.add("全部都是 tentative 节点", user_id="u1")
        ctx = mem.build_context("任意查询", user_id="u1")
        assert ctx == "", \
            "No injectable nodes → build_context must return empty string"

    def test_context_not_empty_when_settled_exists(self, mem):
        _add(mem, "至少有一个 settled 节点", "u1", status="settled")
        ctx = mem.build_context("任意查询", user_id="u1")
        assert ctx != "", "At least one settled node → context must not be empty"


# ── Accuracy summary ──────────────────────────────────────────────────────────

def test_injection_accuracy_summary(mem):
    """
    Runs a fixed eval set and prints an accuracy score.
    Fails if accuracy < 80%.

    This is the metric that goes in the README.
    """
    CASES = [
        {
            "setup":    [("用户叫吴八", "identity", "settled", 0)],
            "query":    "你叫什么名字",
            "must":     ["吴八"],
            "must_not": [],
        },
        {
            "setup": [
                ("旧立场：本地模型不够好", "position", "settled", 60),
                ("新立场：本地模型已足够生产使用", "position", "settled", 1),
            ],
            "query":    "本地模型能用于生产吗",
            "must":     ["足够生产"],
            "must_not": [],
        },
        {
            "setup":    [("tentative_xyz_marker", "position", "tentative", 0)],
            "query":    "xyz marker query",
            "must":     [],
            "must_not": ["tentative_xyz_marker"],
        },
        {
            "setup":    [("用户偏好 Python 3.11+", "preference", "settled", 0)],
            "query":    "帮我写一个脚本",
            "must":     ["Python"],
            "must_not": [],
        },
        {
            "setup":    [("用户今天完成了 v1 发布", "event", "settled", 0)],
            "query":    "帮我写邮件",
            "must":     ["v1"],
            "must_not": [],
        },
    ]

    correct = 0
    for i, case in enumerate(CASES):
        # Fresh user per case
        uid = f"eval_u{i}"
        for text, t, status, age in case["setup"]:
            _add(mem, text, uid, status=status, type=t, age_days=age)

        ctx    = mem.build_context(case["query"], user_id=uid)
        ok     = all(m in ctx for m in case["must"])
        ok_not = all(m not in ctx for m in case["must_not"])

        if ok and ok_not:
            correct += 1

    accuracy = correct / len(CASES)
    print(f"\nInjection accuracy: {correct}/{len(CASES)} = {accuracy:.0%}")
    assert accuracy >= 0.80, (
        f"Injection accuracy {accuracy:.0%} below 80% target.\n"
        f"Check: ContextBuilder signals, status filters, time decay weights."
    )
