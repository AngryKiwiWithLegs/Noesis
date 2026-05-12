"""
tests/test_confidence.py

Tests for ConfidenceScorer — four signals.
No API keys, no network, no pipeline needed.

Key assertions:
  - Weak language → low score → stays tentative
  - Repetition → score rises → provisional
  - Cross-tool match → score rises further
  - Contradiction → score drops
  - Identity type → never auto-decays (handled at retrieval layer)

Run:
    pytest tests/test_confidence.py -v
"""
import time
import pytest

from noesis.thoughts.confidence import ConfidenceScorer, _keywords, _is_negation
from noesis.thoughts.types import (
    ThoughtNode, THRESHOLD_PROVISIONAL, THRESHOLD_SETTLED
)


@pytest.fixture
def scorer() -> ConfidenceScorer:
    return ConfidenceScorer()


def _node(
    text:          str,
    type:          str  = "position",
    source_tool:   str  = "claude",
    topic_cluster: str  = "test-topic",
    hash_id:       str  = "abc123",
) -> ThoughtNode:
    return ThoughtNode(
        hash_id       = hash_id,
        type          = type,
        text          = text,
        user_id       = "u1",
        source_tool   = source_tool,
        topic_cluster = topic_cluster,
        created_at    = time.time(),
    )


def _rec(
    text:          str,
    hash_id:       str  = "xyz999",
    source_tool:   str  = "claude",
    topic_cluster: str  = "test-topic",
    type:          str  = "position",
) -> dict:
    return {
        "hash_id": hash_id, "text": text,
        "source_tool": source_tool, "topic_cluster": topic_cluster,
        "type": type,
    }


# ── Assertion strength signal ─────────────────────────────────────────────────

class TestAssertionStrength:

    def test_strong_zh_raises_score(self, scorer):
        node  = _node("我决定使用 sqlite-vec 作为热库")
        score = scorer.score(node, [])
        assert score > 0.35, f"Strong ZH assertion should score > 0.35, got {score}"

    def test_weak_zh_lowers_score(self, scorer):
        node  = _node("也许可以考虑一下 sqlite-vec")
        score = scorer.score(node, [])
        assert score < 0.35, f"Weak ZH assertion should score < 0.35, got {score}"

    def test_strong_en_raises_score(self, scorer):
        node  = _node("I've decided to use sqlite-vec for the hot store")
        score = scorer.score(node, [])
        assert score > 0.35

    def test_weak_en_lowers_score(self, scorer):
        node  = _node("Maybe sqlite-vec could work, perhaps")
        score = scorer.score(node, [])
        assert score < 0.35

    def test_neutral_is_middle(self, scorer):
        node  = _node("使用 sqlite-vec 存储向量数据")
        score = scorer.score(node, [])
        assert 0.15 < score < 0.65, f"Neutral should be mid-range, got {score}"


# ── Repetition signal ─────────────────────────────────────────────────────────

class TestRepetition:

    def test_no_repetition_baseline(self, scorer):
        node  = _node("用户认为 BM25 是首选检索方案")
        score = scorer.score(node, [])
        # No history → repetition component = 0
        assert score < THRESHOLD_PROVISIONAL

    def test_one_repetition_raises_score(self, scorer):
        node  = _node("用户认为 BM25 是首选检索方案", hash_id="node1")
        hist  = [_rec("BM25 是首选检索方案", hash_id="node0")]
        score = scorer.score(node, hist)
        score_no_hist = scorer.score(node, [])
        assert score > score_no_hist, "Repetition should raise score"

    def test_multiple_repetitions_push_to_provisional(self, scorer):
        node  = _node("RAG 对小数据集是过度工程", hash_id="main")
        hist  = [
            _rec("RAG 对小数据集是过度工程", hash_id=f"h{i}")
            for i in range(4)
        ]
        score = scorer.score(node, hist)
        assert score >= THRESHOLD_PROVISIONAL, (
            f"4 repetitions should reach provisional ({THRESHOLD_PROVISIONAL}), got {score}"
        )


# ── Cross-tool signal ─────────────────────────────────────────────────────────

class TestCrossTool:

    def test_same_tool_no_boost(self, scorer):
        node = _node("本地模型足够用", source_tool="claude", hash_id="n1")
        hist = [_rec("本地模型足够用", source_tool="claude", hash_id="h1")]
        score_same = scorer.score(node, hist)

        node2 = _node("本地模型足够用", source_tool="claude", hash_id="n2")
        score_no   = scorer.score(node2, [])

        # Same tool adds repetition but NOT cross-tool boost
        assert score_same >= score_no  # repetition still helps

    def test_different_tool_boosts_score(self, scorer):
        node   = _node("本地模型足够用", source_tool="claude", hash_id="n1")
        hist   = [_rec("本地模型足够用", source_tool="gpt-4o", hash_id="h1")]
        score  = scorer.score(node, hist)

        node2  = _node("本地模型足够用", source_tool="claude", hash_id="n2")
        score0 = scorer.score(node2, [])

        assert score > score0, "Cross-tool match should boost confidence"

    def test_cross_tool_can_reach_provisional(self, scorer):
        """
        Cross-tool corroboration + moderate assertion should reach provisional.
        """
        node  = _node("我认为本地模型足够用", source_tool="claude", hash_id="n1")
        hist  = [
            _rec("本地模型足够用", source_tool="gpt-4o", hash_id="h1"),
            _rec("本地模型足够用", source_tool="gpt-4o", hash_id="h2"),
        ]
        score = scorer.score(node, hist)
        assert score >= THRESHOLD_PROVISIONAL, (
            f"Cross-tool + assertion should reach provisional, got {score}"
        )


# ── Contradiction signal ──────────────────────────────────────────────────────

class TestContradiction:

    def test_contradiction_node_penalises(self, scorer):
        node  = _node("RAG 不是过度工程", hash_id="n1")
        hist  = [_rec("RAG 是否过度工程", type="contradiction", hash_id="c1")]
        score = scorer.score(node, hist)

        node2 = _node("RAG 不是过度工程", hash_id="n2")
        score0 = scorer.score(node2, [])

        assert score < score0, "Contradiction node should lower score"

    def test_no_contradiction_full_score(self, scorer):
        node  = _node("使用 sqlite-vec", hash_id="n1")
        hist  = [_rec("sqlite-vec 是好选择", hash_id="h1")]  # supporting, not contradicting
        score = scorer.score(node, hist)

        # no_contradiction component should be 1.0 (no contradiction found)
        no_contr = scorer._no_contradiction(node, hist)
        assert no_contr == 1.0


# ── Status transitions ────────────────────────────────────────────────────────

class TestStatusTransitions:

    def test_low_score_stays_tentative(self, scorer):
        node   = _node("也许可以试试")
        score  = scorer.score(node, [])
        status = scorer.next_status(score, "tentative")
        assert status == "tentative"

    def test_medium_score_becomes_provisional(self, scorer):
        status = scorer.next_status(THRESHOLD_PROVISIONAL + 0.01, "tentative")
        assert status == "provisional"

    def test_high_score_becomes_settled(self, scorer):
        status = scorer.next_status(THRESHOLD_SETTLED + 0.01, "tentative")
        assert status == "settled"

    def test_no_downgrade_on_score(self, scorer):
        """next_status must not downgrade (decay is handled separately)."""
        status = scorer.next_status(0.1, "provisional")
        assert status == "provisional", "next_status must not downgrade"


# ── Helper function tests ─────────────────────────────────────────────────────

class TestHelpers:

    def test_keywords_filters_stopwords(self):
        kws = _keywords("用户认为 RAG 是过度工程的一种")
        assert "的" not in kws
        assert "rag" in kws or "RAG" in kws.union({k.upper() for k in kws})

    def test_keywords_handles_english(self):
        kws = _keywords("the user prefers local models over cloud")
        assert "the" not in kws
        assert "prefers" in kws or "user" in kws

    def test_negation_detected(self):
        assert _is_negation("用户不认为 RAG 有必要", "RAG 是好方案")

    def test_negation_not_false_positive(self):
        assert not _is_negation("用户认为 RAG 很好", "RAG 是好方案")

    def test_negation_en(self):
        assert _is_negation("I do not think RAG is useful", "RAG is useful")
