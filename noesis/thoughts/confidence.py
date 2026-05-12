"""
noesis/thoughts/confidence.py

Four signals that determine whether a captured thought
should be promoted from tentative → provisional → settled.

Design principle: time is the best filter.
Don't try to judge quality at write time.
Let repetition, assertion strength, cross-tool consistency,
and absence of contradiction do the work over days and weeks.
"""
from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

from .types import (
    ThoughtNode,
    ThoughtStatus,
    THRESHOLD_PROVISIONAL,
    THRESHOLD_SETTLED,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Assertion-strength vocabulary ─────────────────────────────────────────────
# Maps keyword patterns → strength score (0–1)
_STRONG_ZH = [
    (r"我决定|我确定|我确信|明确|一定要|绝对", 0.9),
    (r"我认为|我认为|我觉得|我判断|我倾向于", 0.7),
    (r"应该|需要|必须|值得", 0.6),
]
_WEAK_ZH = [
    (r"也许|可能|或许|大概|感觉上|说不定", 0.2),
    (r"不确定|待定|待验证|不太清楚", 0.1),
]
_STRONG_EN = [
    (r"\b(I decided|I'm certain|I'm sure|definitely|absolutely)\b", 0.9),
    (r"\b(I think|I believe|I feel|I judge)\b", 0.7),
    (r"\b(should|must|need to|worth)\b", 0.6),
]
_WEAK_EN = [
    (r"\b(maybe|perhaps|possibly|might|probably)\b", 0.2),
    (r"\b(not sure|uncertain|TBD|unclear)\b", 0.1),
]


class ConfidenceScorer:
    """
    Weights (must sum to 1.0):
      repetition      0.35  — strongest signal, hardest to fake
      assertion        0.25  — language marker (how strongly stated)
      no_contradiction 0.25  — absence of conflicting judgement
      cross_tool       0.15  — same stance in different AI tools
    """

    WEIGHTS = {
        "repetition":       0.35,
        "assertion":        0.25,
        "no_contradiction": 0.25,
        "cross_tool":       0.15,
    }

    # ── Main entry ────────────────────────────────────────────────────────────

    def score(
        self,
        node:     ThoughtNode,
        history:  list[dict],   # list of existing hot-store records for same user
    ) -> float:
        s = {
            "repetition":       self._repetition(node, history),
            "assertion":        self._assertion(node.text),
            "no_contradiction": self._no_contradiction(node, history),
            "cross_tool":       self._cross_tool(node, history),
        }
        total = sum(self.WEIGHTS[k] * v for k, v in s.items())
        logger.debug(
            f"confidence({node.hash_id[:8]}): "
            + " ".join(f"{k}={v:.2f}" for k, v in s.items())
            + f" → {total:.2f}"
        )
        return round(min(total, 1.0), 3)

    def next_status(self, score: float, current: ThoughtStatus) -> ThoughtStatus:
        if score >= THRESHOLD_SETTLED:
            return "settled"
        if score >= THRESHOLD_PROVISIONAL:
            return "provisional"
        return current  # don't downgrade here; decay handles that separately

    # ── Signal 1: repetition ──────────────────────────────────────────────────

    def _repetition(self, node: ThoughtNode, history: list[dict]) -> float:
        """
        Count how many *other* records in the hot store are semantically
        similar (same topic_cluster + similar enough text keywords).
        Each occurrence adds +0.15, capped at 1.0.
        """
        cluster   = node.topic_cluster
        node_kws  = _keywords(node.text)
        count     = 0

        for rec in history:
            if rec.get("hash_id") == node.hash_id:
                continue
            if rec.get("topic_cluster") != cluster:
                continue
            rec_kws = _keywords(rec.get("text", ""))
            overlap = len(node_kws & rec_kws)
            if overlap >= 2:
                count += 1

        return min(count * 0.15, 1.0)

    # ── Signal 2: assertion strength ─────────────────────────────────────────

    def _assertion(self, text: str) -> float:
        """
        Scan text for strong/weak assertion markers.
        Returns a score in [0, 1].
        """
        score = 0.5  # default: neutral

        for pattern, strength in _STRONG_ZH + _STRONG_EN:
            if re.search(pattern, text, re.IGNORECASE):
                score = max(score, strength)

        for pattern, weakness in _WEAK_ZH + _WEAK_EN:
            if re.search(pattern, text, re.IGNORECASE):
                score = min(score, weakness)

        return score

    # ── Signal 3: no contradiction ────────────────────────────────────────────

    def _no_contradiction(
        self, node: ThoughtNode, history: list[dict]
    ) -> float:
        """
        If any record in the same topic cluster contains a negation of
        this node's text, penalise confidence.
        Full score = 1.0 (no contradiction found).
        """
        cluster  = node.topic_cluster
        node_kws = _keywords(node.text)

        for rec in history:
            if rec.get("hash_id") == node.hash_id:
                continue
            if rec.get("topic_cluster") != cluster:
                continue
            if rec.get("type") == "contradiction":
                # Any contradiction node in same cluster → penalise
                return 0.2
            rec_kws = _keywords(rec.get("text", ""))
            overlap = len(node_kws & rec_kws)
            if overlap >= 2 and _is_negation(rec.get("text", ""), node.text):
                return 0.1

        return 1.0

    # ── Signal 4: cross-tool consistency ─────────────────────────────────────

    def _cross_tool(
        self, node: ThoughtNode, history: list[dict]
    ) -> float:
        """
        If the same stance appears in records from a *different* source_tool,
        confidence gets a significant boost.
        """
        cluster   = node.topic_cluster
        node_tool = node.source_tool
        node_kws  = _keywords(node.text)

        for rec in history:
            if rec.get("source_tool") == node_tool:
                continue
            if rec.get("topic_cluster") != cluster:
                continue
            rec_kws = _keywords(rec.get("text", ""))
            if len(node_kws & rec_kws) >= 2:
                return 1.0  # Found cross-tool corroboration

        return 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

STOP = {
    "的", "了", "是", "在", "我", "你", "他", "她", "它",
    "和", "或", "但", "that", "this", "the", "a", "an",
    "is", "are", "was", "were", "be", "to", "of", "and",
    "or", "but", "in", "on", "at", "for", "with",
}


def _keywords(text: str) -> set[str]:
    """
    Extract meaningful tokens from mixed Chinese/Latin text.
    - CJK: character bigrams to handle compound phrases
    - Latin: whole-word tokens (preserves model IDs like 'gpt-4o')
    """
    tokens: set[str] = set()
    # Latin words
    tokens.update(t for t in re.findall(r"[\w][\w-]*", text.lower())
                  if len(t) > 1 and t not in STOP)
    # CJK character bigrams (captures meaningful multi-char units)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(cjk_chars) - 1):
        tokens.add(cjk_chars[i] + cjk_chars[i + 1])
    return tokens


def _is_negation(candidate: str, reference: str) -> bool:
    """Very naive negation detector — enough for a v1."""
    NEG_ZH = ["不", "否", "没有", "不是", "不认为", "反对", "放弃"]
    NEG_EN = ["not", "no", "never", "disagree", "wrong", "against"]
    candidate_lower = candidate.lower()
    ref_kws = _keywords(reference)

    has_negation = any(n in candidate_lower for n in NEG_ZH + NEG_EN)
    has_overlap  = bool(ref_kws & _keywords(candidate))

    return has_negation and has_overlap
