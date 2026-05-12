"""
noesis/thoughts/types.py

All core dataclasses. Foundation for every other module.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional


# ── Type aliases ──────────────────────────────────────────────────────────────

ThoughtType = Literal[
    "position",       # Your current stance on something
    "question",       # Open question driving knowledge acquisition
    "synthesis",      # Insight from merging two ideas (often cross-tool)
    "contradiction",  # Conflicting judgements — human resolves
    "event",          # Specific thing that happened or was decided
    "identity",       # Who you are, your role, stable facts about you
    "preference",     # How you like things done
]

ThoughtStatus = Literal[
    "tentative",    # Just captured — stored, never injected, not in graph
    "provisional",  # Emerging — injected when topic-relevant
    "settled",      # Stable — always injected, full graph links
    "superseded",   # Replaced by a newer node — filtered from retrieval
]

# How long (days) before a node starts decaying in confidence
# None = never decays
HALF_LIFE_DAYS: dict[str, Optional[float]] = {
    "event":         7.0,
    "question":     30.0,
    "position":     90.0,
    "preference":   90.0,
    "synthesis":   180.0,
    "contradiction": None,
    "identity":      None,
}

# Confidence thresholds for status transitions
THRESHOLD_PROVISIONAL = 0.45
THRESHOLD_SETTLED      = 0.75


# ── Core node ────────────────────────────────────────────────────────────────

@dataclass
class ThoughtNode:
    hash_id:        str
    type:           ThoughtType
    text:           str
    status:         ThoughtStatus       = "tentative"
    confidence:     float               = 0.0
    user_id:        str                 = ""
    source_tool:    str                 = ""
    source_session: str                 = ""
    topic_cluster:  str                 = ""
    created_at:     float               = field(default_factory=time.time)
    fact_ref:       Optional[str]       = None   # pointer to hot-store hash
    evolved_from:   Optional[str]       = None   # predecessor node hash
    superseded_by:  Optional[str]       = None
    supersedes:     Optional[str]       = None

    def is_injectable(self) -> bool:
        return self.status in ("provisional", "settled")

    def half_life(self) -> Optional[float]:
        return HALF_LIFE_DAYS.get(self.type)


# ── Extraction output (before storage) ───────────────────────────────────────

@dataclass
class ThoughtCandidate:
    """What the extractor produces from a conversation turn."""
    type:               ThoughtType
    text:               str
    initial_confidence: float           = 0.20
    source_tool:        str             = ""
    source_session:     str             = ""
    topic_cluster:      str             = ""
    assertion_strength: float           = 0.0   # 0–1, from assertion scorer
    fact_text:          Optional[str]   = None   # raw fact if dual-storing


# ── Pipeline task ─────────────────────────────────────────────────────────────

@dataclass
class ConsolidationTask:
    hash_id:   str
    candidate: ThoughtCandidate
    neighbors: list                     = field(default_factory=list)
    user_id:   str                      = ""
    urgent:    bool                     = False  # True if similarity > 0.95
