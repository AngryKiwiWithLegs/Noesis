"""
noesis/context/builder.py

Coordinates three retrieval signals into a single system-prompt string.

Design:
  - semantic_signal  (graph-expanded hybrid search)  →  topically relevant
  - recency_signal   (recent non-tentative nodes)     →  temporally relevant
  - core_fact_signal (identity + preference)          →  always relevant

Deduplication is by hash_id.
Trimming is by token budget (rough CJK-aware estimate).
Tentative nodes are never injected — the signal functions
only return provisional/settled nodes.
"""
from __future__ import annotations

import logging
from typing import Optional

from .signals import semantic_signal, recency_signal, core_fact_signal

logger = logging.getLogger(__name__)


def _token_est(text: str) -> int:
    cjk   = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin = len(text) - cjk
    return cjk // 2 + latin // 4 + 1


class ContextBuilder:
    """
    Primary read-path coordinator.

    Usage (injected into Memory):
        builder = ContextBuilder(vector_store, embedding, retriever, linker)
        context_str = builder.build(query, user_id, budget_tokens=1200)
    """

    def __init__(
        self,
        vector_store,
        embedding_model,
        retriever,
        linker:          Optional[object] = None,
        recency_n:       int   = 3,
        core_top_k:      int   = 3,
        semantic_top_k:  int   = 8,
    ):
        self.vs           = vector_store
        self.emb          = embedding_model
        self.retriever    = retriever
        self.linker       = linker
        self.recency_n    = recency_n
        self.core_top_k   = core_top_k
        self.semantic_k   = semantic_top_k

    # ── Public ────────────────────────────────────────────────────────────────

    def build(
        self,
        query:         str,
        user_id:       str,
        budget_tokens: int = 1200,
    ) -> str:
        """
        Returns a ready-to-inject system-prompt string.
        Empty string if no injectable nodes exist.
        """
        candidates: list[dict] = []

        # Signal 1: semantic + graph expansion
        try:
            candidates += semantic_signal(
                query, user_id, self.retriever, top_k=self.semantic_k
            )
        except Exception as e:
            logger.warning(f"semantic_signal failed: {e}")

        # Signal 2: recency
        try:
            candidates += recency_signal(user_id, self.vs, n=self.recency_n)
        except Exception as e:
            logger.warning(f"recency_signal failed: {e}")

        # Signal 3: core identity / preference
        try:
            candidates += core_fact_signal(user_id, self.vs, top_k=self.core_top_k)
        except Exception as e:
            logger.warning(f"core_fact_signal failed: {e}")

        if not candidates:
            return ""

        return self._format(self._dedup_trim(candidates, budget_tokens))

    # ── Internals ─────────────────────────────────────────────────────────────

    def _dedup_trim(
        self,
        candidates: list[dict],
        budget:     int,
    ) -> list[dict]:
        seen:  set[str]  = set()
        out:   list[dict] = []
        used:  int        = 0

        for m in candidates:
            mid = m.get("id") or m.get("hash_id", "")
            if not mid or mid in seen:
                continue
            seen.add(mid)

            text = m.get("text", "")
            cost = _token_est(text)
            if used + cost > budget:
                break

            out.append(m)
            used += cost

        logger.debug(
            f"build_context: {len(candidates)} candidates → "
            f"{len(out)} injected ({used} tokens)"
        )
        return out

    @staticmethod
    def _format(nodes: list[dict]) -> str:
        if not nodes:
            return ""
        lines = []
        for m in nodes:
            t    = m.get("type",   "?")
            s    = m.get("status", "?")
            text = m.get("text",   "").strip()
            lines.append(f"- [{t}·{s}] {text}")
        return "以下是关于用户的已知信息：\n" + "\n".join(lines)
