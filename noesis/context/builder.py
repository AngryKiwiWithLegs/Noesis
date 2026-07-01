"""
noesis/context/builder.py

Coordinates four retrieval signals into a single system-prompt string.

Design:
  - semantic_signal  (graph-expanded hybrid search)  →  topically relevant
  - recency_signal   (recent non-tentative nodes)     →  temporally relevant
  - core_fact_signal (identity + preference)          →  always relevant
  - wiki_signal      (compiled wiki/ pages)           →  grounds positions
                                                         with documented knowledge

Deduplication is by hash_id.
Trimming is by token budget (rough CJK-aware estimate).
Tentative nodes are never injected — the signal functions
only return provisional/settled nodes.
"""
from __future__ import annotations

import logging
from typing import Optional

from .signals import (
    semantic_signal,
    recency_signal,
    core_fact_signal,
    wiki_signal,
)

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
        vault_path:      Optional[str] = None,
        wiki_top_k:      int   = 3,
        wiki_budget_pct: float = 0.25,
    ):
        self.vs           = vector_store
        self.emb          = embedding_model
        self.retriever    = retriever
        self.linker       = linker
        self.recency_n    = recency_n
        self.core_top_k   = core_top_k
        self.semantic_k   = semantic_top_k
        self.vault_path   = vault_path
        self.wiki_k       = wiki_top_k
        self.wiki_pct     = wiki_budget_pct

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

        # Split the budget: wiki gets its own slice so documented knowledge
        # doesn't crowd out live thoughts (and vice versa).
        wiki_budget = int(budget_tokens * self.wiki_pct)
        thought_budget = budget_tokens - wiki_budget

        thoughts = self._dedup_trim(candidates, thought_budget)

        # Signal 4: wiki knowledge (compiled documents)
        wiki_nodes: list[dict] = []
        if self.vault_path:
            try:
                wiki_nodes = wiki_signal(
                    query, self.vault_path, self.emb, top_k=self.wiki_k
                )
            except Exception as e:
                logger.warning(f"wiki_signal failed: {e}")
        wiki_trimmed = self._dedup_trim(wiki_nodes, wiki_budget)

        if not thoughts and not wiki_trimmed:
            return ""

        return self._format(thoughts, wiki_trimmed)

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
    def _format(thoughts: list[dict], wiki: list[dict] | None = None) -> str:
        sections = []
        if thoughts:
            lines = []
            for m in thoughts:
                t    = m.get("type",   "?")
                s    = m.get("status", "?")
                text = m.get("text",   "").strip()
                lines.append(f"- [{t}·{s}] {text}")
            sections.append("以下是关于用户的已知信息：\n" + "\n".join(lines))

        if wiki:
            lines = []
            for m in wiki:
                cite = m.get("source", "")
                text = m.get("text", "").strip()
                lines.append(f"- {text} {cite}".rstrip())
            sections.append("相关文档知识（wiki）：\n" + "\n".join(lines))

        return "\n\n".join(sections)
