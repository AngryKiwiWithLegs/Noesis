"""
noesis/context/signals.py

Three signals that feed ContextBuilder.

Signal 1 — semantic_signal:
    Uses HybridRetriever with 1-hop graph expansion.
    Finds nodes that are topically related, even if not directly similar.

Signal 2 — recency_signal:
    Returns the N most recent non-tentative nodes.
    "What just happened" is almost always relevant.

Signal 3 — core_fact_signal:
    Returns identity + preference nodes unconditionally.
    These are always relevant: who you are never goes out of context.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..vector_stores.sqlite_vec import SqliteVecStore


def semantic_signal(
    query:    str,
    user_id:  str,
    retriever,
    top_k:    int = 8,
) -> list[dict]:
    """
    Semantic search + optional graph expansion.
    If retriever supports search_with_graph_expansion, uses it.
    Falls back to plain search otherwise.
    """
    if retriever is None:
        return []

    if hasattr(retriever, "search_with_graph_expansion"):
        return retriever.search_with_graph_expansion(
            query, user_id=user_id, top_k=top_k
        )

    return retriever.search(
        query,
        user_id=user_id,
        top_k=top_k,
        filter={"status": {"$in": ["provisional", "settled"]}},
    )


def recency_signal(
    user_id:      str,
    vector_store: "SqliteVecStore",
    n:            int = 3,
) -> list[dict]:
    """
    Most recent N confirmed nodes, regardless of semantic relevance.
    Always included — context of the last few exchanges matters.
    """
    return vector_store.get_recent(
        user_id, n=n,
        filter={"status": {"$in": ["provisional", "settled"]}},
    )


def core_fact_signal(
    user_id:      str,
    vector_store: "SqliteVecStore",
    top_k:        int = 3,
) -> list[dict]:
    """
    Identity and preference nodes — injected unconditionally.
    Who you are and how you like things done is always relevant.
    """
    return vector_store.get_by_type(
        user_id,
        types=["identity", "preference"],
        top_k=top_k,
    )
