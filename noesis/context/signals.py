"""
noesis/context/signals.py

Four signals that feed ContextBuilder.

Signal 1 — semantic_signal:
    Uses HybridRetriever with 1-hop graph expansion.
    Finds nodes that are topically related, even if not directly similar.

Signal 2 — recency_signal:
    Returns the N most recent non-tentative nodes.
    "What just happened" is almost always relevant.

Signal 3 — core_fact_signal:
    Returns identity + preference nodes unconditionally.
    These are always relevant: who you are never goes out of context.

Signal 4 — wiki_signal:
    Searches compiled wiki/ pages for query-relevant knowledge.
    This is the LLM-Wiki integration: knowledge compiled from documents
    grounds the positions stored in thoughts/. (NOESIS_SCHEMA.md)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..vector_stores.sqlite_vec import SqliteVecStore

logger = logging.getLogger(__name__)


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


def wiki_signal(
    query:      str,
    vault_path: str,
    embedding_model = None,
    top_k:      int = 3,
) -> list[dict]:
    """
    Search compiled wiki/ pages for knowledge relevant to the query.

    Returns dicts shaped like thought nodes (id/text/type/status) so the
    ContextBuilder can treat them uniformly. Wiki entries carry
    type="wiki" and a [[wiki/...]] citation in the text.

    Ranking: embedding similarity when an embedding model is available,
    with a BM25 keyword fallback (reuses rank-bm25). If neither is
    available, falls back to substring matching.
    """
    wiki_dir = Path(vault_path).expanduser() / "wiki"
    if not wiki_dir.exists():
        return []

    # Gather pages (exclude index/log)
    pages = []
    for p in sorted(wiki_dir.glob("*.md")):
        if p.stem in ("index", "log"):
            continue
        try:
            from ..wiki.writer import WikiWriter
            writer = WikiWriter(vault_path)
            page = writer.read_page(p.stem)
            if page and page.body.strip():
                pages.append(page)
        except Exception:
            continue
    if not pages:
        return []

    query_lower = query.lower()
    scored: list[tuple[float, "WikiPage"]] = []

    # Prefer embedding similarity (most accurate)
    if embedding_model is not None:
        try:
            qvec = embedding_model.embed(query)
            for page in pages:
                pvec = embedding_model.embed(page.title + " " + page.body[:500])
                sim = _cosine(qvec, pvec)
                scored.append((sim, page))
            scored.sort(key=lambda x: x[0], reverse=True)
        except Exception as e:
            logger.debug(f"wiki_signal embedding search failed: {e}")
            scored = []
    else:
        scored = [(_bm25_like(query_lower, page), page) for page in pages]
        scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    for score, page in scored[:top_k]:
        if score <= 0:
            continue
        # Shape like a thought node so ContextBuilder handles it uniformly
        out.append({
            "id": f"wiki/{page.page_id}",
            "text": f"{page.body[:300]}",
            "type": "wiki",
            "status": page.status,
            "topic_cluster": page.topic_cluster,
            "source": f"[[wiki/{page.page_id}]]",
            "score": float(score),
        })
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _bm25_like(query_lower: str, page) -> float:
    """Cheap keyword-overlap score (fallback when no embedding model).

    Counts how many query terms appear in the page title+body, weighted
    slightly toward title hits. Good enough to surface obviously-relevant
    pages without a full BM25 index.
    """
    terms = re.findall(r"\w+", query_lower)
    if not terms:
        return 0.0
    title_lower = page.title.lower()
    body_lower = page.body.lower()
    score = 0.0
    for t in terms:
        if len(t) < 2:
            continue
        if t in title_lower:
            score += 2.0
        if t in body_lower:
            score += 1.0
    return score
