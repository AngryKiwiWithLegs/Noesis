"""
noesis/retrieval/hybrid.py

Three-tier retrieval:
  1. BM25  — exact keyword match (good for names, terms, numbers)
  2. Vec   — semantic similarity (good for paraphrases, concepts)
  3. RRF   — Reciprocal Rank Fusion with time-decay weighting
  4. CrossEncoder reranker (optional, enabled when >300 nodes)

Scale tiers:
  <300 nodes → single-pass vector only  (<10ms)
  300–2000   → BM25 + Vec + RRF        (~20ms)
  >2000      → full pipeline + reranker (~50ms)
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Half-life for each thought type (days). Identity never decays.
_HALF_LIFE: dict[str, Optional[float]] = {
    "event":         7.0,
    "question":     30.0,
    "position":     90.0,
    "preference":   90.0,
    "synthesis":   180.0,
    "contradiction": None,
    "identity":      None,
}


def _decay(node: dict, now: float) -> float:
    hl  = _HALF_LIFE.get(node.get("type", "position"))
    if hl is None:
        return 1.0
    age = (now - float(node.get("created_at", now))) / 86400
    return math.exp(-0.693 * age / hl)


class HybridRetriever:

    def __init__(self, vector_store, embedding_model, reranker=None, linker=None):
        self.vs       = vector_store
        self.emb      = embedding_model
        self.reranker = reranker
        self.linker   = linker   # ObsidianLinker, optional

        # BM25 cache per user_id
        self._bm25:       dict[str, object]     = {}
        self._bm25_docs:  dict[str, list[dict]] = {}
        self._bm25_built: dict[str, float]      = {}

    def search(
        self,
        query:     str,
        user_id:   str,
        filter:    Optional[dict] = None,
        top_k:     int  = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        total = self.vs.count(user_id=user_id)
        vec   = self.emb.embed(query)
        filt  = {**(filter or {}), "user_id": user_id}

        if total < 300:
            return self.vs.search(vec, top_k=top_k, filter=filt,
                                   min_score=min_score)

        fetch = top_k * 4
        vec_res  = self.vs.search(vec, top_k=fetch, filter=filt)
        bm25_res = self._bm25_search(query, fetch, user_id, filt)
        merged   = self._rrf(vec_res, bm25_res, fetch)

        # Apply status filter post-merge (BM25 doesn't filter)
        if filter and "status" in filter:
            wanted = filter["status"]
            if isinstance(wanted, str):
                merged = [m for m in merged if m.get("status") == wanted]
            elif isinstance(wanted, dict) and "$in" in wanted:
                merged = [m for m in merged if m.get("status") in wanted["$in"]]

        merged = [m for m in merged if m.get("score", 1) >= min_score]

        if total >= 300 and self.reranker is not None:
            return self.reranker.rerank(query, merged, top_k=top_k)

        return merged[:top_k]

    def search_with_graph_expansion(
        self,
        query:   str,
        user_id: str,
        top_k:   int = 5,
        hops:    int = 1,
    ) -> list[dict]:
        """
        Hybrid search + 1-hop graph expansion via [[]] vault links.

        Steps:
          1. Seed nodes from hybrid search (provisional + settled only)
          2. For each seed, follow [[]] links in Obsidian vault
          3. Collect expanded candidates (skip tentative)
          4. Re-rank expanded set with CrossEncoder if available
        """
        filter_ok = {"status": {"$in": ["provisional", "settled"]},
                     "user_id": user_id}

        seeds = self.search(query, user_id=user_id,
                            filter=filter_ok, top_k=top_k)

        if self.linker is None or hops == 0:
            return seeds

        # Graph expansion
        expanded: dict[str, dict] = {
            (s.get("id") or s.get("hash_id", "")): s
            for s in seeds
        }

        for seed in seeds:
            sid = seed.get("id") or seed.get("hash_id", "")
            try:
                for neighbour_id in self.linker.get_related_hashes(sid):
                    if neighbour_id in expanded:
                        continue
                    node = self.vs.get(neighbour_id)
                    if node and node.get("status") not in ("tentative", "superseded"):
                        expanded[neighbour_id] = node
            except Exception as e:
                logger.debug(f"Graph expand error ({sid[:8]}): {e}")

        candidates = list(expanded.values())

        if len(candidates) <= top_k:
            return candidates

        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, top_k=top_k)

        return candidates[:top_k]

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _bm25_search(
        self,
        query:   str,
        top_k:   int,
        user_id: str,
        filter:  dict,
    ) -> list[dict]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.debug("rank-bm25 not installed; skipping BM25 pass")
            return []

        self._maybe_rebuild_bm25(user_id, filter)
        bm25 = self._bm25.get(user_id)
        docs = self._bm25_docs.get(user_id, [])
        if bm25 is None or not docs:
            return []

        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])

        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                break
            d = dict(docs[i])
            d["score"] = float(min(scores[i] / 10.0, 1.0))  # normalise roughly
            results.append(d)
        return results

    def _maybe_rebuild_bm25(self, user_id: str, filter: dict):
        last = self._bm25_built.get(user_id, 0.0)
        if time.time() - last < 60:   # rebuild at most once per minute
            return
        try:
            from rank_bm25 import BM25Okapi
            docs = self.vs.get_all(user_id)
            corpus = [d.get("text", "").lower().split() for d in docs]
            self._bm25[user_id]       = BM25Okapi(corpus) if corpus else None
            self._bm25_docs[user_id]  = docs
            self._bm25_built[user_id] = time.time()
        except Exception as e:
            logger.warning(f"BM25 rebuild failed: {e}")

    # ── RRF with time decay ───────────────────────────────────────────────────

    def _rrf(
        self,
        vec_res:  list[dict],
        bm25_res: list[dict],
        top_k:    int,
        k:        int = 60,
    ) -> list[dict]:
        scores: dict[str, float] = {}
        docs:   dict[str, dict]  = {}
        now = time.time()

        for rank, r in enumerate(vec_res):
            nid = r.get("id") or r.get("hash_id", "")
            d   = _decay(r, now)
            scores[nid] = scores.get(nid, 0) + d / (k + rank + 1)
            docs[nid]   = r

        for rank, r in enumerate(bm25_res):
            nid = r.get("id") or r.get("hash_id", "")
            d   = _decay(r, now)
            scores[nid] = scores.get(nid, 0) + d / (k + rank + 1)
            docs[nid]   = r

        ordered = sorted(scores, key=lambda x: -scores[x])[:top_k]
        result  = []
        for nid in ordered:
            m = dict(docs[nid])
            m["score"] = scores[nid]
            result.append(m)
        return result


# ── CrossEncoder reranker ─────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Uses ms-marco-MiniLM-L-6-v2 to re-score candidates.
    Loaded lazily; only runs when >300 nodes in store.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model     = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading reranker '{self.model_name}' …")
                self._model = CrossEncoder(self.model_name)
            except ImportError as e:
                raise ImportError("pip install sentence-transformers") from e
        return self._model

    def rerank(
        self,
        query:      str,
        candidates: list[dict],
        top_k:      int = 5,
    ) -> list[dict]:
        if not candidates:
            return []
        try:
            pairs  = [(query, c.get("text", "")) for c in candidates]
            scores = self.model.predict(pairs)
            ranked = sorted(
                zip(candidates, scores),
                key=lambda x: -float(x[1]),
            )
            return [r[0] for r in ranked[:top_k]]
        except Exception as e:
            logger.warning(f"Reranker failed: {e}; returning unsorted")
            return candidates[:top_k]
