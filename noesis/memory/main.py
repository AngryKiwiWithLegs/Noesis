"""
noesis/memory/main.py  — Week 2

Two-phase commit:
  Phase 1 (sync <15ms): embed → similarity check → insert tentative
  Phase 2 (async):      LLM extraction → confidence → status → cold store
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from ..vector_stores.sqlite_vec import SqliteVecStore
from ..embeddings.local import LocalEmbedding
from ..cold_stores.obsidian import ObsidianStore
from ..cold_stores.base import ColdStoreBase
from ..thoughts.types import ConsolidationTask, ThoughtCandidate

logger = logging.getLogger(__name__)

URGENT_THRESHOLD = 0.95


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _token_est(text: str) -> int:
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cjk // 2 + (len(text) - cjk) // 4 + 1


class Memory:
    """
    Noesis Memory — public API, Mem0-compatible.

    Key read interface:
        build_context(query, user_id) → str for system prompt injection

    Key write interface:
        add(messages, user_id)        → stores thought, returns {results:[...]}
    """

    def __init__(
        self,
        vector_store:    SqliteVecStore,
        embedding_model: LocalEmbedding,
        cold_store:      Optional[ColdStoreBase] = None,
        pipeline:        Any = None,
        retriever:       Any = None,
        context_builder: Any = None,
    ):
        self.vector_store = vector_store
        self.embedding    = embedding_model
        self.cold_store   = cold_store
        self._pipeline    = pipeline
        self._retriever   = retriever
        self._ctx_builder = context_builder
        self._synced:    set[str]         = set()
        self._last_sync: dict[str, float] = {}

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "Memory":
        vs_cfg   = config.get("vector_store", {}).get("config", {})
        db_path  = Path(vs_cfg.get("db_path", "~/.noesis/hot.db")).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store = SqliteVecStore(str(db_path))

        emb_cfg  = config.get("embedder", {}).get("config", {})
        embedding = LocalEmbedding(emb_cfg.get("model", "all-MiniLM-L6-v2"))

        cold_store: Optional[ColdStoreBase] = None
        if cs := config.get("cold_store"):
            vp = cs.get("config", {}).get("vault_path", "~/NoesisVault")
            cold_store = ObsidianStore(vp)

        inst = cls(vector_store=vector_store,
                   embedding_model=embedding,
                   cold_store=cold_store)

        if llm_cfg := config.get("llm"):
            inst._attach_pipeline(llm_cfg)

        # Week 3: attach HybridRetriever + ContextBuilder
        inst._attach_retriever(config)

        return inst

    @classmethod
    def from_config_file(cls, path: str) -> "Memory":
        import yaml
        with open(Path(path).expanduser()) as f:
            return cls.from_config(yaml.safe_load(f))

    def _attach_pipeline(self, llm_cfg: dict):
        try:
            from ..thoughts.confidence import ConfidenceScorer
            from .pipeline import ConsolidationPipeline

            # Resolve api_key: explicit config first, then env var, else fall back.
            provider = llm_cfg.get("provider", "anthropic")
            model    = llm_cfg.get("model", "claude-haiku-4-5-20251001")
            api_key  = llm_cfg.get("api_key")
            if not api_key:
                env_var = {
                    "anthropic": "ANTHROPIC_API_KEY",
                    "openai":    "OPENAI_API_KEY",
                }.get(provider, "OPENAI_API_KEY")
                api_key = os.environ.get(env_var)

            if api_key:
                from ..thoughts.extractor import CloudLLMExtractor
                extractor = CloudLLMExtractor(
                    provider=provider, model=model, api_key=api_key,
                )
                logger.info("Pipeline attached with Cloud LLM extractor (%s)", provider)
            else:
                from ..thoughts.extractor import MockExtractor
                extractor = MockExtractor()
                logger.warning(
                    "No API key for provider '%s' (config or env); "
                    "falling back to MockExtractor. Thoughts will NOT be "
                    "classified/consolidated — set an API key to enable extraction.",
                    provider,
                )
            self._pipeline = ConsolidationPipeline(
                vector_store=self.vector_store,
                embedding   =self.embedding,
                cold_store  =self.cold_store,
                extractor   =extractor,
                scorer      =ConfidenceScorer(),
            )
        except Exception as e:
            logger.warning(f"Pipeline attach failed: {e}")

    def _attach_retriever(self, config: dict):
        """Build HybridRetriever + ContextBuilder and attach to self."""
        try:
            from ..retrieval.hybrid import HybridRetriever
            from ..context.builder import ContextBuilder

            vault_path = None
            if cs := config.get("cold_store"):
                vault_path = cs.get("config", {}).get("vault_path")

            linker = None
            if vault_path:
                try:
                    from ..graphs.obsidian_linker import ObsidianLinker
                    linker = ObsidianLinker(
                        vault_path=vault_path,
                        vector_store=self.vector_store,
                        embedding_model=self.embedding,
                    )
                except Exception as le:
                    logger.debug(f"Linker not attached: {le}")

            retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedding_model=self.embedding,
                linker=linker,
            )
            self._retriever   = retriever
            self._ctx_builder = ContextBuilder(
                vector_store=self.vector_store,
                embedding_model=self.embedding,
                retriever=retriever,
                linker=linker,
            )
            logger.info("HybridRetriever + ContextBuilder attached")
        except Exception as e:
            logger.warning(f"Retriever attach failed: {e}")

    def attach_pipeline(self, pipeline) -> "Memory":
        """Inject a pre-built pipeline (used in tests)."""
        self._pipeline = pipeline
        return self

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        messages,
        user_id:       str = "default",
        type:          str = "position",
        source_tool:   str = "",
        session_id:    str = "",
        topic_cluster: str = "",
        **kwargs,
    ) -> dict:
        text = self._to_text(messages)
        if not text.strip():
            return {"results": []}

        h   = _hash(text)
        vec = self.embedding.embed(text)

        # Phase 1: sync, must stay <15ms
        neighbors = self.vector_store.search(
            vec, top_k=5, filter={"user_id": user_id}
        )
        max_score = max((n.get("score", 0) for n in neighbors), default=0.0)

        self.vector_store.insert(h, vec, {
            "text": text, "type": type, "status": "tentative",
            "confidence": 0.0, "user_id": user_id,
            "source_tool": source_tool, "source_session": session_id,
            "topic_cluster": topic_cluster, "created_at": time.time(),
        })

        # Phase 2: async
        if self._pipeline is not None:
            task = ConsolidationTask(
                hash_id  = h,
                candidate= ThoughtCandidate(
                    type=type, text=text,
                    initial_confidence=0.2, assertion_strength=0.5,
                    source_tool=source_tool, source_session=session_id,
                    topic_cluster=topic_cluster,
                ),
                neighbors=neighbors,
                user_id  =user_id,
            )
            if max_score > URGENT_THRESHOLD:
                self._pipeline.put_urgent(task)
            else:
                self._pipeline.put(task)
        else:
            if self.cold_store:
                try:
                    self.cold_store.write(h, {
                        "text": text, "type": type, "status": "tentative",
                        "confidence": 0.0, "user_id": user_id,
                        "source_tool": source_tool, "source_session": session_id,
                        "topic_cluster": topic_cluster,
                    })
                except Exception as e:
                    logger.warning(f"Cold write [{h[:8]}]: {e}")

        return {"results": [{"id": h, "status": "tentative"}]}

    # ── Read ──────────────────────────────────────────────────────────────────

    def build_context(
        self,
        query: str,
        user_id: str = "default",
        budget_tokens: int = 1200,
    ) -> str:
        self._sync_if_needed(user_id)

        # Week 3: delegate to ContextBuilder (hybrid retrieval + graph expansion)
        if self._ctx_builder is not None:
            return self._ctx_builder.build(query, user_id, budget_tokens)

        # Fallback: inline three-signal retrieval (Week 1/2)
        vec = self.embedding.embed(query)
        candidates: list[dict] = []
        candidates += self.vector_store.search(
            vec, top_k=5, filter={"user_id": user_id, "status": "settled"})
        candidates += [
            p for p in self.vector_store.search(
                vec, top_k=4,
                filter={"user_id": user_id, "status": "provisional"},
                min_score=0.68)
        ]
        candidates += self.vector_store.get_recent(user_id, n=3)
        candidates += self.vector_store.get_by_type(
            user_id, types=["identity", "preference"], top_k=3)
        return self._fmt(candidates, budget_tokens)

    def attach_context_builder(self, builder) -> "Memory":
        """Inject a pre-built ContextBuilder (used in tests)."""
        self._ctx_builder = builder
        return self

    def search(
        self, query: str, user_id: str = "default",
        top_k: int = 5, limit: int = 0, **kwargs,
    ) -> list[dict]:
        self._sync_if_needed(user_id)
        k = limit if limit > 0 else top_k
        return self.vector_store.search(
            self.embedding.embed(query), top_k=k,
            filter={"user_id": user_id})

    def get(self, hash_id: str) -> Optional[str]:
        if self.cold_store:
            try:
                return self.cold_store.read(hash_id)
            except FileNotFoundError:
                pass
        n = self.vector_store.get(hash_id)
        return n["text"] if n else None

    def delete_all(self, user_id: str = "default"):
        self.vector_store.delete_all(user_id)

    def status(self, user_id: str = "default") -> dict:
        nodes = self.vector_store.get_all(user_id)
        c: dict[str, int] = {}
        for n in nodes:
            s = n.get("status", "tentative")
            c[s] = c.get(s, 0) + 1
        return {
            "total": len(nodes),
            "settled": c.get("settled", 0),
            "provisional": c.get("provisional", 0),
            "tentative": c.get("tentative", 0),
            "pipeline_depth": self._pipeline.queue_depth if self._pipeline else 0,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _sync_if_needed(self, user_id: str):
        if user_id in self._synced:
            return
        if self.cold_store:
            since = self._last_sync.get(user_id, 0.0)
            for hid in self.cold_store.scan_modified(since):
                try:
                    text = self.cold_store.read(hid)
                    if self.vector_store.exists(hid):
                        self.vector_store.update(hid, {"text": text})
                except Exception as e:
                    logger.warning(f"Sync [{hid[:8]}]: {e}")
            self._last_sync[user_id] = time.time()
        self._synced.add(user_id)

    def _fmt(self, candidates: list[dict], budget: int) -> str:
        seen, out, used = set(), [], 0
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
        if not out:
            return ""
        lines = [
            f"- [{m.get('type','?')}·{m.get('status','?')}] {m.get('text','').strip()}"
            for m in out
        ]
        return "以下是关于用户的已知信息：\n" + "\n".join(lines)

    @staticmethod
    def _to_text(messages) -> str:
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list):
            parts = []
            for m in messages:
                if isinstance(m, dict):
                    role, content = m.get("role", ""), m.get("content", "")
                    if content:
                        parts.append(f"{role}: {content}" if role else content)
                elif isinstance(m, str):
                    parts.append(m)
            return "\n".join(parts)
        return str(messages)
