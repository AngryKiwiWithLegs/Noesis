"""
noesis/memory/pipeline.py

Async consolidation pipeline.

Phase 1 (in Memory.add, sync, <15ms):
  embed → similarity check → insert as 'tentative'

Phase 2 (here, background thread):
  LLM extraction → confidence scoring → status update → cold store → graph links

Two queues ensure urgent tasks (high-similarity conflicts) are resolved first.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Optional

from ..thoughts.types import ConsolidationTask, ThoughtCandidate, ThoughtNode

if TYPE_CHECKING:
    from ..vector_stores.sqlite_vec import SqliteVecStore
    from ..cold_stores.base import ColdStoreBase
    from ..thoughts.extractor import AbstractExtractor
    from ..thoughts.confidence import ConfidenceScorer

logger = logging.getLogger(__name__)


class ConsolidationPipeline:
    """
    Background worker that runs Phase 2 of every add() call.

    Processing order:
      1. Drain urgent queue first (similarity > 0.95 conflicts)
      2. Then normal queue

    For each task:
      a. Run LLM extractor on raw text → ThoughtCandidates
      b. Score confidence for each candidate
      c. Update/create nodes in hot store
      d. Write to cold store (Obsidian vault)
      e. Trigger graph linker (if available)
    """

    def __init__(
        self,
        vector_store: "SqliteVecStore",
        embedding,
        cold_store:   Optional["ColdStoreBase"]   = None,
        extractor:    Optional["AbstractExtractor"] = None,
        scorer:       Optional["ConfidenceScorer"]  = None,
        linker:       Optional[object]              = None,
    ):
        self.vector_store = vector_store
        self.embedding    = embedding
        self.cold_store   = cold_store
        self.extractor    = extractor
        self.scorer       = scorer
        self.linker       = linker

        self._urgent: queue.Queue[ConsolidationTask] = queue.Queue()
        self._normal: queue.Queue[ConsolidationTask] = queue.Queue()
        self._errors: list[tuple[str, Exception]]    = []  # for tests

        self._thread = threading.Thread(
            target=self._run,
            name="noesis-pipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info("ConsolidationPipeline started")

    # ── Enqueue ───────────────────────────────────────────────────────────────

    def put(self, task: ConsolidationTask):
        self._normal.put(task)

    def put_urgent(self, task: ConsolidationTask):
        task.urgent = True
        self._urgent.put(task)

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def queue_depth(self) -> int:
        return self._urgent.qsize() + self._normal.qsize()

    def drain(self, timeout: float = 5.0) -> bool:
        """Block until both queues are empty (useful in tests)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._urgent.empty() and self._normal.empty():
                time.sleep(0.05)   # one more short wait for in-flight tasks
                return True
            time.sleep(0.05)
        return False

    # ── Worker ────────────────────────────────────────────────────────────────

    def _run(self):
        while True:
            task = self._next_task()
            try:
                self._process(task)
            except Exception as e:
                logger.error(f"Pipeline error [{task.hash_id[:8]}]: {e}", exc_info=True)
                self._errors.append((task.hash_id, e))
            finally:
                # Mark done on whichever queue it came from
                try:
                    if task.urgent:
                        self._urgent.task_done()
                    else:
                        self._normal.task_done()
                except Exception:
                    pass

    def _next_task(self) -> ConsolidationTask:
        """Drain urgent queue before touching normal queue."""
        try:
            return self._urgent.get_nowait()
        except queue.Empty:
            return self._normal.get()   # blocks until available

    # ── Processing ────────────────────────────────────────────────────────────

    def _process(self, task: ConsolidationTask):
        t0 = time.perf_counter()

        # 1. Extract thoughts from raw text via LLM
        candidates = self._extract(task)

        if not candidates:
            # Nothing worth keeping — delete the tentative placeholder
            self.vector_store.soft_delete(task.hash_id)
            logger.debug(f"[{task.hash_id[:8]}] No thoughts extracted, node deleted")
            return

        # 2. Process first candidate → update existing node
        primary = candidates[0]
        self._apply_candidate(task.hash_id, primary, task)

        # 3. Additional candidates → create new nodes
        for extra in candidates[1:]:
            self._create_extra_node(extra, task)

        elapsed = time.perf_counter() - t0
        logger.debug(
            f"[{task.hash_id[:8]}] Pipeline done in {elapsed*1000:.0f}ms, "
            f"{len(candidates)} thought(s)"
        )

    def _extract(self, task: ConsolidationTask) -> list[ThoughtCandidate]:
        if self.extractor is None:
            # No extractor: treat raw text as a low-confidence position
            return [task.candidate] if task.candidate else []

        raw_text = (
            task.candidate.text
            if task.candidate
            else self.vector_store.get(task.hash_id) or {}
        )
        if isinstance(raw_text, dict):
            raw_text = raw_text.get("text", "")

        return self.extractor.extract(
            raw_text,
            source_tool=task.candidate.source_tool if task.candidate else "",
            session_id =task.candidate.source_session if task.candidate else "",
        )

    def _apply_candidate(
        self,
        hash_id:   str,
        candidate: ThoughtCandidate,
        task:      ConsolidationTask,
    ):
        import hashlib

        # Score confidence
        history = self.vector_store.get_all(task.user_id)
        node = ThoughtNode(
            hash_id       = hash_id,
            type          = candidate.type,
            text          = candidate.text,
            user_id       = task.user_id,
            source_tool   = candidate.source_tool,
            source_session= candidate.source_session,
            topic_cluster = candidate.topic_cluster,
        )

        score      = self._score(node, history)
        new_status = self._next_status(score, "tentative")

        # Update hot store
        self.vector_store.update(hash_id, {
            "text":          candidate.text,
            "type":          candidate.type,
            "status":        new_status,
            "confidence":    score,
            "topic_cluster": candidate.topic_cluster,
            "source_tool":   candidate.source_tool,
        })

        # Re-embed with cleaned text (might differ from raw)
        # TODO Week 3: update vector in vec_items too

        # Update cold store
        if self.cold_store:
            self._cold_write(hash_id, candidate, task.user_id, new_status, score)

        # Graph linking
        if self.linker:
            try:
                self.linker.link(hash_id)
            except Exception as e:
                logger.warning(f"Graph link failed [{hash_id[:8]}]: {e}")

    def _create_extra_node(
        self,
        candidate: ThoughtCandidate,
        task:      ConsolidationTask,
    ):
        import hashlib as _h

        h   = _h.sha256(candidate.text.encode()).hexdigest()[:12]
        vec = self.embedding.embed(candidate.text)

        history    = self.vector_store.get_all(task.user_id)
        node       = ThoughtNode(
            hash_id=h, type=candidate.type,
            text=candidate.text, user_id=task.user_id,
            source_tool=candidate.source_tool,
            source_session=candidate.source_session,
            topic_cluster=candidate.topic_cluster,
        )
        score      = self._score(node, history)
        new_status = self._next_status(score, "tentative")

        self.vector_store.insert(h, vec, {
            "text":           candidate.text,
            "type":           candidate.type,
            "status":         new_status,
            "confidence":     score,
            "user_id":        task.user_id,
            "source_tool":    candidate.source_tool,
            "source_session": candidate.source_session,
            "topic_cluster":  candidate.topic_cluster,
            "created_at":     time.time(),
        })

        if self.cold_store:
            self._cold_write(h, candidate, task.user_id, new_status, score)

    def _cold_write(
        self,
        hash_id:    str,
        candidate:  ThoughtCandidate,
        user_id:    str,
        status:     str,
        confidence: float,
    ):
        try:
            existing = self.cold_store.read(hash_id)
            # File exists → just update status/confidence in frontmatter
            self.cold_store.update_status(hash_id, status, confidence)
        except FileNotFoundError:
            self.cold_store.write(hash_id, {
                "text":           candidate.text,
                "type":           candidate.type,
                "status":         status,
                "confidence":     confidence,
                "user_id":        user_id,
                "source_tool":    candidate.source_tool,
                "source_session": candidate.source_session,
                "topic_cluster":  candidate.topic_cluster,
            })
        except Exception as e:
            logger.warning(f"Cold write failed [{hash_id[:8]}]: {e}")

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def _score(self, node: ThoughtNode, history: list[dict]) -> float:
        if self.scorer is None:
            return node.confidence or 0.2
        return self.scorer.score(node, history)

    @staticmethod
    def _next_status(score: float, current: str) -> str:
        from ..thoughts.types import THRESHOLD_PROVISIONAL, THRESHOLD_SETTLED
        if score >= THRESHOLD_SETTLED:
            return "settled"
        if score >= THRESHOLD_PROVISIONAL:
            return "provisional"
        return current
