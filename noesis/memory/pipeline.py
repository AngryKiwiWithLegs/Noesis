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


# Domain keyword map for supersession detection.
# Contradictory stances share a DOMAIN (e.g. both are "database" stances) but
# assert different ENTITIES (MySQL vs PostgreSQL). This map identifies which
# domain a stance is about, so _maybe_supersede can match old-vs-new on domain.
# Keys are domain labels; values are keyword lists (case-insensitive substring).
_DOMAIN_KEYWORDS = {
    "database": ["mysql", "postgres", "postgresql", "sqlite", "mongodb", "redis",
                 "memcached", "dynamodb", "cassandra", "clickhouse", "duckdb",
                 "database", "oracle", "mariadb", "cockroach"],
    "language": ["python", "java", "javascript", "typescript", "rust", "go",
                 "golang", "c++", "c#", "ruby", "php", "kotlin", "swift", "scala",
                 "elixir", "haskell", "perl", "lua", "dart", "julia", "bash"],
    "cloud": ["aws", "gcp", "azure", "cloudflare", "vercel", "netlify",
              "heroku", "digitalocean", "linode", "kubernetes", "docker", "swarm"],
    "frontend": ["react", "vue", "angular", "svelte", "solid", "tailwind",
                 "bootstrap", "css", "webpack", "vite", "rollup", "grunt", "gulp"],
    "api": ["rest", "graphql", "grpc", "soap", "rpc", "webhook", "openapi"],
    "editor": ["vim", "neovim", "emacs", "vscode", "sublime", "helix", "intellij"],
    "messaging": ["kafka", "rabbitmq", "nats", "sqs", "pubsub", "pulsar"],
    "search": ["elasticsearch", "meilisearch", "solr", "typesense", "algolia"],
    "ci": ["jenkins", "github actions", "gitlab ci", "circleci", "travis",
           "drone", "buildkite", "teamcity"],
    "version_control": ["git", "svn", "mercurial", "perforce", "fossil"],
    "webserver": ["nginx", "apache", "caddy", "traefik", "haproxy", "envoy"],
    "baas": ["firebase", "supabase", "appwrite", "amplify", "parse"],
    "payments": ["stripe", "adyen", "paypal", "square", "braintree", "postmark", "sendgrid"],
    "data_format": ["json", "xml", "yaml", "toml", "protobuf", "avro", "msgpack"],
    "orchestration": ["kubernetes", "swarm", "nomad", "mesos", "ecs"],
    "email": ["sendgrid", "postmark", "mailgun", "ses", "mailchimp"],
}


def _stance_domain(text: str) -> str | None:
    """Identify the domain of a stance from its keywords. Returns the domain
    label (e.g. 'database') or None if no domain keyword is found."""
    t = text.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return domain
    return None


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

        # Supersession: if this new node replaces a prior contradictory stance
        # in the same cluster, retire the old node so it stops being injected.
        # Only fires when the new node itself reaches injectable status — there
        # is no point retiring an old stance in favour of a tentative one.
        if new_status in ("provisional", "settled"):
            try:
                self._maybe_supersede(hash_id, candidate, task.user_id, history)
            except Exception as e:
                logger.warning(f"Supersession check failed [{hash_id[:8]}]: {e}")

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

    def _maybe_supersede(
        self,
        new_hash:    str,
        candidate:   ThoughtCandidate,
        user_id:     str,
        history:     list[dict],
    ) -> None:
        """Detect and apply supersession: when the new node is a stance
        replacement (contains a replacement verb) for a prior stance on the
        SAME domain, mark the old node 'superseded' so it stops being injected.

        Design rationale: contradictory stances ("I use MySQL" → "I switched to
        PostgreSQL") share a TOPIC DOMAIN (databases) but assert different
        ENTITIES (MySQL vs PostgreSQL). Embedding similarity is *low* for such
        pairs (the model encodes the differing entity names), so we detect the
        shared domain via a keyword map instead. The replacement verb is the
        key gate that distinguishes a mind-change from corroboration.

        Fires when ALL hold:
          1. new text has a replacement verb (switched/moved/migrated/...)
          2. new node type is a stance (preference/position/identity)
          3. an old node exists with: same type, injectable status, and the
             SAME domain keyword (e.g. both mention a DB name, both a language)
        """
        from ..thoughts.confidence import has_replacement_verb

        if not has_replacement_verb(candidate.text):
            return

        SUPERSedeABLE_TYPES = ("preference", "position", "identity")
        if candidate.type not in SUPERSedeABLE_TYPES:
            return

        # Determine the domain of the new stance. If we can't identify a
        # domain, we can't safely supersede (too risky to match anything).
        new_domain = _stance_domain(candidate.text)
        if new_domain is None:
            return

        # Scan history for a prior stance in the SAME domain
        for rec in history:
            if rec.get("hash_id") == new_hash:
                continue
            # Type check: relaxed — a mind-change can cross preference↔position
            # (e.g. "I prefer MySQL" is a preference, "I switched to PostgreSQL"
            # is classified as a position by the extractor). What matters is
            # that both are stance-like (preference/position/identity), not
            # that they share the exact type label.
            if rec.get("type") not in SUPERSedeABLE_TYPES:
                continue
            if rec.get("status") not in ("provisional", "settled"):
                continue

            old_domain = _stance_domain(rec.get("text", ""))
            if old_domain != new_domain:
                continue

            # Found a contradicting prior stance in the same domain — supersede.
            old_hash = rec["hash_id"]
            self.vector_store.update(old_hash, {
                "status": "superseded",
                "superseded_by": new_hash,
            })
            if self.cold_store:
                try:
                    self.cold_store.mark_superseded(old_hash, new_hash)
                except Exception as e:
                    logger.warning(f"Cold-store supersede failed [{old_hash[:8]}]: {e}")
            logger.info(
                f"Superseded [{old_hash[:8]}] '{rec.get('text','')[:40]}' "
                f"→ [{new_hash[:8]}] '{candidate.text[:40]}' (domain={new_domain})"
            )
            # Only supersede one prior stance per new node — the first match.
            return

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
