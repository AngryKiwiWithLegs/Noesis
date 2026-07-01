"""
noesis/wiki/lint.py

The Lint operation. Karpathy's pattern treats lint as non-optional: a wiki
that isn't audited rots. This pass checks four classes of problem and,
where it can, resolves them.

Implements NOESIS_SCHEMA.md rule #2 (the question→answer flow):
  "LLM Wiki lint pass scans for status:open question nodes, adds them to
   the knowledge-ingestion priority queue. When a wiki node answers a
   question, it sets: status: answered, answered_by: [[wiki/xxx]]"

Checks:
  1. open_questions  — thoughts of type:question with no answered_by ref
  2. orphan_pages    — wiki pages no thought or cluster links to
  3. contradictions  — wiki pages in the same cluster with opposing stances
  4. coverage        — clusters that have thoughts but no wiki page (knowledge gaps)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from .writer import WikiWriter
from .extractor import infer_cluster

logger = logging.getLogger(__name__)

# Replacement-verb domains reused from the supersession logic — two wiki
# pages in the same domain asserting different entities likely contradict.
try:
    from ..memory.pipeline import _stance_domain, _DOMAIN_KEYWORDS
except Exception:  # pragma: no cover - pipeline import is best-effort
    _stance_domain = None
    _DOMAIN_KEYWORDS = {}


class WikiLinter:
    """Audits the wiki layer and resolves what it can."""

    def __init__(
        self,
        vault_path: str,
        writer: Optional[WikiWriter] = None,
        vector_store=None,
    ):
        self.vault = Path(vault_path).expanduser().resolve()
        self.writer = writer or WikiWriter(vault_path)
        self.vs = vector_store

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Run all checks, return a report dict. Safe to call repeatedly."""
        report = {
            "open_questions": self.find_open_questions(),
            "orphan_pages": self.find_orphans(),
            "contradictions": self.find_contradictions(),
            "coverage_gaps": self.find_coverage_gaps(),
        }
        summary = {k: len(v) for k, v in report.items()}
        logger.info("Wiki lint complete: %s", summary)
        self.writer.append_log("lint", "wiki", str(summary))
        return report

    # ── Check 1: open questions (schema rule #2, read side) ───────────────────

    def find_open_questions(self) -> list[dict]:
        """Find type:question thoughts that have no answered_by field.

        These are the schema's "knowledge-ingestion priority queue": open
        questions the wiki should aim to answer.
        """
        if self.vs is None:
            return []
        out = []
        # Walk all users — questions are a global knowledge need.
        for user_id in self._all_user_ids():
            nodes = self.vs.get_all(user_id)
            for n in nodes:
                if n.get("type") != "question":
                    continue
                status = n.get("status", "")
                # 'answered' is set by mark_answered; skip already-answered
                if status == "answered":
                    continue
                # Check the cold-store frontmatter for answered_by
                if self._has_answered_by(n.get("id", "")):
                    continue
                out.append({
                    "hash": n.get("id", ""),
                    "text": n.get("text", ""),
                    "cluster": n.get("topic_cluster", ""),
                    "user_id": user_id,
                })
        return out

    # ── Check 2: orphan pages ─────────────────────────────────────────────────

    def find_orphans(self) -> list[str]:
        """Wiki pages that no thought or cluster links to.

        An orphan has no inbound [[wiki/page_id]] ref from anywhere in
        thoughts/ or clusters/. It may still be reachable via index.md,
        but it's graph-isolated.
        """
        page_ids = set(self.writer.list_pages())
        if not page_ids:
            return []

        # Gather all [[wiki/...]] refs that exist in thoughts/ and clusters/
        referenced: set[str] = set()
        for sub in ("thoughts", "clusters"):
            d = self.vault / sub
            if not d.exists():
                continue
            for p in d.glob("*.md"):
                content = p.read_text(encoding="utf-8", errors="replace")
                for m in re.finditer(r"\[\[wiki/([^\]]+)\]\]", content):
                    referenced.add(m.group(1).strip())

        return sorted(page_ids - referenced)

    # ── Check 3: contradictions ───────────────────────────────────────────────

    def find_contradictions(self) -> list[tuple[str, str, str]]:
        """Wiki pages in the same cluster that assert different stance domains.

        Returns (page_a, page_b, domain) tuples. Uses the same _stance_domain
        logic as the supersession detector: two pages in the same cluster
        whose bodies mention different entities in the same domain (e.g.
        'use PostgreSQL' vs 'use MySQL') are likely contradictory.
        """
        pages = []
        for pid in self.writer.list_pages():
            p = self.writer.read_page(pid)
            if p:
                pages.append(p)

        # Group by cluster
        by_cluster: dict[str, list] = {}
        for p in pages:
            by_cluster.setdefault(p.topic_cluster or "general", []).append(p)

        contradictions = []
        if _stance_domain is None:
            return contradictions

        for cluster, group in by_cluster.items():
            if len(group) < 2:
                continue
            domains = {}
            for p in group:
                dom = _stance_domain(p.body)
                if dom:
                    domains.setdefault(dom, []).append(p.page_id)
            # If the same cluster has pages in the same domain but they're
            # different pages, flag them
            for dom, pids in domains.items():
                if len(pids) >= 2:
                    contradictions.append((pids[0], pids[1], dom))
        return contradictions

    # ── Check 4: coverage gaps ────────────────────────────────────────────────

    def find_coverage_gaps(self) -> list[str]:
        """Clusters that have thoughts but no wiki page — knowledge gaps.

        These are domains the user has opinions/questions about but the wiki
        hasn't compiled any source material for yet.
        """
        wiki_clusters = set()
        for pid in self.writer.list_pages():
            p = self.writer.read_page(pid)
            if p and p.topic_cluster:
                wiki_clusters.add(p.topic_cluster)

        thought_clusters = self._all_thought_clusters()
        return sorted(thought_clusters - wiki_clusters - {"general", ""})

    # ── Resolution: mark a question answered (schema rule #2, write side) ─────

    def mark_answered(self, question_hash: str, wiki_page_id: str) -> bool:
        """Mark a question thought as answered by a wiki page.

        Implements schema rule #2's write side:
          status: answered
          answered_by: [[wiki/{wiki_page_id}]]

        Patches the thought's frontmatter in the vault (cold store) and, if
        a vector_store is attached, updates the hot-store status too.
        """
        # Verify the wiki page exists
        if not (self.vault / "wiki" / f"{wiki_page_id}.md").exists():
            logger.warning("Cannot answer with non-existent wiki page: %s", wiki_page_id)
            return False

        # Patch the thought's frontmatter
        from ..cold_stores.obsidian import ObsidianStore
        # We reuse the frontmatter-patching logic; ObsidianStore works by hash.
        store = ObsidianStore.__new__(ObsidianStore)
        store.root = self.vault
        store._patch_frontmatter(question_hash, {
            "status": "answered",
            # Quote the value so YAML reads it as a string, not a nested list
            # (the [[ ]] syntax would otherwise parse as a YAML flow sequence).
            "answered_by": f'"[[wiki/{wiki_page_id}]]"',
        })

        # Update hot-store status if available
        if self.vs is not None and self.vs.exists(question_hash):
            self.vs.update(question_hash, {"status": "answered"})

        self.writer.append_log(
            "answer", wiki_page_id,
            f"question {question_hash[:8]} answered",
        )
        logger.info("Question %s answered by wiki/%s", question_hash[:8], wiki_page_id)
        return True

    # ── Internals ──────────────────────────────────────────────────────────────

    def _all_user_ids(self) -> list[str]:
        """Best-effort enumeration of user IDs in the hot store."""
        if self.vs is None:
            return []
        # get_all requires a user_id; we scan the DB directly if possible
        try:
            rows = self.vs._con.execute(
                "SELECT DISTINCT user_id FROM items WHERE user_id != ''"
            ).fetchall()
            return [r[0] for r in rows if r[0]]
        except Exception:
            return []

    def _all_thought_clusters(self) -> set[str]:
        """Clusters that appear in the hot store's thoughts."""
        if self.vs is None:
            return set()
        try:
            rows = self.vs._con.execute(
                "SELECT DISTINCT topic_cluster FROM items "
                "WHERE topic_cluster IS NOT NULL AND topic_cluster != ''"
            ).fetchall()
            return {r[0] for r in rows if r[0]}
        except Exception:
            return set()

    def _has_answered_by(self, hash_id: str) -> bool:
        """Check whether a thought's frontmatter has an answered_by field."""
        path = self.vault / "thoughts" / f"{hash_id}.md"
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8", errors="replace")
        return bool(re.search(r"^answered_by:", content, re.MULTILINE))
