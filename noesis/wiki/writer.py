"""
noesis/wiki/writer.py

Writes wiki/*.md pages into the Obsidian vault, plus the two Karpathy
special files:
  wiki/index.md  — routing catalog (one row per page; cheap to scan)
  wiki/log.md    — append-only chronological timeline of wiki actions

Mirrors noesis/cold_stores/obsidian.py conventions:
  - YAML frontmatter between --- fences
  - Required schema fields: hash · created · type · status · topic_cluster
  - Trailing `Related:` line for graph_linker-written [[wiki/...]] refs
  - _patch_frontmatter-style regex updates
  - pyyaml for frontmatter parsing
"""
from __future__ import annotations

import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .types import WikiPage, _now_iso

logger = logging.getLogger(__name__)

# Pages excluded from list_pages() / index rebuild — they are structural,
# not knowledge nodes.
_RESERVED = {"index", "log"}

# ── Template ──────────────────────────────────────────────────────────────────
# Same frontmatter shape as _THOUGHT_TEMPLATE (obsidian.py:29) so the existing
# graph_linker can match wiki pages by topic_cluster, plus wiki-specific fields.

_WIKI_TEMPLATE = """\
---
hash: {hash_id}
created: {created}
updated: {updated}
type: {page_type}
status: {status}
topic_cluster: {topic_cluster}
title: {title}
sources: {sources}
citations: {citations}
---
{body}

Related:
"""


class WikiWriter:
    """Reads and writes wiki/*.md pages into the vault."""

    def __init__(self, vault_path: str):
        self.root = Path(vault_path).expanduser().resolve()
        self.wiki_dir = self.root / "wiki"
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API: write ─────────────────────────────────────────────────────

    def write_page(self, page: WikiPage):
        """Create or overwrite wiki/{page_id}.md from a WikiPage."""
        path = self.page_path(page.page_id)
        content = _WIKI_TEMPLATE.format(
            hash_id=page.hash,
            created=page.created,
            updated=page.updated,
            page_type=page.page_type,
            status=page.status,
            topic_cluster=page.topic_cluster or "general",
            title=page.title,
            sources=_fmt_list(page.sources),
            citations=_fmt_list(page.citations),
            body=page.body.strip(),
        )
        path.write_text(content, encoding="utf-8")

    def upsert(self, page: WikiPage):
        """Insert or update, preserving the original `created` timestamp.

        On update: bump `updated`, merge sources/citations, and append body
        (per the schema's "same topic_cluster node: merge Related, never
        overwrite body" spirit). This makes ingest idempotent.
        """
        existing = self.read_page(page.page_id)
        if existing is None:
            self.write_page(page)
            return
        # Merge: keep created, refresh updated, union sources/citations,
        # append new body content if it isn't already present.
        merged_sources = _dedupe(existing.sources + page.sources)
        merged_cites = _dedupe(existing.citations + page.citations)
        body = existing.body
        if page.body.strip() and page.body.strip() not in body:
            body = (body.rstrip() + "\n\n" + page.body.strip()).strip()
        merged = WikiPage(
            page_id=page.page_id,
            title=page.title or existing.title,
            body=body,
            page_type=page.page_type or existing.page_type,
            topic_cluster=page.topic_cluster or existing.topic_cluster,
            status=page.status if page.status != "draft" else existing.status,
            sources=merged_sources,
            citations=merged_cites,
            related=existing.related,
            created=existing.created,   # preserve
            updated=_now_iso(),         # refresh
        )
        self.write_page(merged)

    def mark_answered(self, page_id: str):
        """Set status:answered on a page (question resolved by lint)."""
        self._patch_frontmatter(page_id, {"status": "answered"})

    def update_related(self, page_id: str, refs: list[str]):
        """Rewrite the Related: line of a wiki page with [[ref]] links."""
        path = self.page_path(page_id)
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")
        rel_str = " ".join(f"[[{r}]]" for r in refs) if refs else ""
        content = re.sub(
            r"^Related:.*$",
            f"Related: {rel_str}",
            content,
            flags=re.MULTILINE,
        )
        path.write_text(content, encoding="utf-8")

    # ── Public API: read ──────────────────────────────────────────────────────

    def read_page(self, page_id: str) -> Optional[WikiPage]:
        """Parse wiki/{page_id}.md back into a WikiPage, or None if absent."""
        path = self.page_path(page_id)
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        parts = raw.split("---", 2)
        if len(parts) < 3:
            return None
        fm = _parse_frontmatter(parts[1])
        body = parts[2]
        # Strip the trailing Related: line from the body
        body_lines = [l for l in body.splitlines() if not l.startswith("Related:")]
        body_text = "\n".join(body_lines).strip()
        return WikiPage(
            page_id=page_id,
            title=fm.get("title", page_id),
            body=body_text,
            page_type=fm.get("type", "concept"),
            topic_cluster=fm.get("topic_cluster", "general"),
            status=fm.get("status", "draft"),
            sources=_parse_list(fm.get("sources")),
            citations=_parse_list(fm.get("citations")),
            created=fm.get("created", ""),
            updated=fm.get("updated", ""),
        )

    def read_body(self, page_id: str) -> str:
        """Return just the body text (no frontmatter, no Related line)."""
        page = self.read_page(page_id)
        return page.body if page else ""

    def list_pages(self) -> list[str]:
        """Return page_ids of all knowledge pages (excludes index/log)."""
        if not self.wiki_dir.exists():
            return []
        return sorted(
            p.stem for p in self.wiki_dir.glob("*.md")
            if p.stem not in _RESERVED
        )

    def list_all(self) -> list[str]:
        """All wiki page_ids including reserved (index/log)."""
        if not self.wiki_dir.exists():
            return []
        return sorted(p.stem for p in self.wiki_dir.glob("*.md"))

    # ── Special files: index.md + log.md ──────────────────────────────────────

    def rebuild_index(self):
        """Rewrite wiki/index.md as a routing catalog.

        One row per page: link · type · cluster · status · one-line summary.
        This is Karpathy's "routing file" — cheap to scan, drives Query.
        """
        rows = ["# Wiki Index", "", "Compiled knowledge base. Auto-generated.", ""]
        pages = []
        for pid in self.list_pages():
            p = self.read_page(pid)
            if p:
                pages.append(p)
        # Group by cluster for readability
        pages.sort(key=lambda p: (p.topic_cluster or "general", p.page_id))
        cluster = ""
        for p in pages:
            if p.topic_cluster != cluster:
                cluster = p.topic_cluster or "general"
                rows.append(f"\n## {cluster}\n")
            rows.append(
                f"- [[wiki/{p.page_id}]] · {p.page_type} · {p.status} "
                f"— {p.summary(70)}"
            )
        rows.append("")
        path = self.wiki_dir / "index.md"
        path.write_text("\n".join(rows), encoding="utf-8")

    def append_log(self, action: str, page_id: str, detail: str = ""):
        """Append one timestamped line to wiki/log.md (append-only)."""
        path = self.wiki_dir / "log.md"
        if not path.exists():
            path.write_text("# Wiki Log\n\nAppend-only timeline.\n\n", encoding="utf-8")
        ts = _now_iso()
        line = f"- {ts} · {action} · `{page_id}`"
        if detail:
            line += f" — {detail}"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_index(self) -> str:
        path = self.wiki_dir / "index.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    # ── Internals ──────────────────────────────────────────────────────────────

    def page_path(self, page_id: str) -> Path:
        return self.wiki_dir / f"{page_id}.md"

    def _patch_frontmatter(self, page_id: str, updates: dict):
        """Regex frontmatter patcher — mirrors obsidian.py:_patch_frontmatter."""
        path = self.page_path(page_id)
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")
        for key, val in updates.items():
            pattern = rf"^{re.escape(key)}:.*$"
            replacement = f"{key}: {val}"
            if re.search(pattern, content, re.MULTILINE):
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            else:
                # Insert before the closing --- of frontmatter (first ---...---)
                content = content.replace("---\n", f"---\n{replacement}\n", 1)
        path.write_text(content, encoding="utf-8")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_list(items: list[str]) -> str:
    """Render a list as a YAML-friendly inline value for frontmatter."""
    if not items:
        return "[]"
    return "[" + ", ".join(items) + "]"


def _parse_list(val) -> list[str]:
    """Parse a frontmatter list value (YAML list or inline [a, b])."""
    if not val:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return []
    return [x.strip().strip('"').strip("'") for x in s.split(",") if x.strip()]


def _parse_frontmatter(fm_text: str) -> dict:
    """Parse a frontmatter block (between --- fences) into a dict."""
    try:
        import yaml
        return yaml.safe_load(fm_text) or {}
    except Exception:
        return {}


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving dedup."""
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
