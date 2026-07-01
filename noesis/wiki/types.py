"""
noesis/wiki/types.py

Data model for the LLM Wiki module. Mirrors the thoughts/types.py pattern:
  - a stored node (WikiPage)  ~ ThoughtNode
  - extraction output          ~ ThoughtCandidate
  - a raw source descriptor    ~ (no thoughts equivalent; wiki-specific)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional


# ── Type aliases ──────────────────────────────────────────────────────────────

# Karpathy's wiki uses three page archetypes:
WikiPageType = Literal[
    "concept",   # a definition / explanation of something ("sqlite-vec")
    "entity",    # a named thing with attributes ("PostgreSQL", "Anthropic")
    "summary",   # a synthesis of a document (paper digest, meeting notes)
]

# Wiki pages are stable knowledge, so the status set differs from thoughts.
# "draft" = freshly ingested, not yet linted; "published" = verified current.
WikiStatus = Literal[
    "draft",       # freshly compiled — awaiting lint
    "published",   # verified current — eligible for context injection
    "stale",       # source changed since last compile — needs re-ingest
    "answered",    # a question page that has been resolved
]


def _slug_hash(page_id: str) -> str:
    """Deterministic 12-char hash for a wiki page, derived from its slug.

    Honors the schema's required `hash` frontmatter field while keeping
    wiki page filenames human-readable (descriptive slugs, not hashes).
    """
    return hashlib.sha256(page_id.encode()).hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Stored node ───────────────────────────────────────────────────────────────

@dataclass
class WikiPage:
    """A wiki/*.md page. The stored artifact.

    `page_id` is a descriptive slug (e.g. "sqlite-vec"); the filename is
    wiki/{page_id}.md. This matches Karpathy's convention of named concept
    pages and the human-readable [[wiki/sqlite-vec]] link format.
    """
    page_id:        str
    title:          str
    body:           str                          = ""
    page_type:      WikiPageType                 = "concept"
    topic_cluster:  str                          = ""
    status:         WikiStatus                   = "draft"
    sources:        list[str]                    = field(default_factory=list)
    citations:      list[str]                    = field(default_factory=list)
    related:        list[str]                    = field(default_factory=list)
    created:        str                          = field(default_factory=_now_iso)
    updated:        str                          = field(default_factory=_now_iso)

    @property
    def hash(self) -> str:
        """Frontmatter `hash` field — deterministic, derived from page_id."""
        return _slug_hash(self.page_id)

    def summary(self, n: int = 80) -> str:
        """One-line routing summary, for index.md and the query signal."""
        # Prefer the first non-heading paragraph as the summary.
        first_line = ""
        for line in self.body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first_line = stripped
                break
        flat = " ".join(first_line.split()) if first_line else " ".join(self.body.split())
        return flat[:n] + ("…" if len(flat) > n else "")


# ── Extraction output (before write) ─────────────────────────────────────────

@dataclass
class WikiCandidate:
    """What a WikiExtractor produces from one chunk of a source document.

    Parallels ThoughtCandidate: structured knowledge, pre-storage.
    """
    page_id:        str
    title:          str
    page_type:      WikiPageType                 = "concept"
    topic_cluster:  str                          = ""
    body:           str                          = ""
    sources:        list[str]                    = field(default_factory=list)
    citations:      list[str]                    = field(default_factory=list)


# ── Raw source descriptor ────────────────────────────────────────────────────

@dataclass
class DocSource:
    """A document being ingested into the wiki.

    `content` is the extracted plain text (markdown body, PDF text, etc.).
    `fmt` drives chunking strategy. `meta` carries title/author/url/etc.
    """
    path:    str
    fmt:     Literal["markdown", "text", "pdf"] = "text"
    content: str                                  = ""
    title:   str                                  = ""
    meta:    dict                                 = field(default_factory=dict)
