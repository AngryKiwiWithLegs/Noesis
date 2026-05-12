"""
noesis/cold_stores/obsidian.py

Writes thought nodes as Markdown files into an Obsidian vault.
Each file is human-readable, human-editable, and carries
full YAML frontmatter so Noesis can sync edits back to the hot store.

Directory layout inside the vault:
    thoughts/   ← Noesis writes here
    clusters/   ← Shared with LLM Wiki (topic aggregation nodes)
    wiki/       ← LLM Wiki writes here (Noesis reads for graph links)
    NOESIS_SCHEMA.md ← Integration protocol
"""
from __future__ import annotations

import re
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .base import ColdStoreBase

logger = logging.getLogger(__name__)

# ── Templates ─────────────────────────────────────────────────────────────────

_THOUGHT_TEMPLATE = """\
---
hash: {hash_id}
created: {created}
user_id: {user_id}
type: {type}
status: {status}
confidence: {confidence:.2f}
topic_cluster: {topic_cluster}
source_tool: {source_tool}
source_session: {source_session}
{optional_fields}\
---
{text}

Related:
"""

_SCHEMA = """\
# Noesis Vault Schema v1.0

## Directory ownership
thoughts/   Noesis (auto-generated from AI conversations)
wiki/       LLM Wiki (compiled from documents you feed it)
clusters/   Shared — either system may write

## Required frontmatter fields (all nodes)
hash · created · type · status · topic_cluster

## Cross-layer link rules
1. Noesis graph_linker: finds matching wiki/ nodes by topic_cluster,
   writes [[wiki/xxx]] into thoughts/ Related field.
2. LLM Wiki lint pass: scans thoughts/questions/ for status:open nodes,
   adds them to the knowledge-ingestion priority queue.
   When a wiki node answers a question, it sets:
     status: answered
     answered_by: [[wiki/xxx]]

## Conflict resolution
- Same hash written by both systems: last-write-wins + conflict_flag: true
- Same topic_cluster node: merge Related field, never overwrite body

## Graph tier convention
Tier 1 (clusters/): one node per topic, <100 total — daily view
Tier 2 (thoughts/ + wiki/): detail, accessed by drilling into a cluster
tentative nodes: stored in thoughts/ but NO [[]] links — invisible in graph
"""


class ObsidianStore(ColdStoreBase):

    def __init__(self, vault_path: str):
        self.root = Path(vault_path).expanduser().resolve()
        self._ensure_dirs()
        self._write_schema()

    # ── Public API ────────────────────────────────────────────────────────────

    def write(self, hash_id: str, metadata: dict):
        path = self._thought_path(hash_id)
        optional = self._optional_frontmatter(metadata)
        content  = _THOUGHT_TEMPLATE.format(
            hash_id       = hash_id,
            created       = _now_iso(),
            user_id       = metadata.get("user_id", ""),
            type          = metadata.get("type", "position"),
            status        = metadata.get("status", "tentative"),
            confidence    = float(metadata.get("confidence", 0.0)),
            topic_cluster = metadata.get("topic_cluster", ""),
            source_tool   = metadata.get("source_tool", ""),
            source_session= metadata.get("source_session", ""),
            optional_fields = optional,
            text          = metadata.get("text", ""),
        )
        path.write_text(content, encoding="utf-8")

    def read(self, hash_id: str) -> str:
        """Return the body text (after frontmatter, before Related line)."""
        path = self._thought_path(hash_id)
        if not path.exists():
            raise FileNotFoundError(f"Node not found: {hash_id}")
        raw   = path.read_text(encoding="utf-8")
        parts = raw.split("---", 2)
        body  = parts[2] if len(parts) >= 3 else raw
        # Strip the trailing Related: line
        lines = [l for l in body.splitlines()
                 if not l.startswith("Related:")]
        return "\n".join(lines).strip()

    def scan_modified(self, since: float) -> list[str]:
        """Return hash IDs of thought files modified after `since`."""
        td = self.root / "thoughts"
        if not td.exists():
            return []
        return [
            p.stem for p in td.glob("*.md")
            if p.stat().st_mtime > since
        ]

    def mark_superseded(self, old_hash: str, new_hash: str):
        self._patch_frontmatter(old_hash, {
            "status":        "superseded",
            "superseded_by": new_hash,
            "superseded_at": _now_iso(),
        })
        self._patch_frontmatter(new_hash, {"supersedes": old_hash})

    def update_status(self, hash_id: str, status: str, confidence: float):
        self._patch_frontmatter(hash_id, {
            "status":     status,
            "confidence": f"{confidence:.2f}",
        })

    def update_related(
        self,
        hash_id:     str,
        related:     list[str],
        dedup_flags: list[str] | None = None,
    ):
        path = self._thought_path(hash_id)
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")

        related_str = " ".join(f"[[{r}]]" for r in related) if related else ""
        if dedup_flags:
            related_str += "\nDedup: " + " ".join(f"[[{d}]]" for d in dedup_flags)

        content = re.sub(
            r"^Related:.*$",
            f"Related: {related_str}",
            content,
            flags=re.MULTILINE | re.DOTALL,
        )
        path.write_text(content, encoding="utf-8")

    def read_frontmatter(self, hash_id: str) -> dict:
        try:
            import yaml
        except ImportError:
            return {}
        path = self._thought_path(hash_id)
        if not path.exists():
            return {}
        raw   = path.read_text(encoding="utf-8")
        parts = raw.split("---", 2)
        if len(parts) < 2:
            return {}
        try:
            return yaml.safe_load(parts[1]) or {}
        except Exception:
            return {}

    # ── Cluster nodes (shared tier) ───────────────────────────────────────────

    def write_cluster(self, cluster_id: str, metadata: dict):
        path = self.root / "clusters" / f"_topic_{cluster_id}.md"
        body = metadata.get("body", f"# {metadata.get('name', cluster_id)}\n")
        fm   = (
            f"---\n"
            f"hash: _topic_{cluster_id}\n"
            f"type: topic_cluster\n"
            f"name: {metadata.get('name', cluster_id)}\n"
            f"created: {_now_iso()}\n"
            f"member_tools: {metadata.get('member_tools', [])}\n"
            f"---\n"
        )
        path.write_text(fm + body, encoding="utf-8")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _thought_path(self, hash_id: str) -> Path:
        return self.root / "thoughts" / f"{hash_id}.md"

    def _optional_frontmatter(self, metadata: dict) -> str:
        lines = []
        for key in ("fact_ref", "evolved_from", "superseded_by", "supersedes"):
            if val := metadata.get(key):
                lines.append(f"{key}: {val}")
        return ("\n".join(lines) + "\n") if lines else ""

    def _patch_frontmatter(self, hash_id: str, updates: dict):
        path = self._thought_path(hash_id)
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")
        for key, val in updates.items():
            pattern = rf"^{re.escape(key)}:.*$"
            replacement = f"{key}: {val}"
            if re.search(pattern, content, re.MULTILINE):
                content = re.sub(pattern, replacement,
                                 content, flags=re.MULTILINE)
            else:
                # Insert before the closing --- of frontmatter
                content = content.replace("---\n", f"---\n{replacement}\n", 1)
        path.write_text(content, encoding="utf-8")

    def _ensure_dirs(self):
        for sub in ("thoughts", "clusters", "wiki"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def _write_schema(self):
        schema_path = self.root / "NOESIS_SCHEMA.md"
        if not schema_path.exists():
            schema_path.write_text(_SCHEMA, encoding="utf-8")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
