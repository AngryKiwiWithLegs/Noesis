"""
noesis/graphs/cluster.py

Manages topic_cluster aggregation nodes in the Obsidian vault.

Each cluster node (clusters/_topic_<id>.md) is the Tier-1 graph entry point:
  - All thoughts in the topic link to it
  - It links back to its member thoughts
  - Cross-tool sources are tracked
  - Open questions are listed

The cluster file is the "landing page" for everything you've thought about a topic.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ClusterManager:

    def __init__(self, vault_path: str, vector_store):
        self.vault  = Path(vault_path).expanduser().resolve()
        self.vs     = vector_store
        self._dir   = self.vault / "clusters"
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def get_or_create(self, cluster_id: str, metadata: dict = None) -> Path:
        """Return path to cluster file, creating it if absent."""
        path = self._path(cluster_id)
        if not path.exists():
            self._create(cluster_id, metadata or {})
        return path

    def add_member(self, cluster_id: str, hash_id: str, source_tool: str = ""):
        """Add a thought node as a member of this cluster."""
        path = self.get_or_create(cluster_id)
        content = path.read_text(encoding="utf-8")

        ref = f"[[{hash_id}]]"
        if ref in content:
            return   # already linked

        # Add to the Thoughts section
        if "## Thoughts\n" in content:
            content = content.replace(
                "## Thoughts\n",
                f"## Thoughts\n{ref}\n"
            )
        else:
            content += f"\n{ref}\n"

        # Track source tool
        if source_tool:
            content = self._add_source_tool(content, source_tool)

        # Update last_updated
        content = self._update_field(content, "last_updated", _now_iso())

        path.write_text(content, encoding="utf-8")
        logger.debug(f"Cluster [{cluster_id}]: added member {hash_id[:8]}")

    def add_open_question(self, cluster_id: str, question_hash: str, question_text: str):
        """Register an open question in the cluster's question list."""
        path = self.get_or_create(cluster_id)
        content = path.read_text(encoding="utf-8")

        ref = f"- [[{question_hash}]] {question_text[:80]}"
        if question_hash in content:
            return

        if "## Open Questions\n" in content:
            content = content.replace(
                "## Open Questions\n",
                f"## Open Questions\n{ref}\n"
            )
        else:
            content += f"\n## Open Questions\n{ref}\n"

        path.write_text(content, encoding="utf-8")

    def close_question(self, cluster_id: str, question_hash: str, answer_hash: str):
        """Mark a question as answered."""
        path = self._path(cluster_id)
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")
        # Move from Open Questions to Resolved
        ref_old = f"[[{question_hash}]]"
        if ref_old not in content:
            return
        content = content.replace(
            ref_old, f"~~{ref_old}~~ → [[{answer_hash}]]"
        )
        path.write_text(content, encoding="utf-8")

    def list_clusters(self) -> list[str]:
        """Return list of existing cluster IDs."""
        return [
            p.stem.replace("_topic_", "", 1)
            for p in self._dir.glob("_topic_*.md")
        ]

    def get_members(self, cluster_id: str) -> list[str]:
        """Return hash IDs of all member nodes."""
        path = self._path(cluster_id)
        if not path.exists():
            return []
        content = path.read_text(encoding="utf-8")
        return re.findall(r"\[\[([a-f0-9]{6,12})\]\]", content)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _path(self, cluster_id: str) -> Path:
        return self._dir / f"_topic_{cluster_id}.md"

    def _create(self, cluster_id: str, metadata: dict):
        name   = metadata.get("name", cluster_id.replace("-", " ").title())
        now    = _now_iso()
        content = (
            f"---\n"
            f"hash: _topic_{cluster_id}\n"
            f"type: topic_cluster\n"
            f"name: {name}\n"
            f"created: {now}\n"
            f"last_updated: {now}\n"
            f"member_tools: []\n"
            f"---\n"
            f"# {name}\n\n"
            f"## Thoughts\n"
            f"\n"
            f"## Open Questions\n"
            f"\n"
        )
        self._path(cluster_id).write_text(content, encoding="utf-8")
        logger.info(f"Created cluster: {cluster_id}")

    def _add_source_tool(self, content: str, tool: str) -> str:
        m = re.search(r"^member_tools:\s*\[(.*?)\]", content, re.MULTILINE)
        if not m:
            return content
        current_tools = [t.strip().strip('"\'') for t in m.group(1).split(",") if t.strip()]
        if tool not in current_tools:
            current_tools.append(tool)
        new_val = ", ".join(f'"{t}"' for t in current_tools)
        return re.sub(
            r"^member_tools:.*$",
            f"member_tools: [{new_val}]",
            content,
            flags=re.MULTILINE,
        )

    def _update_field(self, content: str, field: str, value: str) -> str:
        return re.sub(
            rf"^{re.escape(field)}:.*$",
            f"{field}: {value}",
            content,
            flags=re.MULTILINE,
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
