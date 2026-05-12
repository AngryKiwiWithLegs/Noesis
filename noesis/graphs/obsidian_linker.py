"""
noesis/graphs/obsidian_linker.py

After a node is written to the vault, link it to its neighbours.

Two-tier graph:
  Tier 1 → clusters/ nodes (always linked, the visible graph)
  Tier 2 → top-3 similar nodes within the same cluster (detail layer)

Dedup detection:
  similarity > DEDUP_THRESHOLD → flag as potential duplicate,
  do NOT auto-merge (human decides in Obsidian)

Tentative nodes never receive any [[]] links — they stay invisible
in the Obsidian graph until they earn provisional status.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEDUP_THRESHOLD   = 0.92   # above this → mark dedup_flag
RELATED_THRESHOLD = 0.65   # below this → not related enough to link


class ObsidianLinker:

    def __init__(
        self,
        vault_path:       str,
        vector_store,
        embedding_model,
        dedup_threshold:  float = DEDUP_THRESHOLD,
        related_threshold: float = RELATED_THRESHOLD,
    ):
        self.vault       = Path(vault_path).expanduser().resolve()
        self.vs          = vector_store
        self.emb         = embedding_model
        self.dedup_th    = dedup_threshold
        self.related_th  = related_threshold

    def link(self, hash_id: str, top_k: int = 3):
        """
        Main entry point. Called by the pipeline after a node is confirmed.
        Skips tentative nodes.
        """
        node = self.vs.get(hash_id)
        if not node:
            return
        if node.get("status") == "tentative":
            return   # tentative nodes are invisible in graph

        vec = self.vs.get_vector(hash_id)
        if vec is None:
            logger.debug(f"No vector for {hash_id[:8]}, skipping link")
            return

        # Search neighbours (exclude self, exclude superseded)
        neighbours = self.vs.search(
            vec, top_k=top_k + 2,
            filter={"user_id": node.get("user_id", "")},
        )
        neighbours = [
            n for n in neighbours
            if n.get("id") != hash_id and n.get("status") != "superseded"
        ]

        related:    list[str] = []
        dedup_flags: list[str] = []

        for n in neighbours:
            score = n.get("score", 0)
            if score >= self.dedup_th:
                dedup_flags.append(n["id"])
            elif score >= self.related_th:
                related.append(n["id"][:8])   # use short hash for readability

        related = related[:top_k]

        # 1. Update Related + dedup_flag in the node's md file
        self._update_md(hash_id, related, dedup_flags)

        # 2. Link to cluster node (tier 1)
        cluster = node.get("topic_cluster", "")
        if cluster:
            self._link_to_cluster(hash_id, cluster, node)

    # ── Cluster tier ──────────────────────────────────────────────────────────

    def _link_to_cluster(self, hash_id: str, cluster: str, node: dict):
        cluster_path = self.vault / "clusters" / f"_topic_{cluster}.md"

        if not cluster_path.exists():
            # Create cluster node
            cluster_path.parent.mkdir(parents=True, exist_ok=True)
            cluster_path.write_text(
                f"---\nhash: _topic_{cluster}\ntype: topic_cluster\n"
                f"name: {cluster}\n---\n# {cluster}\n\n## Thoughts\n",
                encoding="utf-8",
            )

        # Append the hash to the cluster's Thoughts section (idempotent)
        content = cluster_path.read_text(encoding="utf-8")
        ref = f"[[{hash_id}]]"
        if ref not in content:
            content += f"{ref}\n"
            cluster_path.write_text(content, encoding="utf-8")

        # Add [[cluster]] back-link to the thought node
        self._add_cluster_link(hash_id, cluster)

    def _add_cluster_link(self, hash_id: str, cluster: str):
        path = self.vault / "thoughts" / f"{hash_id}.md"
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")
        cluster_ref = f"[[_topic_{cluster}]]"
        if cluster_ref not in content:
            content = re.sub(
                r"^Related:(.*?)$",
                lambda m: f"Related:{m.group(1)} {cluster_ref}",
                content,
                flags=re.MULTILINE,
            )
            path.write_text(content, encoding="utf-8")

    # ── Node md update ────────────────────────────────────────────────────────

    def _update_md(
        self,
        hash_id:     str,
        related:     list[str],
        dedup_flags: list[str],
    ):
        path = self.vault / "thoughts" / f"{hash_id}.md"
        if not path.exists():
            return

        content = path.read_text(encoding="utf-8")

        # Build Related line
        rel_str = " ".join(f"[[{r}]]" for r in related) if related else ""
        content = re.sub(
            r"^Related:.*$",
            f"Related: {rel_str}",
            content,
            flags=re.MULTILINE,
        )

        # Build dedup_flag line in frontmatter
        content = re.sub(r"^dedup_flag:.*\n", "", content, flags=re.MULTILINE)
        if dedup_flags:
            flag_val = ",".join(dedup_flags)
            # Insert after first --- block
            content = content.replace(
                "---\n", f"---\ndedup_flag: {flag_val}\n", 1
            )

        path.write_text(content, encoding="utf-8")

    # ── Related nodes for graph expansion (read) ──────────────────────────────

    def get_related_hashes(self, hash_id: str) -> list[str]:
        """
        Parse the Related: line of a node's md file,
        return list of hash IDs that are in the hot store.
        Used by HybridRetriever for 1-hop graph expansion.
        """
        path = self.vault / "thoughts" / f"{hash_id}.md"
        if not path.exists():
            return []
        content = path.read_text(encoding="utf-8")
        m = re.search(r"^Related:(.*?)$", content, re.MULTILINE)
        if not m:
            return []
        refs = re.findall(r"\[\[([a-f0-9]{6,12})\]\]", m.group(1))
        return [r for r in refs if self.vs.exists(r)]
