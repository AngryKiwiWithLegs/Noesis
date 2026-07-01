"""
noesis/graphs/obsidian_linker.py

After a node is written to the vault, link it to its neighbours.

Three-tier graph:
  Tier 1 → clusters/ nodes (always linked, the visible graph)
  Tier 2 → top-3 similar nodes within the same cluster (detail layer)
  Tier 3 → matching wiki/ nodes by topic_cluster (cross-layer links)

Dedup detection:
  similarity > DEDUP_THRESHOLD → flag as potential duplicate,
  do NOT auto-merge (human decides in Obsidian)

Tentative nodes never receive any [[]] links — they stay invisible
in the Obsidian graph until they earn provisional status.

Wiki cross-linking (NOESIS_SCHEMA.md rule #1):
  For a confirmed thought, find wiki/ pages sharing the same topic_cluster
  and append [[wiki/xxx]] refs to the thought's Related field. This is the
  contract that ties LLM Wiki knowledge to live thoughts.
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

        # 3. Cross-link to matching wiki/ pages (tier 3, schema rule #1)
        if cluster:
            self._link_to_wiki(hash_id, cluster)

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

    # ── Wiki tier (cross-layer, schema rule #1) ────────────────────────────────

    def _link_to_wiki(self, hash_id: str, cluster: str):
        """Find wiki/ pages in the same topic_cluster and append their
        [[wiki/xxx]] refs to the thought's Related field.

        Implements NOESIS_SCHEMA.md rule #1:
          "graph_linker finds matching wiki/ nodes by topic_cluster,
           writes [[wiki/xxx]] into thoughts/ Related field."
        """
        wiki_pages = self._wiki_pages_in_cluster(cluster)
        if not wiki_pages:
            return

        path = self.vault / "thoughts" / f"{hash_id}.md"
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")

        changed = False
        for pid in wiki_pages:
            ref = f"[[wiki/{pid}]]"
            if ref not in content:
                # Append to the Related: line
                content = re.sub(
                    r"^Related:(.*?)$",
                    lambda m, r=ref: f"Related:{m.group(1)} {r}",
                    content,
                    flags=re.MULTILINE,
                )
                changed = True
        if changed:
            path.write_text(content, encoding="utf-8")

    def _wiki_pages_in_cluster(self, cluster: str) -> list[str]:
        """Return page_ids of wiki/*.md pages whose topic_cluster matches.

        Parses each wiki page's frontmatter with pyyaml. Excludes the
        reserved index.md / log.md files.
        """
        wiki_dir = self.vault / "wiki"
        if not wiki_dir.exists():
            return []
        out = []
        for p in sorted(wiki_dir.glob("*.md")):
            stem = p.stem
            if stem in ("index", "log"):
                continue
            fm = self._read_wiki_frontmatter(p)
            if fm.get("topic_cluster", "general") == cluster:
                out.append(stem)
        return out

    @staticmethod
    def _read_wiki_frontmatter(path) -> dict:
        try:
            import yaml
            raw = path.read_text(encoding="utf-8")
            parts = raw.split("---", 2)
            if len(parts) < 3:
                return {}
            return yaml.safe_load(parts[1]) or {}
        except Exception:
            return {}

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
        Parse the Related: line of a node's md file, return refs that resolve
        to content in the hot store OR the wiki layer.

        Captures three ref kinds (the old [a-f0-9]{6,12}-only regex silently
        dropped wiki/ and _topic_ refs):
          [[a1b2c3d4]]      → hex thought hash (resolved via hot store)
          [[wiki/slug]]     → wiki page (resolved via the wiki/ directory)
          [[_topic_cluster]]→ cluster node (resolved via clusters/ dir)

        Used by HybridRetriever for 1-hop graph expansion.
        """
        path = self.vault / "thoughts" / f"{hash_id}.md"
        if not path.exists():
            return []
        content = path.read_text(encoding="utf-8")
        m = re.search(r"^Related:(.*?)$", content, re.MULTILINE)
        if not m:
            return []
        out: list[str] = []
        for ref in re.findall(r"\[\[([^\]]+)\]\]", m.group(1)):
            if ref.startswith("wiki/"):
                # Wiki page ref — resolve via filesystem
                pid = ref[len("wiki/"):]
                if (self.vault / "wiki" / f"{pid}.md").exists():
                    out.append(ref)
            elif ref.startswith("_topic_"):
                # Cluster node ref — resolve via filesystem
                if (self.vault / "clusters" / f"{ref}.md").exists():
                    out.append(ref)
            else:
                # Hex thought hash — resolve via hot store
                if re.fullmatch(r"[a-f0-9]{6,12}", ref) and self.vs.exists(ref):
                    out.append(ref)
        return out
