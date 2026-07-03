"""
noesis/exporters.py

Export Noesis memory data to external formats.

The inverse of :mod:`noesis.importers` — dumps the hot store's thought
nodes (and optionally wiki pages) to JSON, Markdown, or fine-tuning
dataset format.

Formats
-------
- **json**      — Full structured dump, round-trippable through the
                  ``--source json`` importer.
- **markdown**  — One ``.md`` file per thought, matching the cold store's
                  frontmatter template.  Human-readable, Obsidian-compatible.
- **finetune**  — OpenAI chat-format ``.jsonl``, one line per confirmed
                  thought (settled/provisional only).  Ready for
                  ``openai`` fine-tuning or DPO training.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Schema columns that make up a thought node ───────────────────────────────

_THOUGHT_FIELDS = (
    "hash_id", "text", "type", "status", "confidence",
    "user_id", "source_tool", "source_session", "topic_cluster",
    "created_at", "fact_ref", "evolved_from", "superseded_by", "extra",
)

# Statuses considered "confirmed knowledge" for fine-tuning export
_CONFIRMED_STATUSES = ("settled", "provisional")

SUPPORTED_FORMATS = ("json", "markdown", "finetune")


# ── Public entry point ───────────────────────────────────────────────────────

def export(
    fmt: str,
    memory,
    user_id: str,
    output_path: str | Path,
    include_wiki: bool = False,
    include_superseded: bool = False,
) -> dict:
    """
    Export memory data for *user_id* to *output_path* in format *fmt*.

    Parameters
    ----------
    fmt               : "json" | "markdown" | "finetune"
    memory            : a Memory instance (needs .vector_store and optionally .cold_store)
    user_id           : which user's thoughts to export
    output_path       : file path (json/finetune) or directory (markdown)
    include_wiki      : also export wiki pages (json/markdown only)
    include_superseded: include superseded/retired thoughts

    Returns
    -------
    dict with keys: format, thoughts_exported, wiki_exported, output_path
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unknown format: {fmt!r}. Use one of {SUPPORTED_FORMATS}")

    output_path = Path(output_path)

    # Gather thoughts from the hot store
    thoughts = memory.vector_store.get_all(user_id)
    if not include_superseded:
        # get_all already excludes superseded, but be explicit
        thoughts = [t for t in thoughts if t.get("status") != "superseded"]
    elif include_superseded:
        # If caller wants superseded too, we need a raw query because
        # get_all filters them out.
        raw = memory.vector_store._con.execute(
            "SELECT * FROM items WHERE user_id=?", [user_id]
        ).fetchall()
        thoughts = [memory.vector_store._to_dict(r) for r in raw]

    # Gather wiki pages if requested
    wiki_pages: List[dict] = []
    if include_wiki and memory.cold_store:
        wiki_pages = _collect_wiki(memory.cold_store.root)

    # Dispatch to formatter
    if fmt == "json":
        count = _export_json(thoughts, wiki_pages, user_id, output_path)
    elif fmt == "markdown":
        count = _export_markdown(thoughts, wiki_pages, output_path)
    elif fmt == "finetune":
        count = _export_finetune(thoughts, user_id, output_path)

    return {
        "format":           fmt,
        "thoughts_exported": count,
        "wiki_exported":    len(wiki_pages) if include_wiki else 0,
        "output_path":      str(output_path),
    }


def count_exportable(
    memory,
    user_id: str,
    include_superseded: bool = False,
    include_wiki: bool = False,
) -> dict:
    """
    Preview counts without writing (for --dry-run).

    Returns
    -------
    dict with: total, confirmed, tentative, superseded, wiki
    """
    # Raw query to get all statuses
    raw = memory.vector_store._con.execute(
        "SELECT status FROM items WHERE user_id=?", [user_id]
    ).fetchall()

    by_status: Dict[str, int] = {}
    for r in raw:
        s = r["status"] if "status" in r.keys() else r[0]
        by_status[s] = by_status.get(s, 0) + 1

    total = len(raw)
    superseded = by_status.get("superseded", 0)
    wiki_count = 0
    if include_wiki and memory.cold_store:
        wiki_count = len(_collect_wiki(memory.cold_store.root))

    return {
        "total":       total,
        "confirmed":   sum(by_status.get(s, 0) for s in _CONFIRMED_STATUSES),
        "tentative":   by_status.get("tentative", 0),
        "superseded":  superseded,
        "wiki":        wiki_count,
    }


# ── JSON ─────────────────────────────────────────────────────────────────────

def _export_json(
    thoughts: List[dict],
    wiki_pages: List[dict],
    user_id: str,
    output_path: Path,
) -> int:
    """Export as a structured JSON file, round-trippable through the importer."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    data: Dict[str, Any] = {
        "_meta": {
            "exported_at": now,
            "user_id":     user_id,
            "format":      "noesis-json-v1",
            "thought_count": 0,
            "wiki_count":    len(wiki_pages),
        },
        "conversations": [],
    }

    for t in thoughts:
        entry = {}
        for field in _THOUGHT_FIELDS:
            val = t.get(field)
            if field == "extra" and isinstance(val, str):
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
            if field == "created_at" and val is not None:
                val = _epoch_to_iso(val)
            entry[field] = val
        data["conversations"].append(entry)

    data["_meta"]["thought_count"] = len(data["conversations"])

    if wiki_pages:
        data["wiki"] = wiki_pages

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(data["conversations"])


# ── Markdown ─────────────────────────────────────────────────────────────────

def _export_markdown(
    thoughts: List[dict],
    wiki_pages: List[dict],
    output_path: Path,
) -> int:
    """Export as individual markdown files, Obsidian-compatible."""
    output_path.mkdir(parents=True, exist_ok=True)
    thoughts_dir = output_path / "thoughts"
    thoughts_dir.mkdir(exist_ok=True)

    count = 0
    for t in thoughts:
        hash_id = t.get("hash_id", f"unknown-{count}")
        md = _thought_to_markdown(t)
        (thoughts_dir / f"{hash_id}.md").write_text(md, encoding="utf-8")
        count += 1

    if wiki_pages:
        wiki_dir = output_path / "wiki"
        wiki_dir.mkdir(exist_ok=True)
        for page in wiki_pages:
            page_id = page.get("page_id", "unknown")
            md = _wiki_to_markdown(page)
            (wiki_dir / f"{page_id}.md").write_text(md, encoding="utf-8")

    return count


def _thought_to_markdown(t: dict) -> str:
    """Render a thought node as markdown with frontmatter.

    Mirrors the cold store's _THOUGHT_TEMPLATE format for consistency.
    """
    created = _epoch_to_iso(t.get("created_at")) or ""
    conf = t.get("confidence", 0.0)

    lines = [
        "---",
        f"hash: {t.get('hash_id', '')}",
        f"created: {created}",
        f"user_id: {t.get('user_id', '')}",
        f"type: {t.get('type', 'position')}",
        f"status: {t.get('status', 'tentative')}",
        f"confidence: {conf:.2f}" if isinstance(conf, (int, float)) else "confidence: 0.00",
        f"topic_cluster: {t.get('topic_cluster', '')}",
        f"source_tool: {t.get('source_tool', '')}",
        f"source_session: {t.get('source_session', '')}",
    ]

    # Optional fields
    for key in ("fact_ref", "evolved_from", "superseded_by"):
        if val := t.get(key):
            lines.append(f"{key}: {val}")

    lines.append("---")
    lines.append("")  # blank line after frontmatter
    lines.append(t.get("text", ""))
    lines.append("")
    return "\n".join(lines)


def _wiki_to_markdown(page: dict) -> str:
    """Render a wiki page dict as markdown with frontmatter."""
    lines = [
        "---",
        f"page_id: {page.get('page_id', '')}",
        f"title: {page.get('title', '')}",
        f"page_type: {page.get('page_type', 'concept')}",
        f"topic_cluster: {page.get('topic_cluster', '')}",
        f"status: {page.get('status', 'draft')}",
        f"created: {page.get('created', '')}",
        f"updated: {page.get('updated', '')}",
        "---",
        "",
    ]
    if page.get("title"):
        lines.append(f"# {page['title']}")
        lines.append("")
    lines.append(page.get("body", ""))
    return "\n".join(lines)


# ── Fine-tuning ──────────────────────────────────────────────────────────────

def _export_finetune(
    thoughts: List[dict],
    user_id: str,
    output_path: Path,
) -> int:
    """
    Export as OpenAI fine-tuning chat format (.jsonl).

    Only confirmed thoughts (settled/provisional) are included — tentative
    and superseded nodes are excluded because they're not verified knowledge.

    Each line is a JSON object:
        {"messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},
        ]}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = (
        f"You are an assistant with persistent memory about the user. "
        f"Answer consistently with what you know about them."
    )

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for t in thoughts:
            status = t.get("status", "tentative")
            if status not in _CONFIRMED_STATUSES:
                continue

            text = t.get("text", "").strip()
            if not text:
                continue

            # Split the thought into a user/assistant exchange.
            # The stored text is typically "user: ...\nassistant: ..." or
            # just a cleaned assertion. We try to parse the roles; if that
            # fails, we frame it as a user statement + assistant confirmation.
            user_msg, asst_msg = _split_to_dialogue(text, t)

            entry = {
                "messages": [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": asst_msg},
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    return count


def _split_to_dialogue(text: str, thought: dict) -> tuple[str, str]:
    """
    Split a thought's text into a (user_message, assistant_message) pair
    suitable for fine-tuning.

    If the text already has "role:" prefixes (from _to_text formatting),
    extract them directly. Otherwise, construct a natural dialogue from
    the thought type.
    """
    # Try parsing "user: ...\nassistant: ..." format
    if "user:" in text.lower() and "assistant:" in text.lower():
        user_msg = ""
        asst_msg = ""
        current_role = None
        for line in text.split("\n"):
            lower = line.lower().strip()
            if lower.startswith("user:"):
                current_role = "user"
                user_msg = line.split(":", 1)[1].strip()
            elif lower.startswith("assistant:"):
                current_role = "assistant"
                asst_msg = line.split(":", 1)[1].strip()
            elif current_role == "user":
                user_msg += "\n" + line
            elif current_role == "assistant":
                asst_msg += "\n" + line
        if user_msg and asst_msg:
            return user_msg.strip(), asst_msg.strip()
        if user_msg:
            return user_msg.strip(), f"Got it. I'll remember that: {user_msg.strip()}"

    # Single-assertion: frame based on thought type
    thought_type = thought.get("type", "position")
    topic = thought.get("topic_cluster", "")

    type_prompts = {
        "identity":   "Tell me about yourself.",
        "preference": "What are your preferences?",
        "position":   "What's your stance on this?",
        "event":      "What did you decide?",
        "question":   "What would you like to know?",
    }
    prompt = type_prompts.get(thought_type, "What should I know?")

    return prompt, text


# ── Wiki collection ──────────────────────────────────────────────────────────

def _collect_wiki(vault_root: Path) -> List[dict]:
    """Read all wiki pages from the vault and return as dicts."""
    wiki_dir = vault_root / "wiki"
    if not wiki_dir.exists():
        return []

    pages: List[dict] = []
    try:
        from ..wiki.writer import WikiWriter
        writer = WikiWriter(str(vault_root))
        for page_id in writer.list_pages():
            page = writer.read_page(page_id)
            if page is None:
                continue
            pages.append({
                "page_id":       page.page_id,
                "title":         page.title,
                "body":          page.body,
                "page_type":     page.page_type,
                "topic_cluster": page.topic_cluster,
                "status":        page.status,
                "sources":       page.sources,
                "citations":     page.citations,
                "related":       page.related,
                "created":       page.created,
                "updated":       page.updated,
            })
    except Exception as e:
        logger.warning(f"Wiki collection failed: {e}")

    return pages


# ── Helpers ──────────────────────────────────────────────────────────────────

def _epoch_to_iso(epoch: Optional[float]) -> Optional[str]:
    """Convert a Unix epoch timestamp to ISO-8601 UTC string."""
    if epoch is None:
        return None
    try:
        dt = datetime.fromtimestamp(float(epoch), tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, OSError):
        return None
