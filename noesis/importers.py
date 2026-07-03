"""
Multi-source conversation import normalizers.

Each platform's export has a different JSON (or HTML) structure.  These
normalizers turn every supported format into a common intermediate
representation:

    [
        (conversation_id, messages, created_at | None),
        ...
    ]

where ``messages`` is a list of ``{"role": "user"|"assistant", "content": str}``
dicts — exactly what :meth:`Memory.add` expects.

Supported sources
-----------------
- **chatgpt** — ``conversations.json`` (Settings → Data Controls → Export)
- **claude**  — ``conversations.json`` from the privacy export archive
- **gemini**  — Google Takeout export (HTML by default; some JSON variants
                produced by browser extensions are also recognised)
- **meta**    — ``message_*.json`` from Facebook's Download Your Information
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Public type aliases ──────────────────────────────────────────────────────

Message = Dict[str, str]               # {"role": "user"|"assistant", "content": "..."}
Conversation = List[Message]           # a single thread
NormalisedTurn = Tuple[str, Conversation, Optional[float]]
# (conversation_id, messages, base_timestamp_epoch_seconds | None)

SUPPORTED_SOURCES = ("auto", "chatgpt", "claude", "gemini", "meta", "text", "json")


# ── Entry point ──────────────────────────────────────────────────────────────

def normalize(source: str, path: str | Path) -> List[NormalisedTurn]:
    """
    Parse *path* according to *source* and return normalised conversations.

    Parameters
    ----------
    source : one of :data:`SUPPORTED_SOURCES`
        ``"auto"`` attempts structure-based detection.
    path : path to the export file (JSON or HTML).

    Returns
    -------
    list of (conversation_id, messages, created_at) tuples
    """
    path = Path(path)

    # Validate source upfront (before reading the file)
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unknown source: {source!r}")

    if source == "text":
        return _normalize_text(path)
    if source == "json":
        return _normalize_generic_json(path)

    if source == "auto":
        source = detect_source(path)
        logger.info("Auto-detected source: %s", source)

    raw = path.read_bytes()

    if source == "gemini" and path.suffix.lower() in (".html", ".htm"):
        return _normalize_gemini_html(raw)

    data = json.loads(raw)

    if source == "chatgpt":
        return _normalize_chatgpt(data)
    if source == "claude":
        return _normalize_claude(data)
    if source == "gemini":
        return _normalize_gemini_json(data)
    if source == "meta":
        return _normalize_meta(data)

    raise ValueError(f"Unknown source: {source!r}")


# ── Auto-detection ───────────────────────────────────────────────────────────

def detect_source(path: str | Path) -> str:
    """
    Inspect *path* to guess the export format.

    Heuristics (first match wins):
        .html / .htm        → gemini  (Takeout default)
        JSON with "mapping" → chatgpt
        JSON with "chat_messages" → claude
        JSON with "participants"  → meta
        JSON with "messages" + "sender_name" → meta
        otherwise           → json (generic {role, content} list)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".html", ".htm"):
        return "gemini"
    if suffix in (".txt", ".md", ".markdown"):
        return "text"

    # Need to peek at JSON structure
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "text"

    # Walk into the common wrapper shapes
    probe = data
    if isinstance(data, dict):
        # {conversations: [...]} wrapper (Noesis export or ChatGPT list)
        probe_list = data.get("conversations")
        if isinstance(probe_list, list) and probe_list:
            probe = probe_list[0]
    elif isinstance(data, list) and data:
        probe = data[0]

    if isinstance(probe, dict):
        if "mapping" in probe:
            return "chatgpt"
        if "chat_messages" in probe:
            return "claude"
        if "participants" in probe:
            return "meta"
        if "sender_name" in probe:
            return "meta"
        # Noesis-generated experiment export: {user, assistant}
        if "user" in probe and "assistant" in probe:
            return "json"

    # Generic list of {role, content}
    return "json"


# ── ChatGPT ──────────────────────────────────────────────────────────────────

def _normalize_chatgpt(data) -> List[NormalisedTurn]:
    """
    Parse ChatGPT ``conversations.json``.

    The export is a list of conversations, each containing a ``mapping``
    dict that maps node IDs to ``{message: {author.role, content.parts[]}}``.
    """
    convos = _as_convo_list(data)
    results: List[NormalisedTurn] = []

    for convo in convos:
        conv_id = convo.get("title") or convo.get("id") or "chatgpt"
        mapping = convo.get("mapping", {})
        if not mapping:
            continue

        # Build an ordered list of messages by walking the tree from root
        # to leaf.  The mapping is a DAG; we find the root (no parent) and
        # follow children.  Edits create branches — we take the primary
        # (last child) path.
        ordered = _walk_chatgpt_mapping(mapping)

        messages: Conversation = []
        for role, text in ordered:
            if role in ("user", "assistant") and text.strip():
                messages.append({"role": role, "content": text})

        if messages:
            # ChatGPT exports have create_time per conversation
            ts = _safe_float(convo.get("create_time"))
            results.append((conv_id, messages, ts))

    return results


def _walk_chatgpt_mapping(mapping: dict) -> List[Tuple[str, str]]:
    """
    Walk the ChatGPT ``mapping`` DAG depth-first from root to leaf,
    returning [(role, text), ...] in conversation order.
    """
    # Find root: a node with no parent
    roots = [nid for nid, n in mapping.items() if n.get("parent") is None]
    if not roots:
        roots = list(mapping.keys())
    root = roots[0]

    # Build children index
    children: Dict[str, List[str]] = {}
    for nid, n in mapping.items():
        pid = n.get("parent")
        if pid:
            children.setdefault(pid, []).append(nid)

    ordered: List[Tuple[str, str]] = []
    stack = [root]
    visited = set()
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        node = mapping[nid]
        msg = node.get("message")
        if msg:
            role = msg.get("author", {}).get("role", "")
            parts = msg.get("content", {}).get("parts", [])
            text = " ".join(str(p) for p in parts if isinstance(p, (str, int, float)))
            if text.strip():
                ordered.append((role, text))

        # Push children in order; take the primary branch (last child wins
        # for edits — reversed so we traverse deepest first in stack).
        kids = children.get(nid, [])
        for kid in reversed(kids):
            stack.append(kid)

    return ordered


# ── Claude ───────────────────────────────────────────────────────────────────

def _normalize_claude(data) -> List[NormalisedTurn]:
    """
    Parse Claude privacy export ``conversations.json``.

    Structure (as of Feb 2026, per portable-ai-memory.org):
        [{
            "uuid": "...",
            "name": "Chat title",
            "chat_messages": [
                {
                    "text": "...",
                    "sender": {"role": "human" | "assistant"},
                    "created_at": "2024-01-15T10:30:00.000Z",
                    "content": [{"type": "text", "text": "..."}],   # newer format
                    "edited": false,
                    ...
                }
            ]
        }]
    """
    convos = _as_convo_list(data)
    results: List[NormalisedTurn] = []

    for convo in convos:
        conv_id = convo.get("name") or convo.get("uuid") or "claude"
        chat_messages = convo.get("chat_messages", [])
        if not chat_messages:
            continue

        messages: Conversation = []
        first_ts: Optional[float] = None

        for cm in chat_messages:
            sender_role = cm.get("sender", {}).get("role", "")
            role = "assistant" if sender_role == "assistant" else "user"

            # Primary text field
            text = cm.get("text", "")
            # Fallback to content[] array (newer export structure)
            if not text:
                content_parts = cm.get("content", [])
                if isinstance(content_parts, list):
                    text = " ".join(
                        p.get("text", "") for p in content_parts
                        if isinstance(p, dict) and p.get("type") == "text"
                    )

            if text.strip():
                messages.append({"role": role, "content": text.strip()})

            if first_ts is None:
                first_ts = _parse_iso(cm.get("created_at"))

        if messages:
            results.append((conv_id, messages, first_ts))

    return results


# ── Gemini ───────────────────────────────────────────────────────────────────

def _normalize_gemini_json(data) -> List[NormalisedTurn]:
    """
    Parse Gemini conversations exported as JSON by browser extensions.

    Since there is no official JSON schema (Takeout ships HTML), we handle
    the two most common extension shapes:

    Shape A (Gemini Exporter extension):
        [{"conversation": [{"role": "user"|"model", "content": "..."}]}]

    Shape B (gemini-chat-exporter):
        [{"messages": [{"role": "user"|"model", "content": "..."}]}]
    """
    convos = _as_convo_list(data)
    results: List[NormalisedTurn] = []

    for i, convo in enumerate(convos):
        # Extract the message list from either wrapper
        msgs_raw = (
            convo.get("conversation")
            or convo.get("messages")
            or convo.get("turns")
            or []
        )
        conv_id = convo.get("title") or convo.get("id") or f"gemini-{i}"
        messages: Conversation = []
        ts: Optional[float] = None

        for m in msgs_raw:
            role = m.get("role", "")
            # Gemini uses "model" for the assistant
            if role in ("model", "assistant"):
                role = "assistant"
            elif role == "user":
                role = "user"
            else:
                role = "user"  # default

            content = m.get("content") or m.get("text") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            if content and content.strip():
                messages.append({"role": role, "content": content.strip()})

            if ts is None:
                ts = _parse_iso(m.get("created_at") or m.get("timestamp"))

        if messages:
            results.append((conv_id, messages, ts))

    return results


class _GeminiHTMLParser(HTMLParser):
    """
    Parse Google Takeout Gemini HTML export.

    The Takeout HTML is a sequence of conversation blocks.  Each turn
    is wrapped in a tag (often ``<div>`` or ``<p>``) with a class or
    data attribute indicating the sender (e.g. "user" vs "model").

    Since Google's exact markup changes between versions, we use a
    forgiving approach: track text between structural tags, and infer
    roles from class-name hints in the surrounding container.
    """

    ROLE_HINTS_USER = ("user", "human", "you", "self")
    ROLE_HINTS_MODEL = ("model", "ai", "gemini", "assistant", "bot", "bard")

    def __init__(self) -> None:
        super().__init__()
        self.turns: List[Tuple[str, str]] = []
        self._current_role: Optional[str] = None
        self._current_text: List[str] = []
        self._in_turn = False
        # Track nesting of the current turn container
        self._turn_depth = 0

    @staticmethod
    def _classes_match_hint(classes: List[str], hints: tuple) -> bool:
        """
        Check if any CSS class token starts with one of the role hints.

        We match at token boundaries (prefix or after a hyphen), not as
        substrings, to avoid false positives like 'ai' matching 'said'.
        """
        for cls in classes:
            cls_lower = cls.lower()
            for hint in hints:
                if cls_lower == hint or cls_lower.startswith(hint + "-") or cls_lower.endswith("-" + hint):
                    return True
        return False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        classes = _get_classes(attrs)

        if not self._in_turn:
            # Detect a new turn container by role hints in class names
            if self._classes_match_hint(classes, self.ROLE_HINTS_MODEL):
                self._start_turn("assistant")
            elif self._classes_match_hint(classes, self.ROLE_HINTS_USER):
                self._start_turn("user")
        else:
            # Track depth for nested tags inside a turn
            self._turn_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._in_turn:
            if self._turn_depth > 0:
                self._turn_depth -= 1
            else:
                self._end_turn()

    def handle_data(self, data: str) -> None:
        if self._in_turn:
            self._current_text.append(data)

    def _start_turn(self, role: str) -> None:
        self._current_role = role
        self._current_text = []
        self._in_turn = True
        self._turn_depth = 0

    def _end_turn(self) -> None:
        text = "".join(self._current_text).strip()
        if text and self._current_role:
            self.turns.append((self._current_role, text))
        self._in_turn = False
        self._current_role = None
        self._current_text = []

    def flush(self) -> None:
        """Call after parsing to close any dangling turn."""
        if self._in_turn:
            self._end_turn()


def _get_classes(attrs: list) -> List[str]:
    """Extract the class attribute list from HTMLParser attrs."""
    for name, value in attrs:
        if name == "class":
            return (value or "").split()
    return []


def _normalize_gemini_html(raw: bytes) -> List[NormalisedTurn]:
    """Parse a Google Takeout Gemini HTML export into one conversation."""
    parser = _GeminiHTMLParser()
    try:
        parser.feed(raw.decode("utf-8", errors="replace"))
        parser.flush()
    except Exception as e:
        logger.warning("Gemini HTML parse error: %s", e)

    if not parser.turns:
        return []

    messages: Conversation = [
        {"role": role, "content": text}
        for role, text in parser.turns
        if text.strip()
    ]
    if not messages:
        return []

    return [("gemini-takeout", messages, None)]


# ── Meta AI ──────────────────────────────────────────────────────────────────

# Sender names that indicate an AI/bot assistant (case-insensitive substrings)
_META_BOT_HINTS = ("meta ai", "metaai", "ai assistant", "bot")

def _normalize_meta(data) -> List[NormalisedTurn]:
    """
    Parse Meta/Facebook ``message_*.json`` (Download Your Information).

    Structure:
        {
            "participants": [{"name": "Jane Doe"}, {"name": "Meta AI"}],
            "messages": [
                {
                    "sender_name": "Jane Doe",
                    "content": "...",
                    "timestamp_ms": 1700000000000,
                    ...
                }
            ],
            "title": "...",
            "thread_type": "...",
        }

    Role inference: any sender whose name matches a bot hint is
    ``assistant``; all others are ``user``.
    """
    # Meta exports are one conversation per file, but we also accept
    # a list of them.
    if isinstance(data, list):
        convos = data
    elif isinstance(data, dict) and "messages" in data:
        convos = [data]
    else:
        convos = data.get("conversations", []) if isinstance(data, dict) else []

    # Build set of participant names that look like bots
    results: List[NormalisedTurn] = []

    for convo in convos:
        conv_id = convo.get("title") or convo.get("thread_path") or "meta"
        participants = convo.get("participants", [])
        bot_names = {
            p.get("name", "").lower()
            for p in participants
            if any(h in p.get("name", "").lower() for h in _META_BOT_HINTS)
        }

        msgs_raw = convo.get("messages", [])
        # Meta exports messages newest-first, but we sort by timestamp
        # to get chronological order regardless of export direction.
        if all(m.get("timestamp_ms") for m in msgs_raw):
            msgs_raw = sorted(msgs_raw, key=lambda m: m.get("timestamp_ms", 0))
        else:
            # No timestamps — assume already chronological
            pass

        messages: Conversation = []
        first_ts: Optional[float] = None

        for m in msgs_raw:
            sender = m.get("sender_name", "")
            sender_lower = sender.lower()
            if any(h in sender_lower for h in _META_BOT_HINTS) or sender_lower in bot_names:
                role = "assistant"
            else:
                role = "user"

            content = m.get("content", "")
            if not content:
                continue
            content = content.strip()
            if not content:
                continue

            messages.append({"role": role, "content": content})

            if first_ts is None:
                ms = m.get("timestamp_ms")
                if ms:
                    first_ts = ms / 1000.0

        if messages:
            results.append((conv_id, messages, first_ts))

    return results


# ── Plain text & generic JSON ────────────────────────────────────────────────

def _normalize_text(path: Path) -> List[NormalisedTurn]:
    """Import plain text as a single-message conversation."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [("text", [{"role": "user", "content": text}], None)]


def _normalize_generic_json(path: Path) -> List[NormalisedTurn]:
    """
    Generic JSON import — accepts:

    - A list of ``{role, content}`` dicts (chat message format)
    - A list of ``{user, assistant}`` objects (Noesis experiment format)
    - A dict wrapping either of the above in a ``conversations`` key
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    # Unwrap {conversations: [...]} wrapper (e.g. Noesis experiment export)
    if isinstance(data, dict) and "conversations" in data:
        data = data["conversations"]

    if not isinstance(data, list):
        data = [data]

    if not data:
        return []

    results: List[NormalisedTurn] = []

    # Case 1: flat list of {role, content}
    if data and isinstance(data[0], dict) and "role" in data[0]:
        messages: Conversation = []
        for m in data:
            content = m.get("content", "")
            if content:
                messages.append({"role": m.get("role", "user"), "content": str(content)})
        if messages:
            results.append(("json", messages, None))
        return results

    # Case 2: list of {user, assistant} pairs (Noesis experiment format)
    # Case 3: noesis-json-v1 exported thoughts — dicts with {hash_id, text, type, ...}
    #         (no role/user/assistant keys, but a non-empty "text" field)
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        user_text = item.get("user", "")
        asst_text = item.get("assistant", "")

        if user_text or asst_text:
            # Case 2: experiment-style {user, assistant} pair
            messages: Conversation = []
            if user_text:
                messages.append({"role": "user", "content": str(user_text)})
            if asst_text:
                messages.append({"role": "assistant", "content": str(asst_text)})
            if messages:
                ts = _parse_iso(item.get("timestamp"))
                results.append((f"json-{i}", messages, ts))
            continue

        # Case 3: noesis-json-v1 thought — {hash_id, text, type, status, ...}
        # These come back from `noesis export --format json`. Re-importing
        # treats each thought's text as a single user-side utterance; the
        # hash_id is preserved so provenance survives the round-trip.
        text = item.get("text", "")
        if text and "role" not in item:
            ts = _parse_iso(item.get("created_at"))
            cid = item.get("hash_id") or f"json-{i}"
            results.append((str(cid), [{"role": "user", "content": str(text)}], ts))

    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def _as_convo_list(data) -> list:
    """Normalise the top-level JSON into a list of conversation dicts."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # ChatGPT and Noesis both wrap in {conversations: [...]}
        if "conversations" in data and isinstance(data["conversations"], list):
            return data["conversations"]
        # Claude archive is a list, but just in case it's wrapped:
        if "chat_messages" in data:
            return [data]
        # Meta is a single dict with "messages"
        if "messages" in data:
            return [data]
        return [data]
    return []


def _safe_float(val) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_iso(val) -> Optional[float]:
    """
    Parse an ISO-8601 timestamp string into epoch seconds.

    Handles formats like:
        "2024-01-15T10:30:00.000Z"
        "2024-01-15T10:30:00+00:00"
        "2024-01-15T10:30:00" (naive, assumed UTC)
    """
    if not val:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # Python's fromisoformat doesn't like the trailing 'Z' until 3.11+
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            # Try common alternative: "2024-01-15T10:30:00.000Z" without tz
            dt = datetime.fromisoformat(s.replace(" ", "T"))
        except ValueError:
            logger.debug("Could not parse timestamp %r", val)
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()
