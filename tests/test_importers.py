"""
tests/test_importers.py

Tests for the multi-source conversation import normalizers.

Covers:
  - ChatGPT mapping-tree parsing
  - Claude chat_messages structure + role mapping + timestamps
  - Gemini HTML (Takeout) and JSON (extension) parsing
  - Meta AI message_*.json format + role inference + timestamps
  - Auto-detection of source format
  - End-to-end import through memory.add()
"""
import json
from pathlib import Path

import pytest

from noesis.importers import (
    detect_source,
    normalize,
    _normalize_chatgpt,
    _normalize_claude,
    _normalize_gemini_html,
    _normalize_gemini_json,
    _normalize_meta,
    _normalize_generic_json,
    _normalize_text,
    _parse_iso,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def write_json(tmp_path: Path, name: str, data) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def write_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Timestamp parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseISO:
    def test_z_suffix(self):
        ts = _parse_iso("2024-01-15T10:30:00.000Z")
        assert ts is not None
        assert ts == pytest.approx(1705314600.0, abs=1)

    def test_offset_suffix(self):
        ts = _parse_iso("2024-01-15T10:30:00+00:00")
        assert ts is not None
        assert ts == pytest.approx(1705314600.0, abs=1)

    def test_naive_assumed_utc(self):
        ts = _parse_iso("2024-01-15T10:30:00")
        assert ts is not None
        assert ts == pytest.approx(1705314600.0, abs=1)

    def test_none(self):
        assert _parse_iso(None) is None
        assert _parse_iso("") is None

    def test_invalid(self):
        assert _parse_iso("not-a-date") is None

    def test_numeric_passthrough(self):
        assert _parse_iso(1705314600.0) == pytest.approx(1705314600.0)


# ═══════════════════════════════════════════════════════════════════════════════
# ChatGPT
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatGPTNormalize:
    def _make_convo(self):
        """Build a minimal ChatGPT mapping tree with 3 messages."""
        return {
            "title": "Test Convo",
            "id": "abc-123",
            "create_time": 1700000000.0,
            "mapping": {
                "root": {
                    "parent": None,
                    "children": ["n1"],
                },
                "n1": {
                    "parent": "root",
                    "children": ["n2"],
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["What is Python?"]},
                    },
                },
                "n2": {
                    "parent": "n1",
                    "children": ["n3"],
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Python is a programming language."]},
                    },
                },
                "n3": {
                    "parent": "n2",
                    "children": [],
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Thanks!"]},
                    },
                },
            },
        }

    def test_basic_parse(self):
        convos = _normalize_chatgpt([self._make_convo()])
        assert len(convos) == 1
        conv_id, msgs, ts = convos[0]
        assert conv_id == "Test Convo"
        assert ts == 1700000000.0
        assert len(msgs) == 3
        assert msgs[0] == {"role": "user", "content": "What is Python?"}
        assert msgs[1] == {"role": "assistant", "content": "Python is a programming language."}
        assert msgs[2] == {"role": "user", "content": "Thanks!"}

    def test_multi_part_content(self):
        convo = self._make_convo()
        convo["mapping"]["n1"]["message"]["content"]["parts"] = [
            "Part 1", "Part 2", None, 42
        ]
        convos = _normalize_chatgpt([convo])
        _, msgs, _ = convos[0]
        assert msgs[0]["content"] == "Part 1 Part 2 42"

    def test_empty_messages_skipped(self):
        convo = self._make_convo()
        convo["mapping"]["n3"]["message"]["content"]["parts"] = [""]
        convos = _normalize_chatgpt([convo])
        _, msgs, _ = convos[0]
        assert len(msgs) == 2  # n3 dropped

    def test_wrapped_in_conversations_key(self):
        data = {"conversations": [self._make_convo()]}
        convos = _normalize_chatgpt(data)
        assert len(convos) == 1

    def test_multiple_conversations(self):
        c1 = self._make_convo()
        c2 = self._make_convo()
        c2["title"] = "Second"
        convos = _normalize_chatgpt([c1, c2])
        assert len(convos) == 2
        assert convos[1][0] == "Second"

    def test_no_mapping_skipped(self):
        convos = _normalize_chatgpt([{"title": "Empty", "mapping": {}}])
        assert convos == []


# ═══════════════════════════════════════════════════════════════════════════════
# Claude
# ═══════════════════════════════════════════════════════════════════════════════

class TestClaudeNormalize:
    def _make_convo(self):
        return {
            "uuid": "conv-uuid-1",
            "name": "My Chat",
            "chat_messages": [
                {
                    "text": "Hello Claude",
                    "sender": {"role": "human"},
                    "created_at": "2024-01-15T10:30:00.000Z",
                },
                {
                    "text": "Hi! How can I help?",
                    "sender": {"role": "assistant"},
                    "created_at": "2024-01-15T10:30:05.000Z",
                },
            ],
        }

    def test_basic_parse(self):
        convos = _normalize_claude([self._make_convo()])
        assert len(convos) == 1
        conv_id, msgs, ts = convos[0]
        assert conv_id == "My Chat"
        assert ts is not None
        assert ts == pytest.approx(1705314600.0, abs=1)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "Hello Claude"}
        assert msgs[1] == {"role": "assistant", "content": "Hi! How can I help?"}

    def test_human_role_maps_to_user(self):
        convos = _normalize_claude([self._make_convo()])
        _, msgs, _ = convos[0]
        assert msgs[0]["role"] == "user"

    def test_content_array_fallback(self):
        """Newer Claude exports use content[] instead of text."""
        convo = {
            "uuid": "u2",
            "name": "New Format",
            "chat_messages": [
                {
                    "sender": {"role": "human"},
                    "content": [{"type": "text", "text": "From content array"}],
                    "created_at": "2024-02-01T00:00:00.000Z",
                },
                {
                    "sender": {"role": "assistant"},
                    "content": [{"type": "text", "text": "Reply"}],
                    "created_at": "2024-02-01T00:00:01.000Z",
                },
            ],
        }
        convos = _normalize_claude([convo])
        _, msgs, _ = convos[0]
        assert msgs[0]["content"] == "From content array"
        assert msgs[1]["content"] == "Reply"

    def test_text_field_preferred_over_content(self):
        """If both text and content[] exist, text takes priority."""
        convo = {
            "uuid": "u3",
            "name": "Both",
            "chat_messages": [
                {
                    "text": "Primary text",
                    "sender": {"role": "human"},
                    "content": [{"type": "text", "text": "Fallback"}],
                    "created_at": "2024-01-15T10:30:00.000Z",
                },
            ],
        }
        convos = _normalize_claude([convo])
        _, msgs, _ = convos[0]
        assert msgs[0]["content"] == "Primary text"

    def test_empty_chat_messages_skipped(self):
        convos = _normalize_claude([{"uuid": "x", "name": "Empty", "chat_messages": []}])
        assert convos == []

    def test_unknown_sender_defaults_to_user(self):
        convo = {
            "uuid": "u4",
            "name": "Unknown",
            "chat_messages": [
                {
                    "text": "test",
                    "sender": {},
                    "created_at": "2024-01-15T10:30:00.000Z",
                },
            ],
        }
        convos = _normalize_claude([convo])
        _, msgs, _ = convos[0]
        assert msgs[0]["role"] == "user"


# ═══════════════════════════════════════════════════════════════════════════════
# Gemini HTML
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeminiHTMLNormalize:
    def test_basic_parse(self):
        html = """
        <html><body>
          <div class="conversation-turn-user">
            <p>What is the weather?</p>
          </div>
          <div class="model-response-container">
            <p>It's sunny today.</p>
          </div>
        </body></html>
        """
        result = _normalize_gemini_html(html.encode("utf-8"))
        assert len(result) == 1
        conv_id, msgs, ts = result[0]
        assert conv_id == "gemini-takeout"
        assert ts is None
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "What is the weather?"}
        assert msgs[1] == {"role": "assistant", "content": "It's sunny today."}

    def test_role_hints(self):
        """Various class names should be recognised as user or model."""
        html = """
        <div class="human-turn">Hello</div>
        <div class="ai-response">Hi there</div>
        <div class="you-said">How are you?</div>
        <div class="gemini-answer">Good</div>
        """
        result = _normalize_gemini_html(html.encode("utf-8"))
        _, msgs, _ = result[0]
        assert len(msgs) == 4
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"
        assert msgs[3]["role"] == "assistant"

    def test_empty_html(self):
        result = _normalize_gemini_html(b"<html></html>")
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# Gemini JSON
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeminiJSONNormalize:
    def test_shape_a_conversation_key(self):
        data = [{
            "title": "Gemini Chat",
            "conversation": [
                {"role": "user", "content": "Hi"},
                {"role": "model", "content": "Hello!"},
            ],
        }]
        convos = _normalize_gemini_json(data)
        assert len(convos) == 1
        conv_id, msgs, _ = convos[0]
        assert conv_id == "Gemini Chat"
        assert msgs[0] == {"role": "user", "content": "Hi"}
        # "model" should map to "assistant"
        assert msgs[1] == {"role": "assistant", "content": "Hello!"}

    def test_shape_b_messages_key(self):
        data = [{
            "title": "Chat B",
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "model", "content": "Answer"},
            ],
        }]
        convos = _normalize_gemini_json(data)
        assert len(convos) == 1
        _, msgs, _ = convos[0]
        assert msgs[1]["role"] == "assistant"

    def test_with_timestamps(self):
        data = [{
            "title": "Timed",
            "conversation": [
                {"role": "user", "content": "Hi", "created_at": "2024-03-01T12:00:00.000Z"},
            ],
        }]
        convos = _normalize_gemini_json(data)
        _, _, ts = convos[0]
        assert ts is not None
        assert ts == pytest.approx(1709294400.0, abs=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Meta AI
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetaNormalize:
    def _make_data(self):
        return {
            "participants": [
                {"name": "Jane Doe"},
                {"name": "Meta AI"},
            ],
            "messages": [
                # Meta exports newest-first
                {"sender_name": "Meta AI", "content": "I can help!", "timestamp_ms": 1700000005000},
                {"sender_name": "Jane Doe", "content": "Hello there", "timestamp_ms": 1700000000000},
            ],
            "title": "Jane and Meta AI",
        }

    def test_basic_parse(self):
        convos = _normalize_meta(self._make_data())
        assert len(convos) == 1
        conv_id, msgs, ts = convos[0]
        assert conv_id == "Jane and Meta AI"
        # Reversed to chronological
        assert msgs[0] == {"role": "user", "content": "Hello there"}
        assert msgs[1] == {"role": "assistant", "content": "I can help!"}

    def test_timestamp_conversion(self):
        convos = _normalize_meta(self._make_data())
        _, _, ts = convos[0]
        assert ts == pytest.approx(1700000000.0, abs=1)

    def test_role_inference_bot_name(self):
        """sender_name containing 'meta ai' → assistant."""
        data = {
            "participants": [{"name": "Bob"}],
            "messages": [
                {"sender_name": "Bob", "content": "Hi", "timestamp_ms": 1000},
                {"sender_name": "Meta AI", "content": "Hello", "timestamp_ms": 2000},
            ],
        }
        convos = _normalize_meta(data)
        _, msgs, _ = convos[0]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_role_inference_from_participants(self):
        """Bot detected from participants list even if sender_name differs."""
        data = {
            "participants": [{"name": "Alice"}, {"name": "AI Assistant"}],
            "messages": [
                {"sender_name": "Alice", "content": "Q", "timestamp_ms": 1000},
                {"sender_name": "AI Assistant", "content": "A", "timestamp_ms": 2000},
            ],
        }
        convos = _normalize_meta(data)
        _, msgs, _ = convos[0]
        assert msgs[1]["role"] == "assistant"

    def test_empty_messages_skipped(self):
        data = {
            "participants": [{"name": "X"}],
            "messages": [
                {"sender_name": "X", "content": "", "timestamp_ms": 1000},
            ],
        }
        convos = _normalize_meta(data)
        assert convos == []

    def test_list_of_conversations(self):
        data = [self._make_data(), self._make_data()]
        convos = _normalize_meta(data)
        assert len(convos) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Generic JSON (Noesis experiment format)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenericJSON:
    def test_role_content_list(self, tmp_path):
        data = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        p = write_json(tmp_path, "gen.json", data)
        convos = _normalize_generic_json(p)
        assert len(convos) == 1
        _, msgs, _ = convos[0]
        assert len(msgs) == 2

    def test_noesis_experiment_format(self, tmp_path):
        data = [
            {"user": "What is 2+2?", "assistant": "4", "index": 1},
            {"user": "Thanks!", "assistant": "You're welcome", "index": 2},
        ]
        p = write_json(tmp_path, "exp.json", data)
        convos = _normalize_generic_json(p)
        assert len(convos) == 2
        _, msgs1, _ = convos[0]
        assert msgs1[0] == {"role": "user", "content": "What is 2+2?"}
        assert msgs1[1] == {"role": "assistant", "content": "4"}


# ═══════════════════════════════════════════════════════════════════════════════
# Plain text
# ═══════════════════════════════════════════════════════════════════════════════

class TestTextImport:
    def test_basic(self, tmp_path):
        p = write_file(tmp_path, "notes.txt", "Some important notes here.")
        convos = _normalize_text(p)
        assert len(convos) == 1
        _, msgs, _ = convos[0]
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Some important notes here."

    def test_empty(self, tmp_path):
        p = write_file(tmp_path, "empty.txt", "  \n  ")
        convos = _normalize_text(p)
        assert convos == []


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoDetect:
    def test_detect_chatgpt(self, tmp_path):
        data = [{"title": "X", "mapping": {"a": {"parent": None}}}]
        p = write_json(tmp_path, "cg.json", data)
        assert detect_source(p) == "chatgpt"

    def test_detect_claude(self, tmp_path):
        data = [{"uuid": "x", "chat_messages": []}]
        p = write_json(tmp_path, "cl.json", data)
        assert detect_source(p) == "claude"

    def test_detect_meta(self, tmp_path):
        data = {
            "participants": [{"name": "X"}],
            "messages": [{"sender_name": "X", "content": "hi"}],
        }
        p = write_json(tmp_path, "meta.json", data)
        assert detect_source(p) == "meta"

    def test_detect_gemini_html(self, tmp_path):
        p = write_file(tmp_path, "gem.html", "<html><body>chat</body></html>")
        assert detect_source(p) == "gemini"

    def test_detect_text(self, tmp_path):
        p = write_file(tmp_path, "notes.txt", "plain text")
        assert detect_source(p) == "text"

    def test_detect_generic_json(self, tmp_path):
        data = [{"role": "user", "content": "hi"}]
        p = write_json(tmp_path, "gen.json", data)
        assert detect_source(p) == "json"

    def test_detect_noesis_experiment_format(self, tmp_path):
        data = [{"user": "q", "assistant": "a"}]
        p = write_json(tmp_path, "exp.json", data)
        assert detect_source(p) == "json"


# ═══════════════════════════════════════════════════════════════════════════════
# normalize() entry point with each source
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeEntry:
    def test_normalize_chatgpt_via_path(self, tmp_path):
        data = [{
            "title": "T",
            "create_time": 1700000000.0,
            "mapping": {
                "r": {"parent": None, "children": ["a"]},
                "a": {"parent": "r", "children": [],
                      "message": {
                          "author": {"role": "user"},
                          "content": {"parts": ["Hi"]},
                      }},
            },
        }]
        p = write_json(tmp_path, "cg.json", data)
        convos = normalize("chatgpt", p)
        assert len(convos) == 1
        _, msgs, ts = convos[0]
        assert msgs[0]["content"] == "Hi"
        assert ts == 1700000000.0

    def test_normalize_auto(self, tmp_path):
        data = [{"uuid": "x", "chat_messages": [
            {"text": "Hi", "sender": {"role": "human"}, "created_at": "2024-01-15T10:30:00.000Z"}
        ]}]
        p = write_json(tmp_path, "auto.json", data)
        convos = normalize("auto", p)
        assert len(convos) == 1
        _, msgs, _ = convos[0]
        assert msgs[0]["role"] == "user"

    def test_normalize_gemini_html_via_path(self, tmp_path):
        html = '<div class="user-turn">Q</div><div class="model-response">A</div>'
        p = write_file(tmp_path, "g.html", html)
        convos = normalize("gemini", p)
        assert len(convos) == 1
        _, msgs, _ = convos[0]
        assert len(msgs) == 2

    def test_normalize_text(self, tmp_path):
        p = write_file(tmp_path, "n.txt", "hello world")
        convos = normalize("text", p)
        assert len(convos) == 1
        _, msgs, _ = convos[0]
        assert msgs[0]["content"] == "hello world"

    def test_normalize_unknown_source_raises(self, tmp_path):
        p = write_file(tmp_path, "x.txt", "data")
        with pytest.raises(ValueError, match="Unknown source"):
            normalize("nonexistent", p)


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end: normalize → memory.add()
# ═══════════════════════════════════════════════════════════════════════════════

class TestImportEndToEnd:
    def test_import_chatgpt_to_memory(self, mem_hot_only, tmp_path):
        data = [{
            "title": "Memory Test",
            "create_time": 1700000000.0,
            "mapping": {
                "r": {"parent": None, "children": ["a", "b"]},
                "a": {"parent": "r", "children": [],
                      "message": {
                          "author": {"role": "user"},
                          "content": {"parts": ["Tell me about databases"]},
                      }},
                "b": {"parent": "r", "children": [],
                      "message": {
                          "author": {"role": "assistant"},
                          "content": {"parts": ["Databases store data"]},
                      }},
            },
        }]
        p = write_json(tmp_path, "cg.json", data)
        convos = normalize("chatgpt", p)
        assert len(convos) == 1
        conv_id, msgs, ts = convos[0]
        mem_hot_only.add(
            msgs,
            source_tool="chatgpt-export",
            session_id=conv_id,
            created_at=ts,
        )
        # Verify it's in the DB
        con = mem_hot_only.vector_store._con
        rows = con.execute(
            "SELECT source_tool, source_session, created_at FROM items"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "chatgpt-export"
        assert rows[0][1] == "Memory Test"
        assert rows[0][2] == 1700000000.0

    def test_import_claude_with_historical_timestamp(self, mem_hot_only, tmp_path):
        """Verify created_at flows through to DB (not just time.time())."""
        data = [{
            "uuid": "u1",
            "name": "Time Test",
            "chat_messages": [
                {
                    "text": "test content",
                    "sender": {"role": "human"},
                    "created_at": "2023-06-15T08:00:00.000Z",
                },
            ],
        }]
        p = write_json(tmp_path, "cl.json", data)
        convos = normalize("claude", p)
        conv_id, msgs, ts = convos[0]
        mem_hot_only.add(msgs, source_tool="claude-export", created_at=ts)
        con = mem_hot_only.vector_store._con
        row = con.execute("SELECT created_at FROM items").fetchone()
        # 2023-06-15T08:00:00Z = 1686816000.0
        assert row[0] == pytest.approx(1686816000.0, abs=1)

    def test_import_meta_to_memory(self, mem_hot_only, tmp_path):
        data = {
            "participants": [{"name": "User"}, {"name": "Meta AI"}],
            "messages": [
                {"sender_name": "User", "content": "Hey", "timestamp_ms": 1700000000000},
                {"sender_name": "Meta AI", "content": "Hi", "timestamp_ms": 1700000005000},
            ],
        }
        p = write_json(tmp_path, "meta.json", data)
        convos = normalize("meta", p)
        conv_id, msgs, ts = convos[0]
        mem_hot_only.add(msgs, source_tool="meta-export", created_at=ts)
        con = mem_hot_only.vector_store._con
        row = con.execute("SELECT source_tool FROM items").fetchone()
        assert row[0] == "meta-export"
