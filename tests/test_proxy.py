"""
tests/test_proxy.py

API proxy tests — all run without real API calls.
Uses FastAPI TestClient + httpx mock to simulate upstream responses.

Run:
    pytest tests/test_proxy.py -v
"""
import json
import time
import pytest

from noesis.memory.main import Memory
from noesis.adapters.api_proxy import (
    create_proxy_app,
    _inject_system,
    _last_user_message,
    _parse_assistant_text,
    _base_url,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path):
    m = Memory.from_config({
        "vector_store": {"config": {"db_path": str(tmp_path / "hot.db")}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
    })
    return m


@pytest.fixture
def app(mem):
    return create_proxy_app(mem)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_models_endpoint(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        assert "data" in r.json()


# ── Helper functions ──────────────────────────────────────────────────────────

class TestHelpers:

    def test_last_user_message_string(self):
        msgs = [
            {"role": "system",    "content": "You are helpful"},
            {"role": "user",      "content": "Hello there"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user",      "content": "My name is Zhang San"},
        ]
        assert _last_user_message(msgs) == "My name is Zhang San"

    def test_last_user_message_empty(self):
        assert _last_user_message([]) == ""
        assert _last_user_message([{"role": "assistant", "content": "hi"}]) == ""

    def test_inject_system_creates_new(self):
        msgs  = [{"role": "user", "content": "Hello"}]
        ctx   = "User is Zhang San."
        out   = _inject_system(msgs, ctx)
        assert out[0]["role"] == "system"
        assert "Zhang San" in out[0]["content"]
        assert out[1]["role"] == "user"

    def test_inject_system_appends_to_existing(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user",   "content": "Hello"},
        ]
        ctx = "User prefers Python."
        out = _inject_system(msgs, ctx)
        system_content = out[0]["content"]
        assert "You are helpful" in system_content
        assert "User prefers Python" in system_content

    def test_inject_does_not_duplicate_context(self):
        msgs = [{"role": "user", "content": "Hello"}]
        out1 = _inject_system(msgs, "Context A")
        out2 = _inject_system(out1, "Context B")
        # System message should exist only once
        sys_msgs = [m for m in out2 if m["role"] == "system"]
        assert len(sys_msgs) == 1

    # Routing
    def test_routing_gpt(self):
        assert "openai.com" in _base_url("gpt-4o")

    def test_routing_claude(self):
        assert "anthropic.com" in _base_url("claude-sonnet-4-6")

    def test_routing_gemini(self):
        assert "google" in _base_url("gemini-pro")

    def test_routing_ollama(self):
        assert "localhost:11434" in _base_url("llama3.2")
        assert "localhost:11434" in _base_url("phi-3")

    def test_routing_default(self):
        assert "openai.com" in _base_url("unknown-model-xyz")

    # Response parsing
    def test_parse_json_response(self):
        raw = json.dumps({
            "choices": [{"message": {"content": "Hello, I am an AI."}}]
        }).encode()
        text = _parse_assistant_text(raw)
        assert text == "Hello, I am an AI."

    def test_parse_sse_stream(self):
        chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            'data: [DONE]\n',
        ]
        raw = "".join(chunks).encode()
        text = _parse_assistant_text(raw)
        assert text == "Hello world"

    def test_parse_empty_response(self):
        assert _parse_assistant_text(b"") == ""
        assert _parse_assistant_text(b"data: [DONE]\n") == ""


# ── Memory injection in request ───────────────────────────────────────────────

class TestMemoryInjection:

    def test_settled_memory_injected(self, mem, app):
        """When a settled memory exists, it should appear in the forwarded request."""
        # Add and settle a memory
        r = mem.add("用户叫李华，是一名数据工程师", user_id="test_u1", type="identity")
        h = r["results"][0]["id"]
        mem.vector_store.update(h, {"status": "settled", "confidence": 0.9})

        # We can't call the real API, but we can test build_context directly
        ctx = mem.build_context("请做自我介绍", user_id="test_u1")
        assert "李华" in ctx, "Settled identity should appear in context"

        # Verify inject_system would put it in the messages
        msgs   = [{"role": "user", "content": "请做自我介绍"}]
        injected = _inject_system(msgs, ctx)
        system = next(m for m in injected if m["role"] == "system")
        assert "李华" in system["content"]

    def test_no_injection_when_no_settled_memories(self, mem, app):
        """If no settled memories, context should be empty string."""
        mem.add("tentative 节点", user_id="empty_user")
        ctx = mem.build_context("任意查询", user_id="empty_user")
        assert ctx == ""


# ── MCP server tools ──────────────────────────────────────────────────────────

class TestMCPServer:

    @pytest.mark.asyncio
    async def test_remember_tool(self, mem):
        from noesis.adapters.mcp import NoesisMCPServer
        server = NoesisMCPServer(mem)
        result = await server._call_tool("remember", {
            "content": "用户是一名 AI 工程师",
            "user_id": "mcp_test",
            "type":    "identity",
        })
        assert "Stored" in result

    @pytest.mark.asyncio
    async def test_recall_empty(self, mem):
        from noesis.adapters.mcp import NoesisMCPServer
        server = NoesisMCPServer(mem)
        result = await server._call_tool("recall", {
            "query":   "anything",
            "user_id": "mcp_empty_user",
        })
        assert "No relevant memories" in result

    @pytest.mark.asyncio
    async def test_recall_with_memory(self, mem):
        from noesis.adapters.mcp import NoesisMCPServer
        mem.add("用户偏好 Rust 语言", user_id="mcp_u2", type="preference")
        res = mem.search("Rust", user_id="mcp_u2")
        if res:
            mem.vector_store.update(res[0]["id"], {"status": "settled"})

        server = NoesisMCPServer(mem)
        result = await server._call_tool("recall", {
            "query":   "编程语言偏好",
            "user_id": "mcp_u2",
        })
        assert "Rust" in result

    @pytest.mark.asyncio
    async def test_status_tool(self, mem):
        from noesis.adapters.mcp import NoesisMCPServer
        server = NoesisMCPServer(mem)
        result = await server._call_tool("memory_status", {"user_id": "default"})
        assert "Total:" in result

    @pytest.mark.asyncio
    async def test_inspect_missing_node(self, mem):
        from noesis.adapters.mcp import NoesisMCPServer
        server = NoesisMCPServer(mem)
        result = await server._call_tool("inspect_memory", {"hash_id": "nonexistent"})
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_initialize_handshake(self, mem):
        from noesis.adapters.mcp import NoesisMCPServer
        server = NoesisMCPServer(mem)
        resp   = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        assert resp["result"]["serverInfo"]["name"] == "noesis"

    @pytest.mark.asyncio
    async def test_tools_list(self, mem):
        from noesis.adapters.mcp import NoesisMCPServer, TOOLS
        server = NoesisMCPServer(mem)
        resp   = await server.handle_message({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        })
        names = [t["name"] for t in resp["result"]["tools"]]
        assert "remember"       in names
        assert "recall"         in names
        assert "inspect_memory" in names
        assert "memory_status"  in names
