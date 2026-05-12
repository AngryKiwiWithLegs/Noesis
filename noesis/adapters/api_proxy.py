"""
noesis/adapters/api_proxy.py

OpenAI-compatible API proxy running at localhost:8080.

How it works:
  1. Tool sends request to http://localhost:8080/v1/chat/completions
  2. Proxy reads memory context for the user
  3. Injects memories into system prompt (invisible to tool)
  4. Routes to real API based on model prefix
  5. Streams response back byte-for-byte (zero extra latency after first token)
  6. After stream ends, asynchronously extracts new memories

Routing table (model name prefix → base URL):
  gpt-      → api.openai.com/v1
  claude-   → api.anthropic.com/v1   (OpenAI-compat endpoint)
  gemini-   → generativelanguage.googleapis.com/v1beta/openai
  mistral-  → api.mistral.ai/v1
  llama / phi / qwen / deepseek → localhost:11434/v1  (Ollama)
  default   → api.openai.com/v1

User identification:
  Send header X-User-ID: <your_id>  (defaults to "default")
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

logger = logging.getLogger(__name__)

# ── Routing ───────────────────────────────────────────────────────────────────

_ROUTES: list[tuple[str, str]] = [
    ("gpt",        "https://api.openai.com/v1"),
    ("o1",         "https://api.openai.com/v1"),
    ("o3",         "https://api.openai.com/v1"),
    ("claude",     "https://api.anthropic.com/v1"),
    ("gemini",     "https://generativelanguage.googleapis.com/v1beta/openai"),
    ("mistral",    "https://api.mistral.ai/v1"),
    ("llama",      "http://localhost:11434/v1"),
    ("phi",        "http://localhost:11434/v1"),
    ("qwen",       "http://localhost:11434/v1"),
    ("deepseek",   "http://localhost:11434/v1"),
    ("mixtral",    "http://localhost:11434/v1"),
]
_DEFAULT_BASE = "https://api.openai.com/v1"


def _base_url(model: str) -> str:
    m = model.lower()
    for prefix, url in _ROUTES:
        if m.startswith(prefix):
            return url
    return _DEFAULT_BASE


# ── App factory ───────────────────────────────────────────────────────────────

def create_proxy_app(memory) -> FastAPI:
    """
    Returns a FastAPI app with the memory instance injected.
    Allows testing without starting a real server.
    """
    app = FastAPI(title="Noesis API Proxy", version="0.1.0")

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "noesis-proxy"}

    @app.get("/v1/models")
    async def models():
        """Pass-through endpoint so tools can list models."""
        return {
            "object": "list",
            "data": [{"id": "noesis-proxy", "object": "model",
                      "created": int(time.time()), "owned_by": "noesis"}],
        }

    # ── Main proxy endpoint ───────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body    = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        user_id  = request.headers.get("x-user-id", "default")
        model    = body.get("model", "gpt-4o-mini")
        messages = body.get("messages", [])
        stream   = body.get("stream", False)

        # ── Inject memory context into system prompt ──────────────────────────
        query   = _last_user_message(messages)
        context = ""
        try:
            context = memory.build_context(query, user_id=user_id)
        except Exception as e:
            logger.warning(f"Memory context failed: {e}")

        if context:
            body["messages"] = _inject_system(messages, context)
            logger.debug(
                f"Injected {len(context)} chars of context for user={user_id}"
            )

        # ── Forward to real API ───────────────────────────────────────────────
        target   = _base_url(model)
        fwd_url  = f"{target}/chat/completions"
        fwd_hdrs = _forward_headers(request)

        if stream:
            return StreamingResponse(
                _stream(fwd_url, body, fwd_hdrs, messages, user_id, memory),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no"},
            )
        else:
            return await _non_stream(fwd_url, body, fwd_hdrs, messages, user_id, memory)

    return app


# ── Streaming ─────────────────────────────────────────────────────────────────

async def _stream(
    url:      str,
    body:     dict,
    headers:  dict,
    messages: list,
    user_id:  str,
    memory,
) -> AsyncIterator[bytes]:
    chunks: list[bytes] = []

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream("POST", url, json=body,
                                     headers=headers) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    yield error_body
                    return

                async for chunk in resp.aiter_bytes():
                    yield chunk
                    chunks.append(chunk)
    except httpx.ConnectError as e:
        err = json.dumps({"error": {"message": f"Cannot reach {url}: {e}",
                                    "type": "connection_error"}})
        yield f"data: {err}\n\ndata: [DONE]\n\n".encode()
        return
    except Exception as e:
        logger.error(f"Stream error: {e}")
        return

    # After stream ends — extract new memories async
    full = b"".join(chunks)
    asyncio.create_task(_extract_and_store(messages, full, user_id, memory))


async def _non_stream(
    url:      str,
    body:     dict,
    headers:  dict,
    messages: list,
    user_id:  str,
    memory,
) -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(url, json=body, headers=headers)
    except httpx.ConnectError as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "connection_error"}},
            status_code=503,
        )

    asyncio.create_task(
        _extract_and_store(messages, resp.content, user_id, memory)
    )
    return JSONResponse(resp.json(), status_code=resp.status_code)


# ── Memory extraction after response ─────────────────────────────────────────

async def _extract_and_store(
    original_messages: list,
    response_bytes:    bytes,
    user_id:           str,
    memory,
):
    """Parse the response and add the full conversation to memory."""
    try:
        assistant_text = _parse_assistant_text(response_bytes)
        if not assistant_text:
            return

        full_turn = original_messages + [
            {"role": "assistant", "content": assistant_text}
        ]

        # Only store user messages + assistant response, not injected system context
        user_turns = [
            m for m in full_turn
            if m.get("role") in ("user", "assistant")
        ]
        if user_turns:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: memory.add(
                    user_turns, user_id=user_id,
                    source_tool="api-proxy",
                    session_id=f"proxy-{int(time.time())}",
                )
            )
    except Exception as e:
        logger.debug(f"Memory extraction failed (non-fatal): {e}")


def _parse_assistant_text(raw: bytes) -> str:
    """Extract assistant text from either SSE stream or JSON response."""
    text = raw.decode("utf-8", errors="replace")

    # Non-streaming: {"choices":[{"message":{"content":"..."}}]}
    if text.strip().startswith("{"):
        try:
            data = json.loads(text)
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            pass

    # Streaming SSE: data: {"choices":[{"delta":{"content":"chunk"}}]}
    parts = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            chunk = json.loads(line)
            delta = chunk["choices"][0].get("delta", {})
            if c := delta.get("content"):
                parts.append(c)
        except Exception:
            continue
    return "".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _last_user_message(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            return str(content)
    return ""


def _inject_system(messages: list, context: str) -> list:
    """
    Prepend or append memory context to the system message.
    Creates a system message if none exists.
    """
    result = list(messages)
    for i, m in enumerate(result):
        if m.get("role") == "system":
            existing = m.get("content", "")
            result[i] = {
                **m,
                "content": f"{existing}\n\n{context}".strip(),
            }
            return result
    # No system message — create one at position 0
    result.insert(0, {"role": "system", "content": context})
    return result


def _forward_headers(request: Request) -> dict:
    """
    Pass through Authorization and API-key headers.
    Strip Noesis-specific headers and hop-by-hop headers.
    """
    skip = {
        "host", "content-length", "transfer-encoding",
        "connection", "x-user-id",
    }
    hdrs = {
        k: v for k, v in request.headers.items()
        if k.lower() not in skip
    }
    hdrs["content-type"] = "application/json"
    return hdrs
