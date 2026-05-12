"""
noesis/adapters/browser_ws.py

WebSocket endpoint at ws://localhost:8082/ingest
Receives conversation turns from the browser extension.

This is Tier 3 — capture-only, cannot inject memories into web UIs.
Memories captured here will be promoted through the normal confidence
lifecycle and become available the next time the user uses an API tool.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def create_ws_app(memory) -> FastAPI:
    app   = FastAPI(title="Noesis Browser WS", version="0.1.0")
    conns: list[WebSocket] = []

    @app.get("/health")
    async def health():
        return {"status": "ok", "connections": len(conns)}

    @app.get("/status/{user_id}")
    async def status(user_id: str = "default"):
        """Browser extension polls this to show memory count in popup."""
        s = memory.status(user_id=user_id)
        return {
            "total":       s["total"],
            "settled":     s["settled"],
            "provisional": s["provisional"],
        }

    @app.websocket("/ingest")
    async def ingest(ws: WebSocket):
        await ws.accept()
        conns.append(ws)
        logger.info(f"Browser extension connected ({len(conns)} active)")

        try:
            while True:
                raw  = await ws.receive_text()
                data = _safe_parse(raw)
                if data is None:
                    await ws.send_text('{"status":"error","message":"invalid json"}')
                    continue

                user_id     = data.get("user_id",      "default")
                source_tool = data.get("source_tool",  "web-unknown")
                session_id  = data.get("session_id",   f"ws-{int(time.time())}")

                # Build conversation turn
                messages: list[dict] = []
                if u := data.get("user_message", "").strip():
                    messages.append({"role": "user",      "content": u})
                if a := data.get("ai_message",   "").strip():
                    messages.append({"role": "assistant", "content": a})

                if messages:
                    memory.add(
                        messages,
                        user_id      = user_id,
                        source_tool  = source_tool,
                        session_id   = session_id,
                    )
                    logger.debug(
                        f"Ingested turn from {source_tool} "
                        f"(user={user_id}, {len(messages)} msg)"
                    )
                    await ws.send_text('{"status":"ok"}')
                else:
                    await ws.send_text('{"status":"skip","reason":"empty turn"}')

        except WebSocketDisconnect:
            logger.info("Browser extension disconnected")
        except Exception as e:
            logger.error(f"WS error: {e}")
        finally:
            conns.remove(ws)


def _safe_parse(raw: str) -> Optional[dict]:
    try:
        d = json.loads(raw)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None
