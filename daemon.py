"""
daemon.py

Noesis local daemon.
Starts the API proxy on :8080 and optionally the WebSocket endpoint on :8082.
MCP server runs separately via stdio (launched by the MCP client).

Usage:
    noesis start          # via CLI
    python daemon.py      # direct
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import uvicorn

logger = logging.getLogger(__name__)


class NoesDaemon:
    def __init__(
        self,
        memory,
        proxy_host:   str = "127.0.0.1",
        proxy_port:   int = 8080,
        ws_host:      str = "127.0.0.1",
        ws_port:      int = 8082,
        enable_ws:    bool = False,
        log_level:    str = "info",
    ):
        self.memory     = memory
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.ws_host    = ws_host
        self.ws_port    = ws_port
        self.enable_ws  = enable_ws
        self.log_level  = log_level
        self._tasks:    list[asyncio.Task] = []
        self._servers:  list[uvicorn.Server] = []

    async def start(self):
        from noesis.adapters.api_proxy import create_proxy_app

        # ── API proxy ─────────────────────────────────────────────────────────
        proxy_app    = create_proxy_app(self.memory)
        proxy_cfg    = uvicorn.Config(
            app       = proxy_app,
            host      = self.proxy_host,
            port      = self.proxy_port,
            log_level = self.log_level,
            access_log= False,
        )
        proxy_server = uvicorn.Server(proxy_cfg)
        self._servers.append(proxy_server)

        tasks = [asyncio.create_task(proxy_server.serve())]

        # ── WebSocket (optional, for browser extension) ───────────────────────
        if self.enable_ws:
            try:
                from noesis.adapters.browser_ws import create_ws_app
                ws_app    = create_ws_app(self.memory)
                ws_cfg    = uvicorn.Config(
                    app       = ws_app,
                    host      = self.ws_host,
                    port      = self.ws_port,
                    log_level = self.log_level,
                    access_log= False,
                )
                ws_server = uvicorn.Server(ws_cfg)
                self._servers.append(ws_server)
                tasks.append(asyncio.create_task(ws_server.serve()))
            except ImportError:
                logger.info("browser_ws not available, skipping")

        self._tasks = tasks

        print(f"\n  Noesis daemon running")
        print(f"  API proxy  → http://{self.proxy_host}:{self.proxy_port}/v1")
        if self.enable_ws:
            print(f"  Browser WS → ws://{self.ws_host}:{self.ws_port}")
        print(f"\n  Set your tool's base URL to:")
        print(f"  http://{self.proxy_host}:{self.proxy_port}/v1\n")

        # Register shutdown signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown)

        await asyncio.gather(*tasks, return_exceptions=True)

    def _shutdown(self):
        logger.info("Shutdown requested")
        for s in self._servers:
            s.should_exit = True
        for t in self._tasks:
            t.cancel()


# ── Config loader ─────────────────────────────────────────────────────────────

def load_memory(config_path: Optional[str] = None):
    """Load Memory from config file, falling back to defaults."""
    from noesis.memory.main import Memory

    if config_path is None:
        config_path = str(Path("~/.noesis/config.yaml").expanduser())

    p = Path(config_path).expanduser()
    if p.exists():
        logger.info(f"Loading config from {p}")
        return Memory.from_config_file(str(p))

    logger.info("No config file found, using defaults")
    return Memory.from_config({
        "vector_store": {"config": {"db_path": "~/.noesis/hot.db"}},
        "embedder":     {"config": {"model": "all-MiniLM-L6-v2"}},
        "cold_store":   {"config": {"vault_path": "~/NoesisVault"}},
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(name)-20s  %(levelname)s  %(message)s",
    )
    config = sys.argv[1] if len(sys.argv) > 1 else None
    memory = load_memory(config)
    daemon = NoesDaemon(memory)
    asyncio.run(daemon.start())
