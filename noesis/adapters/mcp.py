"""
noesis/adapters/mcp.py

MCP server exposing five tools:
  remember(content, user_id)    — store a thought
  recall(query, user_id)        — retrieve relevant thoughts/context
  wiki_query(query)             — retrieve compiled wiki knowledge (documents)
  inspect_memory(hash_id)       — read a specific node (human-edited version)
  memory_status(user_id)        — dashboard summary

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "noesis": {
        "command": "noesis",
        "args": ["mcp"]
      }
    }
  }

Or with python directly:
  {
    "mcpServers": {
      "noesis": {
        "command": "python",
        "args": ["-m", "noesis.adapters.mcp"],
        "cwd": "/path/to/noesis"
      }
    }
  }
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys

logger = logging.getLogger(__name__)


# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "remember",
        "description": (
            "Store a thought or fact into the user's permanent memory. "
            "Use when the user shares important information about themselves, "
            "makes a decision, states a preference, or expresses a clear position."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content":  {"type": "string",
                             "description": "The thought or fact to store"},
                "user_id":  {"type": "string", "default": "default",
                             "description": "User identifier"},
                "type":     {"type": "string",
                             "enum": ["position","question","event",
                                      "preference","identity"],
                             "default": "position"},
                "topic":    {"type": "string", "default": "",
                             "description": "Short topic label (kebab-case)"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall",
        "description": (
            "Retrieve relevant memories before answering. "
            "Returns a formatted string ready for system prompt injection. "
            "Call at the start of any conversation turn where user context matters. "
            "Searches the user's evolving thoughts/positions, NOT compiled documents."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":   {"type": "string",
                            "description": "What you want to know about the user"},
                "user_id": {"type": "string", "default": "default"},
                "budget":  {"type": "integer", "default": 1200,
                            "description": "Max tokens of context to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "wiki_query",
        "description": (
            "Search compiled wiki pages (documents the user has ingested: papers, "
            "notes, manuals) for knowledge relevant to a query. Use this for factual "
            "or documented knowledge, complementing recall() which covers the user's "
            "own evolving thoughts. Returns ranked wiki excerpts with [[wiki/...]] "
            "citations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":  {"type": "string",
                           "description": "What to look up in compiled documents"},
                "top_k":  {"type": "integer", "default": 5,
                           "description": "Max pages to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "inspect_memory",
        "description": (
            "Read the full text of a specific memory node by its hash ID. "
            "Returns the current version including any human edits made in Obsidian."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "hash_id": {"type": "string",
                            "description": "12-character hex node ID"},
            },
            "required": ["hash_id"],
        },
    },
    {
        "name": "memory_status",
        "description": "Show a summary of the user's memory store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "default": "default"},
            },
        },
    },
]


# ── Server ────────────────────────────────────────────────────────────────────

class NoesisMCPServer:
    def __init__(self, memory):
        self.memory = memory

    async def handle_message(self, msg: dict) -> dict | None:
        method = msg.get("method", "")
        mid    = msg.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": mid,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "noesis", "version": "0.1.0"},
                    "capabilities": {"tools": {}},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0", "id": mid,
                "result": {"tools": TOOLS},
            }

        if method == "tools/call":
            name   = msg.get("params", {}).get("name", "")
            args   = msg.get("params", {}).get("arguments", {})
            result = await self._call_tool(name, args)
            return {
                "jsonrpc": "2.0", "id": mid,
                "result": {
                    "content": [{"type": "text", "text": result}],
                    "isError": False,
                },
            }

        if method == "notifications/initialized":
            return None  # Notification — no response

        logger.warning(f"Unknown method: {method}")
        return {
            "jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    async def _call_tool(self, name: str, args: dict) -> str:
        try:
            if name == "remember":
                result = self.memory.add(
                    args.get("content", ""),
                    user_id      = args.get("user_id", "default"),
                    type         = args.get("type", "position"),
                    topic_cluster= args.get("topic", ""),
                    source_tool  = "claude-desktop-mcp",
                )
                n = len(result.get("results", []))
                return f"Stored {n} thought(s)."

            elif name == "recall":
                ctx = self.memory.build_context(
                    args.get("query", ""),
                    user_id      = args.get("user_id", "default"),
                    budget_tokens= args.get("budget", 1200),
                )
                return ctx if ctx else "(No relevant memories found)"

            elif name == "wiki_query":
                from ..context.signals import wiki_signal
                # The vault path lives on the cold store; bail cleanly if the
                # user runs with a hot-only memory (no compiled documents).
                cs = getattr(self.memory, "cold_store", None)
                if cs is None:
                    return "(No wiki configured: no cold store / vault path.)"
                vault_path = str(cs.root)
                hits = wiki_signal(
                    args.get("query", ""),
                    vault_path,
                    embedding_model=getattr(self.memory, "embedding", None),
                    top_k=args.get("top_k", 5),
                )
                if not hits:
                    return "(No matching wiki pages found.)"
                lines = []
                for i, h in enumerate(hits, 1):
                    cite = h.get("source", "")
                    text = " ".join(h.get("text", "").split())
                    lines.append(f"{i}. {cite} — {text[:240]}")
                return "\n".join(lines)

            elif name == "inspect_memory":
                text = self.memory.get(args.get("hash_id", ""))
                return text if text else "(Node not found)"

            elif name == "memory_status":
                s = self.memory.status(args.get("user_id", "default"))
                return (
                    f"Total: {s['total']} nodes\n"
                    f"  settled:     {s['settled']}\n"
                    f"  provisional: {s['provisional']}\n"
                    f"  tentative:   {s['tentative']}\n"
                    f"  pipeline:    {s['pipeline_depth']} queued"
                )
            else:
                return f"Unknown tool: {name}"

        except Exception as e:
            logger.error(f"Tool error [{name}]: {e}", exc_info=True)
            return f"Error: {e}"


async def run_stdio(memory):
    """Run MCP server over stdio (standard MCP transport)."""
    server = NoesisMCPServer(memory)
    reader = asyncio.StreamReader()
    proto  = asyncio.StreamReaderProtocol(reader)
    loop   = asyncio.get_event_loop()

    await loop.connect_read_pipe(lambda: proto, sys.stdin.buffer)
    writer_transport, writer_protocol = await loop.connect_write_pipe(
        lambda: asyncio.BaseProtocol(), sys.stdout.buffer
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, loop)

    logger.info("Noesis MCP server running (stdio)")

    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            msg  = json.loads(line.decode().strip())
            resp = await server.handle_message(msg)
            if resp is not None:
                writer.write((json.dumps(resp) + "\n").encode())
                await writer.drain()
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"MCP loop error: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from pathlib import Path
    # Load memory from default config
    config_path = Path("~/.noesis/config.yaml").expanduser()
    if config_path.exists():
        from ..memory.main import Memory
        mem = Memory.from_config_file(str(config_path))
    else:
        from ..memory.main import Memory
        mem = Memory.from_config({
            "vector_store": {"config": {"db_path": "~/.noesis/hot.db"}},
            "embedder": {"config": {"model": "all-MiniLM-L6-v2"}},
        })
    asyncio.run(run_stdio(mem))
