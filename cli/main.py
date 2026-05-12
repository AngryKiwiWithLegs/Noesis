"""
cli/main.py

Noesis command-line interface.

Commands:
  noesis start   [--config PATH] [--port N]   Start the daemon
  noesis status  [--user USER]                Memory stats
  noesis inspect <hash_id>                    Show a memory node
  noesis sync    [--user USER]                Force sync vault → hot store
  noesis import  --source chatgpt FILE        Batch import conversations
  noesis eval    [--user USER]                Run injection accuracy test
  noesis mcp                                  Start MCP server (stdio)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click


# ── Config loading ────────────────────────────────────────────────────────────

def _get_memory(config_path: Optional[str] = None):
    from daemon import load_memory
    return load_memory(config_path)


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Noesis — Your thinking layer. Yours forever."""
    pass


# ── start ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", default=None, help="Config file path")
@click.option("--port",   "-p", default=8080,  help="Proxy port (default: 8080)")
@click.option("--ws",           is_flag=True,  help="Also start WebSocket server :8082")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def start(config, port, ws, verbose):
    """Start the Noesis daemon (API proxy + optional WebSocket)."""
    import logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level  = level,
        format = "%(asctime)s  %(name)-24s  %(levelname)s  %(message)s",
    )
    memory = _get_memory(config)
    from daemon import NoesDaemon
    daemon = NoesDaemon(
        memory     = memory,
        proxy_port = port,
        enable_ws  = ws,
        log_level  = "debug" if verbose else "warning",
    )
    asyncio.run(daemon.start())


# ── status ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--user", "-u", default="default", help="User ID")
@click.option("--config", "-c", default=None)
def status(user, config):
    """Show memory stats for a user."""
    memory = _get_memory(config)
    s = memory.status(user_id=user)

    click.echo(f"\n  Noesis memory status — user: {user}")
    click.echo(f"  {'─' * 40}")
    click.echo(f"  Total nodes:    {s['total']}")
    click.echo(f"  Settled:        {s['settled']}")
    click.echo(f"  Provisional:    {s['provisional']}")
    click.echo(f"  Tentative:      {s['tentative']}")
    click.echo(f"  Pipeline queue: {s['pipeline_depth']}")

    if s["total"] > 0:
        settled_pct = s["settled"] / s["total"] * 100
        click.echo(f"\n  Settled rate: {settled_pct:.0f}%")
        if settled_pct < 20:
            click.echo("  ↑ Low — more conversations needed to promote thoughts")
    click.echo()


# ── inspect ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("hash_id")
@click.option("--config", "-c", default=None)
def inspect(hash_id, config):
    """Show the full content of a memory node."""
    memory = _get_memory(config)

    # Try hot store first
    node = memory.vector_store.get(hash_id)
    if not node:
        click.echo(f"Node not found: {hash_id}", err=True)
        sys.exit(1)

    click.echo(f"\n  Node: {hash_id}")
    click.echo(f"  {'─' * 50}")
    click.echo(f"  Type:        {node.get('type', '?')}")
    click.echo(f"  Status:      {node.get('status', '?')}")
    click.echo(f"  Confidence:  {node.get('confidence', 0):.2f}")
    click.echo(f"  Source tool: {node.get('source_tool', '?')}")
    click.echo(f"  Topic:       {node.get('topic_cluster', '?')}")
    click.echo(f"\n  Text:")
    click.echo(f"  {node.get('text', '')}")

    # Also try cold store
    if memory.cold_store:
        try:
            md_body = memory.cold_store.read(hash_id)
            click.echo(f"\n  Vault file body:")
            click.echo(f"  {md_body[:500]}")
        except FileNotFoundError:
            pass
    click.echo()


# ── sync ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--user", "-u", default="default")
@click.option("--config", "-c", default=None)
def sync(user, config):
    """Force sync human edits from Obsidian vault to hot store."""
    memory = _get_memory(config)

    if not memory.cold_store:
        click.echo("No cold store configured.", err=True)
        sys.exit(1)

    # Force re-sync by resetting session state
    memory._synced.discard(user)
    memory._last_sync[user] = 0.0
    memory._sync_if_needed(user)

    click.echo(f"Synced vault changes for user: {user}")


# ── import ────────────────────────────────────────────────────────────────────

@cli.command("import")
@click.argument("file", type=click.Path(exists=True))
@click.option("--source", "-s",
              type=click.Choice(["chatgpt", "claude", "text", "json"]),
              required=True, help="Source format")
@click.option("--user",   "-u", default="default")
@click.option("--config", "-c", default=None)
def import_cmd(file, source, user, config):
    """
    Batch import existing conversations into memory.

    Examples:
      noesis import --source chatgpt conversations.json
      noesis import --source text    notes.txt
    """
    memory = _get_memory(config)
    path   = Path(file)
    count  = 0

    if source == "chatgpt":
        count = _import_chatgpt(memory, path, user)
    elif source == "text":
        count = _import_text(memory, path, user)
    elif source == "json":
        count = _import_json(memory, path, user)
    elif source == "claude":
        click.echo("Claude import requires the API. "
                   "Use `noesis import --source json exported.json` instead.")
        sys.exit(1)

    click.echo(f"Imported {count} conversation turn(s) for user: {user}")
    click.echo("All imported nodes start as 'tentative' — "
               "they will be promoted as you use Noesis normally.")


def _import_chatgpt(memory, path: Path, user: str) -> int:
    """Parse ChatGPT conversation export (conversations.json)."""
    data  = json.loads(path.read_text())
    count = 0
    if isinstance(data, list):
        convos = data
    elif isinstance(data, dict):
        convos = data.get("conversations", [data])
    else:
        return 0

    for convo in convos:
        msgs = []
        mapping = convo.get("mapping", {})
        for node in mapping.values():
            m = node.get("message")
            if not m:
                continue
            role = m.get("author", {}).get("role", "")
            parts = m.get("content", {}).get("parts", [])
            text  = " ".join(str(p) for p in parts if isinstance(p, str))
            if role in ("user", "assistant") and text.strip():
                msgs.append({"role": role, "content": text})

        if msgs:
            memory.add(msgs, user_id=user, source_tool="chatgpt-export")
            count += len(msgs)

    return count


def _import_text(memory, path: Path, user: str) -> int:
    """Import plain text as a single thought."""
    text = path.read_text(encoding="utf-8").strip()
    if text:
        memory.add(text, user_id=user, source_tool="text-import")
        return 1
    return 0


def _import_json(memory, path: Path, user: str) -> int:
    """Import a JSON array of {role, content} messages."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        memory.add(data, user_id=user, source_tool="json-import")
        return len(data)
    return 0


# ── eval ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--user", "-u", default="eval_cli")
@click.option("--config", "-c", default=None)
def eval(user, config):
    """Run injection accuracy test and print score."""
    click.echo("\nRunning injection accuracy test…\n")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_injection.py::test_injection_accuracy_summary",
         "-v", "--tb=short", "--no-header"],
        capture_output=False,
    )
    sys.exit(result.returncode)


# ── mcp ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", default=None)
def mcp(config):
    """Start MCP server over stdio (for Claude Desktop integration)."""
    memory = _get_memory(config)
    from noesis.adapters.mcp import run_stdio
    asyncio.run(run_stdio(memory))


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
