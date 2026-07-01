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

  noesis wiki ingest <file|url>               Compile a document into wiki pages
  noesis wiki query <text>                    Search wiki pages for knowledge
  noesis wiki answer <hash> <page_id>         Mark a question answered by a wiki page
  noesis wiki lint                            Audit wiki for open questions / orphans
  noesis wiki status                          Show wiki page stats
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
    from noesis.daemon import load_memory
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
    from noesis.daemon import NoesDaemon
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


# ── wiki ──────────────────────────────────────────────────────────────────────

def _load_config_dict(config: Optional[str] = None) -> dict:
    """Load the raw config dict from the config file, mirroring daemon.load_memory.

    There is no public `_load_config` in noesis.daemon; Memory.from_config_file
    consumes the dict internally but does not return it. We replicate the same
    file-resolution logic so the wiki helpers can read llm/cold_store settings
    without a Memory instance.
    """
    import yaml
    from pathlib import Path
    if config is None:
        config = str(Path("~/.noesis/config.yaml").expanduser())
    p = Path(config).expanduser()
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_vault_path(config: Optional[str] = None) -> str:
    """Resolve the vault path from config (shared with the cold store)."""
    cfg = _load_config_dict(config)
    cs = cfg.get("cold_store", {})
    return cs.get("config", {}).get("vault_path", "~/NoesisVault")


def _build_wiki_extractor(config: Optional[str] = None):
    """Pick Cloud or Mock wiki extractor, mirroring Memory._attach_pipeline."""
    import os
    cfg = _load_config_dict(config)
    llm = cfg.get("llm", {})
    provider = llm.get("provider", "anthropic")
    model = llm.get("model", "claude-haiku-4-5-20251001")
    api_key = llm.get("api_key")
    if not api_key:
        env_var = {"anthropic": "ANTHROPIC_API_KEY",
                   "openai": "OPENAI_API_KEY"}.get(provider, "OPENAI_API_KEY")
        api_key = os.environ.get(env_var)
    if api_key:
        from noesis.wiki import CloudWikiExtractor
        return CloudWikiExtractor(provider=provider, model=model, api_key=api_key)
    from noesis.wiki import MockWikiExtractor
    return MockWikiExtractor()


@cli.group()
def wiki():
    """LLM Wiki operations (ingest documents, audit knowledge)."""
    pass


@wiki.command()
@click.argument("source")
@click.option("--config", "-c", default=None, help="Config file path")
@click.option("--chunk-tokens", default=800, help="Max tokens per chunk")
def ingest(source, config, chunk_tokens):
    """Compile a document (file path or URL) into wiki pages.

    Supports markdown, text, and PDF. Re-ingesting the same source updates
    pages without duplicating them.

    \b
    Examples:
      noesis wiki ingest notes.md
      noesis wiki ingest paper.pdf
      noesis wiki ingest https://example.com/article
    """
    vault = _get_vault_path(config)
    extractor = _build_wiki_extractor(config)
    from noesis.wiki import WikiIngestor
    ing = WikiIngestor(vault, extractor, chunk_tokens=chunk_tokens)
    try:
        pages = ing.ingest(source)
    except FileNotFoundError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    except ImportError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    if not pages:
        click.echo("No reusable knowledge found in source.")
        return
    click.echo(f"Ingested {len(pages)} page(s) from {source}:")
    for pid in pages:
        click.echo(f"  - wiki/{pid}")
    click.echo(f"\nVault: {Path(vault).expanduser() / 'wiki'}")


@wiki.command()
@click.option("--config", "-c", default=None)
def lint(config):
    """Audit the wiki: open questions, orphan pages, contradictions, gaps."""
    memory = _get_memory(config)
    vault = _get_vault_path(config)
    from noesis.wiki import WikiLinter
    linter = WikiLinter(vault, vector_store=memory.vector_store)
    report = linter.run()

    click.echo(f"\n  Wiki lint report — {Path(vault).expanduser().name}")
    click.echo(f"  {'─' * 44}")

    oq = report["open_questions"]
    click.echo(f"  Open questions: {len(oq)}")
    for q in oq[:10]:
        click.echo(f"    · [{q['hash'][:8]}] {q['text'][:60]}")
    if len(oq) > 10:
        click.echo(f"    … and {len(oq)-10} more")

    orphans = report["orphan_pages"]
    click.echo(f"  Orphan pages:   {len(orphans)}")
    for pid in orphans[:10]:
        click.echo(f"    · wiki/{pid}")

    contradictions = report["contradictions"]
    click.echo(f"  Contradictions: {len(contradictions)}")
    for a, b, dom in contradictions[:10]:
        click.echo(f"    · wiki/{a} vs wiki/{b} (domain: {dom})")

    gaps = report["coverage_gaps"]
    click.echo(f"  Coverage gaps:  {len(gaps)}")
    for g in gaps[:10]:
        click.echo(f"    · cluster '{g}' has thoughts but no wiki page")
    click.echo()


@wiki.command()
@click.option("--config", "-c", default=None)
def status(config):
    """Show wiki page count and recent log entries."""
    vault = _get_vault_path(config)
    from noesis.wiki import WikiWriter
    writer = WikiWriter(vault)
    pages = writer.list_pages()

    click.echo(f"\n  Wiki status — {Path(vault).expanduser().name}")
    click.echo(f"  {'─' * 40}")
    click.echo(f"  Total pages: {len(pages)}")

    # Cluster breakdown
    clusters: dict[str, int] = {}
    for pid in pages:
        p = writer.read_page(pid)
        if p:
            c = p.topic_cluster or "general"
            clusters[c] = clusters.get(c, 0) + 1
    if clusters:
        click.echo(f"  Clusters:    {len(clusters)}")
        for c, n in sorted(clusters.items(), key=lambda x: -x[1]):
            click.echo(f"    · {c}: {n}")

    # Last few log lines
    log_path = Path(vault).expanduser() / "wiki" / "log.md"
    if log_path.exists():
        lines = log_path.read_text().strip().splitlines()
        recent = [l for l in lines if l.startswith("- ")][-5:]
        if recent:
            click.echo(f"\n  Recent activity:")
            for l in recent:
                click.echo(f"  {l}")
    click.echo()


@wiki.command()
@click.argument("query_text")
@click.option("--config", "-c", default=None)
@click.option("--top-k", "-k", default=5, help="Max pages to return")
def query(query_text, config, top_k):
    """Search compiled wiki pages for knowledge relevant to a query.

    Ranks pages by embedding similarity (if an embedding model is
    configured) with a BM25 keyword fallback. Prints ranked hits with
    their [[wiki/...]] citation and a short excerpt.

    \b
    Examples:
      noesis wiki query "vector search in sqlite"
      noesis wiki query "postgres json support" -k 3
    """
    vault = _get_vault_path(config)
    # Use the configured embedding model for accurate ranking when present.
    emb = None
    try:
        memory = _get_memory(config)
        emb = getattr(memory, "embedding", None)
    except Exception:
        pass  # fall back to keyword ranking

    from noesis.context.signals import wiki_signal
    hits = wiki_signal(query_text, vault, embedding_model=emb, top_k=top_k)

    if not hits:
        click.echo("No matching wiki pages found.")
        click.echo(f"(Vault: {Path(vault).expanduser() / 'wiki'})")
        return

    click.echo(f"\n  Wiki query — top {len(hits)} match(es)\n")
    for i, h in enumerate(hits, 1):
        score = h.get("score", 0.0)
        cite = h.get("source", "")
        cluster = h.get("topic_cluster", "")
        text = " ".join(h.get("text", "").split())
        click.echo(f"  {i}. {cite}  ({cluster}, score {score:.2f})")
        click.echo(f"     {text[:120]}{'…' if len(text) > 120 else ''}")
    click.echo()


@wiki.command()
@click.argument("hash_id")
@click.argument("page_id")
@click.option("--config", "-c", default=None)
def answer(hash_id, page_id, config):
    """Mark an open question (a thought) as answered by a wiki page.

    Implements schema rule #2's write side: a question thought gains
    status:answered and an answered_by: [[wiki/{page_id}]] reference.

    \b
    Example:
      noesis wiki answer a1b2c3d4e5f6 sqlite-vec
    """
    memory = _get_memory(config)
    vault = _get_vault_path(config)
    from noesis.wiki import WikiLinter
    linter = WikiLinter(vault, vector_store=memory.vector_store)

    ok = linter.mark_answered(hash_id, page_id)
    if not ok:
        click.echo(
            f"Could not answer: wiki page '{page_id}' does not exist.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Question {hash_id[:8]} marked answered by wiki/{page_id}.")
    click.echo(f"  status: answered")
    click.echo(f"  answered_by: [[wiki/{page_id}]]")


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
