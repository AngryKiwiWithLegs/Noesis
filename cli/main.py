"""
cli/main.py

Noesis command-line interface.

Commands:
  noesis start   [--config PATH] [--port N]   Start the daemon
  noesis status  [--user USER]                Memory stats
  noesis inspect <hash_id>                    Show a memory node
  noesis sync    [--user USER]                Force sync vault → hot store
  noesis recluster [--dry-run] [--embedding]  Recompute topic clusters
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
@click.option("--verbose", "-v", is_flag=True,
              help="Show per-file sync details")
def sync(user, config, verbose):
    """Force sync human edits from Obsidian vault to hot store.

    Bidirectional: updates text + metadata from vault edits, detects
    deleted vault files (soft-deletes in hot store), and imports
    new vault-only .md files.
    """
    memory = _get_memory(config)

    if not memory.cold_store:
        click.echo("No cold store configured.", err=True)
        sys.exit(1)

    # Force full sync (phases A + B + C)
    memory._synced.discard(user)
    memory._last_sync[user] = 0.0
    report = memory.sync_full(user)

    parts = []
    if report["text_edits"]:
        parts.append(f"{report['text_edits']} text edit(s)")
    if report["metadata_edits"]:
        parts.append(f"{report['metadata_edits']} metadata change(s)")
    if report["deletions"]:
        parts.append(f"{report['deletions']} deletion(s)")
    if report["additions"]:
        parts.append(f"{report['additions']} addition(s)")

    if parts:
        click.echo(f"Synced for {user}: " + ", ".join(parts))
    else:
        click.echo(f"Everything up-to-date for {user}.")

    if verbose and report:
        click.echo(f"\n  text_edits:     {report['text_edits']}")
        click.echo(f"  metadata_edits: {report['metadata_edits']}")
        click.echo(f"  deletions:      {report['deletions']}")
        click.echo(f"  additions:      {report['additions']}")


# ── recluster ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--user", "-u", default=None,
              help="Recluster one user (default: all users)")
@click.option("--config", "-c", default=None)
@click.option("--dry-run", is_flag=True,
              help="Show the distribution change without writing")
@click.option("--embedding", is_flag=True,
              help="Use embedding-based inference (slower, more accurate)")
def recluster(user, config, dry_run, embedding):
    """Recompute topic_cluster for all nodes using improved inference.

    Fixes nodes stuck in the 'general' bucket from the old 3-keyword inferer.
    By default uses the expanded bilingual keyword map (fast). With
    --embedding, also uses nearest-neighbor similarity against already-
    classified nodes (slower, catches keyword misses).

    \b
    Examples:
      noesis recluster --dry-run          # preview the change
      noesis recluster                    # apply keyword reclassification
      noesis recluster --embedding        # apply keyword + embedding
    """
    memory = _get_memory(config)
    from noesis.thoughts.extractor import (
        _infer_cluster, _infer_cluster_embedding,
    )
    from collections import Counter

    # Determine which users to process
    vs = memory.vector_store
    try:
        users = [r[0] for r in vs._con.execute(
            "SELECT DISTINCT user_id FROM items WHERE user_id != ''"
        ).fetchall()] if user is None else [user]
    except Exception:
        users = [user or "default"]

    # Build the embedding reference set once (nodes already keyword-classified)
    ref_nodes: list[tuple[str, list[float]]] = []
    if embedding:
        for uid in users:
            for n in vs.get_all(uid):
                kw = _infer_cluster(n.get("text", ""))
                if kw != "general":
                    v = vs.get_vector(n["id"]) if "id" in n else vs.get_vector(n.get("hash_id",""))
                    if v:
                        ref_nodes.append((kw, v))

    changed = 0
    before = Counter()
    after = Counter()

    for uid in users:
        nodes = vs.get_all(uid)
        for n in nodes:
            text = n.get("text", "")
            old = n.get("topic_cluster", "general")
            before[old] += 1

            if embedding and ref_nodes:
                new = _infer_cluster_embedding(text, memory.embedding, ref_nodes)
            else:
                new = _infer_cluster(text)
            after[new] += 1

            if new != old:
                changed += 1
                if not dry_run:
                    hid = n.get("hash_id") or n.get("id")
                    vs.update(hid, {"topic_cluster": new})
                    if memory.cold_store and hasattr(memory.cold_store, "_patch_frontmatter"):
                        try:
                            memory.cold_store._patch_frontmatter(hid, {"topic_cluster": new})
                        except Exception:
                            pass

    total = sum(before.values())
    action = "DRY RUN — " if dry_run else ""
    click.echo(f"\n  {action}Reclustered {total} node(s) across {len(users)} user(s)")
    click.echo(f"  Changed: {changed}   ({'would write' if dry_run else 'written'} to hot + cold store)")
    click.echo(f"\n  {'cluster':20s} {'before':>7s} {'after':>7s}")
    click.echo(f"  {'-'*20} {'-'*7} {'-'*7}")
    for c in sorted(set(before) | set(after), key=lambda k: -after.get(k, 0)):
        click.echo(f"  {c:20s} {before.get(c,0):7d} {after.get(c,0):7d}")
    g_before = before.get("general", 0)
    g_after = after.get("general", 0)
    click.echo(f"\n  general: {g_before} ({g_before/max(total,1)*100:.0f}%) → "
               f"{g_after} ({g_after/max(total,1)*100:.0f}%)\n")


# ── import ────────────────────────────────────────────────────────────────────

@cli.command("import")
@click.argument("file", type=click.Path(exists=True))
@click.option("--source", "-s",
              type=click.Choice(["auto", "chatgpt", "claude", "gemini", "meta", "text", "json"]),
              required=True, help="Source format")
@click.option("--user",   "-u", default="default")
@click.option("--config", "-c", default=None)
@click.option("--dry-run", is_flag=True, default=False,
              help="Parse and report counts without writing to DB")
def import_cmd(file, source, user, config, dry_run):
    """
    Batch import existing conversations into memory.

    Supports ChatGPT, Claude, Gemini, and Meta AI exports.
    Use --source auto to detect the format automatically.

    \b
    Examples:
      noesis import --source auto    conversations.json
      noesis import --source chatgpt conversations.json
      noesis import --source claude  conversations.json
      noesis import --source gemini  gemini_export.html
      noesis import --source meta    message_1.json
      noesis import --source text    notes.txt
      noesis import --source auto    export.json --dry-run
    """
    from noesis.importers import normalize, detect_source

    path = Path(file)

    # Resolve auto before parsing
    effective_source = source
    if source == "auto":
        effective_source = detect_source(path)
        click.echo(f"Auto-detected source: {effective_source}")

    convos = normalize(effective_source, path)
    if not convos:
        click.echo("No conversations found to import.")
        return

    total_turns = sum(len(msgs) for _, msgs, _ in convos)
    click.echo(f"Found {len(convos)} conversation(s), {total_turns} turn(s) total.")

    if dry_run:
        for conv_id, msgs, ts in convos:
            preview = (msgs[0]["content"][:60] + "...") if msgs and msgs[0]["content"] else ""
            click.echo(f"  [{conv_id}] {len(msgs)} turns  ts={ts}  {preview}")
        click.echo("Dry run — nothing written to DB.")
        return

    memory = _get_memory(config)
    count  = 0
    for conv_id, msgs, ts in convos:
        result = memory.add(
            msgs,
            user_id=user,
            source_tool=f"{effective_source}-export",
            session_id=conv_id,
            created_at=ts,
        )
        count += len(result.get("results", []))

    click.echo(f"Imported {count} conversation turn(s) for user: {user}")
    click.echo("All imported nodes start as 'tentative' — "
               "they will be promoted as you use Noesis normally.")


# ── export ───────────────────────────────────────────────────────────────────

@cli.command("export")
@click.argument("output", type=click.Path())
@click.option("--format", "-f",
              type=click.Choice(["json", "markdown", "finetune"]),
              required=True, help="Output format")
@click.option("--user",   "-u", default="default")
@click.option("--config", "-c", default=None)
@click.option("--include-wiki", is_flag=True, default=False,
              help="Also export wiki pages (json/markdown only)")
@click.option("--include-superseded", is_flag=True, default=False,
              help="Include superseded/retired thoughts")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show counts without writing files")
def export_cmd(output, format, user, config, include_wiki, include_superseded, dry_run):
    """
    Export your memory store to JSON, Markdown, or fine-tuning format.

    \b
    Examples:
      noesis export --format json     memory_backup.json
      noesis export --format markdown ./my_thoughts/ --include-wiki
      noesis export --format finetune training_data.jsonl
      noesis export --format json     out.json --dry-run
    """
    from noesis.exporters import export, count_exportable

    memory = _get_memory(config)

    if dry_run:
        counts = count_exportable(
            memory, user,
            include_superseded=include_superseded,
            include_wiki=include_wiki,
        )
        click.echo(f"Dry run — exportable items for user '{user}':")
        click.echo(f"  Total thoughts:   {counts['total']}")
        click.echo(f"  Confirmed:        {counts['confirmed']}  (settled + provisional)")
        click.echo(f"  Tentative:        {counts['tentative']}")
        if include_superseded:
            click.echo(f"  Superseded:       {counts['superseded']}")
        if include_wiki:
            click.echo(f"  Wiki pages:       {counts['wiki']}")
        if format == "finetune":
            click.echo(f"\n  Finetune export would produce {counts['confirmed']} entries")
        else:
            total = counts["total"] - (0 if include_superseded else counts["superseded"])
            click.echo(f"\n  {format.capitalize()} export would produce {total} thought(s)")
        click.echo("Dry run — nothing written.")
        return

    result = export(
        format, memory, user, output,
        include_wiki=include_wiki,
        include_superseded=include_superseded,
    )

    click.echo(f"Exported {result['thoughts_exported']} thought(s) "
               f"to {result['output_path']} ({format})")
    if include_wiki:
        click.echo(f"Including {result['wiki_exported']} wiki page(s)")


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
