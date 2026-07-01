#!/usr/bin/env python3
"""
Supersession fix validation — does _maybe_supersede actually retire old stances?

Tests the REAL pipeline path (from_config_file, so the consolidation pipeline
including the new _maybe_supersede runs). For each contradiction scenario:
  1. Add the OLD stance via memory.add() — it consolidates to settled/provisional
  2. Add the NEW stance via memory.add() — supersession should retire the OLD
  3. Query — the OLD node should now be status=superseded (not injected)

Reports the leak rate (OLD stance still injected). Compare to the 93% baseline.
"""
import os, sys, json, time, tempfile, yaml, sqlite3
from datetime import datetime
from pathlib import Path

NOESIS_DIR = "/Users/mac27ssd/Noesis"
RESULTS = Path("/Users/mac27ssd/ZCodeProject/noesis_experiment/results")
sys.path.insert(0, NOESIS_DIR)

# Same 30 scenarios as the contradiction experiment
SCENARIOS = [
    ("I prefer MySQL for my database", "I switched to PostgreSQL, it has better JSON support", "What database do I use now?", "postgresql", "mysql", "database"),
    ("I use Python for everything", "I have moved to Rust for systems programming", "What language do I program in now?", "rust", "python", "language"),
    ("I think monoliths are the way to go", "I now believe microservices are better for scaling", "What is my current view on architecture?", "microservice", "monolith", "architecture"),
    ("I deploy on AWS", "I migrated everything to GCP last month", "Which cloud do I deploy on now?", "gcp", "aws", "cloud"),
    ("I like JavaScript", "I switched to TypeScript, the type safety is worth it", "What do I code in now?", "typescript", "javascript", "language"),
    ("I use REST APIs everywhere", "I have moved to GraphQL for complex apps", "What API style do I prefer now?", "graphql", "rest", "api"),
    ("I use Vim", "I switched to Emacs recently", "Which editor do I use now?", "emacs", "vim", "editor"),
    ("I prefer Redis for caching", "I moved to Memcached, simpler for my needs", "What cache do I use now?", "memcached", "redis", "cache"),
    ("I use RabbitMQ for messaging", "I migrated to Kafka for higher throughput", "What message queue do I use now?", "kafka", "rabbitmq", "messaging"),
    ("I prefer MongoDB", "I switched to PostgreSQL for relational integrity", "What database do I use now?", "postgresql", "mongodb", "database"),
    ("I use Java for backend", "I have moved to Go for concurrency", "What backend language do I use now?", "go", "java", "language"),
    ("I deploy on Heroku", "I migrated to Kubernetes on AWS", "Where do I deploy now?", "kubernetes", "heroku", "cloud"),
    ("I use SVN for version control", "I switched to Git years ago", "What version control do I use now?", "git", "svn", "tools"),
    ("I prefer SOAP web services", "I have moved to REST for simplicity", "What web service style do I use now?", "rest", "soap", "api"),
    ("I use Memcached for sessions", "I migrated to Redis for persistence", "What do I use for sessions now?", "redis", "memcached", "cache"),
    ("I prefer PHP for web", "I switched to Node.js for async I/O", "What web runtime do I use now?", "node", "php", "language"),
    ("I use Apache as my web server", "I moved to Nginx for performance", "What web server do I use now?", "nginx", "apache", "devops"),
    ("I prefer Bash scripting", "I have switched to Python for scripting", "What do I script in now?", "python", "bash", "language"),
    ("I use Firebase for backend", "I migrated to Supabase for open-source", "What BaaS do I use now?", "supabase", "firebase", "backend"),
    ("I prefer XML for data", "I switched to JSON for lighter payloads", "What data format do I use now?", "json", "xml", "api"),
    ("I use Jenkins for CI", "I migrated to GitHub Actions last quarter", "What CI tool do I use now?", "github actions", "jenkins", "devops"),
    ("I prefer Docker Swarm", "I have moved to Kubernetes for orchestration", "What orchestrator do I use now?", "kubernetes", "swarm", "devops"),
    ("I use MySQL for analytics", "I switched to ClickHouse for columnar queries", "What analytics database do I use now?", "clickhouse", "mysql", "data"),
    ("I prefer Grunt for builds", "I moved to Webpack for module bundling", "What build tool do I use now?", "webpack", "grunt", "frontend"),
    ("I use Vue for frontend", "I have switched to Svelte for smaller bundles", "What frontend framework do I use now?", "svelte", "vue", "frontend"),
    ("I prefer Stripe for payments", "I migrated to Adyen for global reach", "What payment processor do I use now?", "adyen", "stripe", "business"),
    ("I use ElasticSearch for search", "I moved to Meilisearch for simplicity", "What search engine do I use now?", "meilisearch", "elastic", "data"),
    ("I prefer Webpack for bundling", "I switched to Vite for faster builds", "What bundler do I use now?", "vite", "webpack", "frontend"),
    ("I use Vercel for hosting", "I migrated to Cloudflare Pages for edge", "Where do I host now?", "cloudflare", "vercel", "cloud"),
    ("I prefer SendGrid for email", "I moved to Postmark for deliverability", "What email service do I use now?", "postmark", "sendgrid", "business"),
]

def run():
    import logging; logging.disable(logging.CRITICAL)
    from noesis.memory.main import Memory

    print("="*72)
    print("  SUPERSESSION FIX VALIDATION (real pipeline path)")
    print(f"  scenarios: {len(SCENARIOS)}")
    print(f"  baseline leak rate (before fix): 28/30 = 93%")
    print("="*72)

    results = []
    new_surfaced = 0
    old_leaked = 0
    superseded_count = 0

    for i, (old, new, query, exp_new, exp_old, cluster) in enumerate(SCENARIOS):
        uid = f"sfix_{i}"
        tmp = Path(tempfile.mkdtemp(prefix=f"sfix_{i}_"))

        # Use from_config_file so the pipeline (with _maybe_supersede) runs
        cfg = yaml.safe_load(open(os.path.expanduser("~/.noesis/config.yaml")))
        cfg["vector_store"]["config"]["db_path"] = str(tmp / "hot.db")
        cfg["cold_store"]["config"]["vault_path"] = str(tmp / "vault")
        cfg_path = tmp / "c.yaml"; yaml.safe_dump(cfg, open(cfg_path, "w"))
        m = Memory.from_config_file(str(cfg_path))

        # Add OLD stance — let pipeline consolidate it
        m.add(old, user_id=uid, type="preference", source_tool="eval", topic_cluster=cluster)
        time.sleep(1.5)  # let pipeline settle

        # Add NEW stance — supersession should fire here
        m.add(new, user_id=uid, type="preference", source_tool="eval", topic_cluster=cluster)
        time.sleep(1.5)

        # Check: is the OLD node now superseded?
        db = sqlite3.connect(str(tmp / "hot.db"))
        rows = db.execute("SELECT status, text FROM items WHERE user_id=?", (uid,)).fetchall()
        db.close()

        old_superseded = any("superseded" == r[0] and exp_old in r[1].lower() for r in rows)
        if old_superseded:
            superseded_count += 1

        # Query via build_context — does OLD leak?
        ctx = m.build_context(query, user_id=uid) or ""
        new_present = exp_new.lower() in ctx.lower()
        old_present = exp_old.lower() in ctx.lower()

        if new_present: new_surfaced += 1
        if old_present: old_leaked += 1

        mark = "✅" if (new_present and not old_present) else "❌"
        print(f"  {mark} {query[:42]:44s} new={new_present} old_leak={old_present} superseded={old_superseded}")

        results.append({"query": query, "old": old, "new": new,
                        "new_surfaced": new_present, "old_leaked": old_present,
                        "old_superseded": old_superseded})

    print(f"\n{'='*72}")
    print("  RESULTS")
    print(f"{'='*72}")
    print(f"  OLD nodes superseded (status changed): {superseded_count}/{len(SCENARIOS)}")
    print(f"  NEW stance surfaced in context:        {new_surfaced}/{len(SCENARIOS)}")
    print(f"  OLD stance LEAKED into context:         {old_leaked}/{len(SCENARIOS)}  ← lower is better")
    print(f"\n  BASELINE (before fix): 28/30 = 93% leak")
    print(f"  AFTER FIX:              {old_leaked}/{len(SCENARIOS)} = {100*old_leaked/len(SCENARIOS):.0f}% leak")
    print(f"  IMPROVEMENT:            {100*(28-old_leaked)/30:.0f}pp reduction")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS / f"supersession_fix_validation_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "baseline_leak": 28, "n": len(SCENARIOS),
                   "superseded_count": superseded_count, "new_surfaced": new_surfaced,
                   "old_leaked": old_leaked, "details": results}, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {path}")

if __name__ == "__main__":
    run()
