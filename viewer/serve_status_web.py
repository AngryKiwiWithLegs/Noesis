#!/usr/bin/env python3
"""
Noesis Experiment Status — Web Dashboard
=========================================
A live web page showing experiment progress.
Usage: python3 serve_status_web.py [--port 8792]
Then open: http://localhost:8792
"""
import argparse, json, os, re, time, sqlite3, subprocess
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DAEMON_LOG = Path("/tmp/noesis_daemon.log")
DB_PATH = os.path.expanduser("~/.noesis/hot.db")
RESULTS_DIR = Path("/Users/mac27ssd/Noesis/experiments/results")


def get_status():
    """Collect all status data."""
    # Process check — also get the process start time
    r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    running = False
    model = "?"
    pid = None
    proc_start_secs = None  # seconds since midnight when the process started
    for line in r.stdout.split("\n"):
        if "ab_ollama_en" in line and "grep" not in line:
            running = True
            parts = line.split()
            pid = parts[1]
            if "gemma3" in line: model = "gemma3:4b"
            elif "qwen2.5" in line: model = "qwen2.5:3b"
            # ps 'lstart' gives the full start time — use ps -p to get it
            try:
                r2 = subprocess.run(["ps", "-p", pid, "-o", "lstart="],
                                    capture_output=True, text=True)
                # e.g. "Mon Aug 11 01:15:23 2026"
                start_str = r2.stdout.strip()
                m = re.search(r"(\d{2}):(\d{2}):(\d{2})", start_str)
                if m:
                    proc_start_secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            except Exception:
                pass

    # Daemon log — count requests, track timestamps
    # Only count requests AFTER the current process started
    all_reqs = 0       # total lifetime (for display)
    run_reqs = 0       # requests since current process started (for progress)
    recent_reqs = []   # last 30 timestamps (for RPM)
    last_ts = None
    if DAEMON_LOG.exists():
        for line in DAEMON_LOG.read_text(errors="ignore").split("\n"):
            if "HTTP Request: POST" in line:
                all_reqs += 1
                m = re.search(r"(\d{2}):(\d{2}):(\d{2})", line)
                if m:
                    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    secs = h * 3600 + mi * 60 + s
                    recent_reqs.append(secs)
                    last_ts = f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
                    # Count as part of this run only if after process start
                    if proc_start_secs is not None and secs >= proc_start_secs:
                        run_reqs += 1

    # Memory store
    mem_count = 0
    profiles = 0
    if os.path.exists(DB_PATH):
        db = sqlite3.connect(DB_PATH)
        mem_count = db.execute("SELECT COUNT(*) FROM items WHERE status != 'superseded'").fetchone()[0]
        for i in range(1, 31):
            n = db.execute("SELECT COUNT(*) FROM items WHERE user_id=?", (f"prof_{i}",)).fetchone()[0]
            if n > 0: profiles += 1
        db.close()

    # Result files
    results = []
    for f in sorted(RESULTS_DIR.glob("ab_ollama_en_*.json"), key=lambda f: f.stat().st_mtime):
        data = json.loads(f.read_text())
        s = data.get("summary", {})
        results.append({
            "file": f.name,
            "model": data.get("model", "?"),
            "n": s.get("total", 0),
            "with_hit": s.get("with_hit", 0),
            "without_hit": s.get("without_hit", 0),
            "with_pct": round(s.get("with_hit", 0) / max(1, s.get("total", 1)) * 100, 1),
            "without_pct": round(s.get("without_hit", 0) / max(1, s.get("total", 1)) * 100, 1),
        })

    # Progress: use run_reqs (requests since this process started), not all_reqs
    # Each question = ~1 proxy request (the direct call goes to Ollama, not the daemon)
    # Subtract ~35 for the build phase (only applies to the first run that built memories)
    questions_done = min(150, run_reqs)
    pct = questions_done / 150 * 100

    # Requests per minute (from recent 30)
    rpm = 0
    if len(recent_reqs) >= 2:
        # Only consider recent reqs after process start
        if proc_start_secs is not None:
            recent_run = [s for s in recent_reqs if s >= proc_start_secs]
        else:
            recent_run = recent_reqs
        if len(recent_run) >= 2:
            span = recent_run[-1] - recent_run[0]
            if span > 0:
                rpm = round(len(recent_run) / (span / 60), 1)

    # ETA
    eta_min = 0
    if running and pct < 100 and rpm > 0:
        remaining = 150 - questions_done
        eta_min = round(remaining / rpm)

    now = datetime.now().strftime("%H:%M:%S")

    return {
        "running": running, "model": model, "pid": pid,
        "now": now, "last_ts": last_ts, "total_reqs": all_reqs,
        "run_reqs": run_reqs,
        "questions_done": questions_done, "pct": round(pct, 1),
        "rpm": rpm, "eta_min": eta_min,
        "mem_count": mem_count, "profiles": profiles,
        "results": results,
        "queue": {
            "gemini": {"status": "done", "n": 111},
            "gemma": {"status": "running" if (running and "gemma" in model) else
                       ("done" if any(r["model"] == "gemma3:4b" for r in results) else "queued")},
            "qwen": {"status": "running" if (running and "qwen" in model) else
                      ("done" if any(r["model"] == "qwen2.5:3b" for r in results) else "queued")},
        }
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Noesis Experiment Status</title>
<style>
  :root { --bg:#0a0a14; --card:#13131f; --border:rgba(255,255,255,0.08); --green:#22d3ee; --yellow:#fbbf24; --gray:#6b7280; --red:#ef4444; --text:#e5e7eb; --muted:#9ca3af; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"SF Mono","Fira Code",monospace; min-height:100vh; padding:20px; }
  .container { max-width:900px; margin:0 auto; }
  h1 { font-size:22px; font-weight:700; margin-bottom:4px; }
  .subtitle { color:var(--muted); font-size:13px; margin-bottom:24px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:18px; }
  .card h2 { font-size:13px; text-transform:uppercase; letter-spacing:1px; color:var(--muted); margin-bottom:12px; }
  .status-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; vertical-align:middle; }
  .status-dot.live { background:var(--green); animation:pulse 2s ease-in-out infinite; }
  .status-dot.done { background:#34d399; }
  .status-dot.queued { background:var(--gray); }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1);} 50%{opacity:0.5;transform:scale(0.8);} }
  .big-num { font-size:32px; font-weight:700; }
  .label { font-size:12px; color:var(--muted); }
  .progress-bar { width:100%; height:24px; background:rgba(255,255,255,0.06); border-radius:8px; overflow:hidden; margin:8px 0; }
  .progress-fill { height:100%; background:linear-gradient(90deg,#0891b2,#22d3ee); border-radius:8px; transition:width 0.5s ease; }
  .row { display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid var(--border); }
  .row:last-child { border-bottom:none; }
  .queue-item { display:flex; align-items:center; padding:8px 0; font-size:14px; }
  .queue-item .model { flex:1; font-weight:600; }
  .queue-item .state { font-size:12px; color:var(--muted); }
  .result-row { display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--border); font-size:13px; }
  .result-row:last-child { border-bottom:none; }
  .badge { padding:2px 8px; border-radius:6px; font-size:11px; font-weight:600; }
  .badge.with { background:rgba(34,211,238,0.15); color:#22d3ee; }
  .badge.without { background:rgba(107,114,128,0.2); color:#9ca3af; }
  .time { color:var(--muted); font-size:12px; }
  #refresh { font-size:11px; color:var(--muted); text-align:center; margin-top:16px; }
</style>
</head>
<body>
<div class="container">
  <h1>Noesis A/B Experiment Status</h1>
  <div class="subtitle">Live tracker · n=150 per model · English profiles</div>

  <div class="grid">
    <div class="card">
      <h2>Current Run</h2>
      <div id="current-run"></div>
    </div>
    <div class="card">
      <h2>Progress</h2>
      <div id="progress"></div>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px;">
    <h2>Experiment Queue</h2>
    <div id="queue"></div>
  </div>

  <div class="card">
    <h2>Completed Results</h2>
    <div id="results"></div>
  </div>

  <div id="refresh">Auto-refresh every 5s · <span id="last-update"></span></div>
</div>

<script>
async function update() {
  const d = await fetch('/api').then(r => r.json());

  // Current run
  let runHtml = '';
  if (d.running) {
    runHtml = `
      <div><span class="status-dot live"></span><b>${d.model}</b> (PID ${d.pid})</div>
      <div class="row"><span class="label">Time now</span><span>${d.now}</span></div>
      <div class="row"><span class="label">Last API call</span><span>${d.last_ts || '—'}</span></div>
      <div class="row"><span class="label">Speed</span><span>${d.rpm} req/min</span></div>
      <div class="row"><span class="label">ETA</span><span>~${d.eta_min} min remaining</span></div>
    `;
  } else {
    runHtml = '<div style="color:var(--muted);padding:8px 0;">No experiment running.</div>';
  }
  document.getElementById('current-run').innerHTML = runHtml;

  // Progress
  const bar = '█'.repeat(Math.floor(d.pct/2.5)) + '░'.repeat(40 - Math.floor(d.pct/2.5));
  document.getElementById('progress').innerHTML = `
    <div class="big-num">${d.pct}%</div>
    <div class="label">~${d.questions_done}/150 questions</div>
    <div class="progress-bar"><div class="progress-fill" style="width:${d.pct}%"></div></div>
    <div class="row" style="margin-top:8px;"><span class="label">Total API requests</span><span>${d.total_reqs}</span></div>
    <div class="row"><span class="label">Memories in store</span><span>${d.mem_count}</span></div>
    <div class="row"><span class="label">Profiles built</span><span>${d.profiles}/30</span></div>
  `;

  // Queue
  const q = d.queue;
  const dotClass = s => s === 'running' ? 'live' : s === 'done' ? 'done' : 'queued';
  const dotColor = s => s === 'running' ? '#22d3ee' : s === 'done' ? '#34d399' : '#6b7280';
  document.getElementById('queue').innerHTML = `
    <div class="queue-item">
      <span class="status-dot done" style="background:#34d399;"></span>
      <span class="model">Gemini Flash</span>
      <span class="state">✓ done (n=111)</span>
    </div>
    <div class="queue-item">
      <span class="status-dot ${dotClass(q.gemma.status)}" style="background:${dotColor(q.gemma.status)};"></span>
      <span class="model">gemma3:4b</span>
      <span class="state">${q.gemma.status === 'running' ? '⟳ running...' : q.gemma.status === 'done' ? '✓ done' : 'queued'}</span>
    </div>
    <div class="queue-item">
      <span class="status-dot ${dotClass(q.qwen.status)}" style="background:${dotColor(q.qwen.status)};"></span>
      <span class="model">qwen2.5:3b</span>
      <span class="state">${q.qwen.status === 'running' ? '⟳ running...' : q.qwen.status === 'done' ? '✓ done' : 'queued'}</span>
    </div>
  `;

  // Results
  if (d.results.length === 0) {
    document.getElementById('results').innerHTML = '<div style="color:var(--muted);padding:8px 0;">No results yet.</div>';
  } else {
    document.getElementById('results').innerHTML = d.results.map(r => `
      <div class="result-row">
        <span><b>${r.model}</b> (n=${r.n})</span>
        <span>
          <span class="badge with">with ${r.with_pct}%</span>
          <span class="badge without">without ${r.without_pct}%</span>
        </span>
      </div>
    `).join('');
  }

  document.getElementById('last-update').textContent = d.now;
}
update();
setInterval(update, 5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/api"):
            body = json.dumps(get_status(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description="Noesis Experiment Status Dashboard")
    ap.add_argument("--port", type=int, default=8792)
    args = ap.parse_args()
    print(f"  → http://localhost:{args.port}")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
