#!/usr/bin/env python3
"""
Live experiment status tracker.
Run in terminal: python3 experiments/status.py
Or: watch -n5 python3 experiments/status.py
"""
import os, re, time, sqlite3
from datetime import datetime
from pathlib import Path

DAEMON_LOG = Path("/tmp/noesis_daemon.log")
DB_PATH = os.path.expanduser("~/.noesis/hot.db")
RESULTS_DIR = Path("/Users/mac27ssd/Noesis/experiments/results")
START_TIME = None

def get_last_request():
    """Get timestamp of last HTTP request from daemon log."""
    if not DAEMON_LOG.exists():
        return None, 0
    lines = DAEMON_LOG.read_text(errors="ignore").strip().split("\n")
    total = 0
    last_ts = None
    for line in lines:
        if "HTTP Request: POST" in line:
            total += 1
            m = re.search(r"(\d{2}:\d{2}:\d{2})", line)
            if m:
                last_ts = m.group(1)
    return last_ts, total

def get_process_status():
    """Check if the experiment process is running."""
    import subprocess
    r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    for line in r.stdout.split("\n"):
        if "ab_ollama_en" in line and "grep" not in line:
            parts = line.split()
            pid = parts[1]
            elapsed = parts[9] if len(parts) > 9 else "?"
            return True, pid, elapsed
    return False, None, None

def get_memory_count():
    """Count total memories in DB."""
    if not os.path.exists(DB_PATH):
        return 0
    db = sqlite3.connect(DB_PATH)
    count = db.execute("SELECT COUNT(*) FROM items WHERE status != 'superseded'").fetchone()[0]
    db.close()
    return count

def get_profiles_built():
    """Count how many of the 30 profiles have memories."""
    if not os.path.exists(DB_PATH):
        return 0
    db = sqlite3.connect(DB_PATH)
    count = 0
    for i in range(1, 31):
        n = db.execute("SELECT COUNT(*) FROM items WHERE user_id=?", (f"prof_{i}",)).fetchone()[0]
        if n > 0:
            count += 1
    db.close()
    return count

def check_results():
    """Check for completed result files."""
    files = sorted(RESULTS_DIR.glob("ab_ollama_en_*.json"), key=lambda f: f.stat().st_mtime)
    return files

def main():
    print("\033[2J\033[H", end="")  # clear screen
    print("=" * 64)
    print("  Noesis A/B Experiment — Live Status")
    print("=" * 64)

    # Process status
    running, pid, elapsed = get_process_status()
    if running:
        print(f"\n  ● RUNNING (PID {pid})")
    else:
        print(f"\n  ○ NOT RUNNING")

    # Last request
    last_ts, total_reqs = get_last_request()
    now = datetime.now().strftime("%H:%M:%S")
    print(f"  Current time:     {now}")
    if last_ts:
        print(f"  Last API request: {last_ts} ({total_reqs} total requests)")

    # Memory store
    mem_count = get_memory_count()
    profiles = get_profiles_built()
    print(f"\n  Memory store:     {mem_count} thoughts")
    print(f"  Profiles built:   {profiles}/30")

    # Estimate progress
    if running and total_reqs > 0:
        # Each question = 2 calls (proxy + direct), but proxy calls go through daemon
        # We count only proxy requests (half of total API calls)
        proxy_reqs = total_reqs
        # Subtract the ~35 build requests from earlier
        exp_reqs = max(0, proxy_reqs - 35)
        questions_done = exp_reqs  # roughly 1 proxy request per question
        total_questions = 150
        pct = min(100, questions_done / total_questions * 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n  Progress: [{bar}] {pct:.0f}%")
        print(f"  ~{questions_done}/{total_questions} questions completed")

    # Check for results
    results = check_results()
    if results:
        print(f"\n  ✓ Completed result files:")
        for r in results:
            print(f"    {r.name}")

    # Queue status
    print(f"\n  Queue:")
    print(f"    [✓] Gemini Flash  (n=111, done)")
    if running:
        print(f"    [⟳] gemma3:4b     (running now)")
        print(f"    [ ] qwen2.5:3b    (queued)")
    else:
        # Check which models have results
        done_models = set()
        for r in results:
            if "gemma3" in r.name:
                done_models.add("gemma3:4b")
            if "qwen2.5" in r.name:
                done_models.add("qwen2.5:3b")
        for model in ["gemma3:4b", "qwen2.5:3b"]:
            status = "[✓]" if model in done_models else "[ ]"
            print(f"    {status} {model}")

    print(f"\n{'─' * 64}")
    print(f"  Refresh every 5s. Ctrl+C to exit.")


if __name__ == "__main__":
    try:
        while True:
            main()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n  Stopped.")
