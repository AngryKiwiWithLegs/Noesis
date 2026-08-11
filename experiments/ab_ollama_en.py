#!/usr/bin/env python3
"""
English A/B Experiment for Ollama Local Models (n=150)
=======================================================
Runs the SAME 30 English profiles × 5 questions = 150 questions as the
Gemini experiment (ab_comparison_en.py), but with local Ollama models.

This gives all three models matching sample sizes for valid statistical
comparison in the paper.

Two groups per question:
  A (treatment): via Noesis proxy → memory injected → model answers
  B (control):   direct to Ollama → no memory → model answers

Usage:
    # First, build memories by running the Gemini experiment
    # (or use --build-only to build just the memory store)
    python3 ab_ollama_en.py --model gemma3:4b
    python3 ab_ollama_en.py --model qwen2.5:3b
"""
import os, sys, json, time, argparse, sqlite3
import httpx
from datetime import datetime
from pathlib import Path

PROXY_URL = "http://127.0.0.1:8080/v1/chat/completions"
OLLAMA_DIRECT = "http://localhost:11434/v1/chat/completions"
RESULTS_DIR = Path("/Users/mac27ssd/Noesis/experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import the profiles from the Gemini experiment to guarantee identical setup
sys.path.insert(0, str(Path(__file__).parent))
from ab_comparison_en import PROFILES


def ask_proxy(model, user_id, question, retries=2):
    """Group A: via Noesis proxy (with memory)."""
    for attempt in range(retries + 1):
        try:
            r = httpx.post(PROXY_URL, headers={
                "Authorization": "Bearer dummy",
                "X-User-ID": user_id, "Content-Type": "application/json",
            }, json={"model": model, "messages": [{"role": "user", "content": question}],
                     "max_tokens": 150, "stream": False}, timeout=120)
            if r.status_code == 200:
                return True, r.json()["choices"][0]["message"]["content"].strip()
            if attempt < retries:
                time.sleep(2); continue
            return False, f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            if attempt < retries: time.sleep(2); continue
            return False, f"ERR: {str(e)[:60]}"
    return False, "retries exhausted"


def ask_direct(model, question, retries=2):
    """Group B: direct to Ollama (no memory)."""
    for attempt in range(retries + 1):
        try:
            r = httpx.post(OLLAMA_DIRECT, headers={"Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": question}],
                      "max_tokens": 150, "stream": False}, timeout=120)
            if r.status_code == 200:
                return True, r.json()["choices"][0]["message"]["content"].strip()
            if attempt < retries:
                time.sleep(2); continue
            return False, f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            if attempt < retries: time.sleep(2); continue
            return False, f"ERR: {str(e)[:60]}"
    return False, "retries exhausted"


def check_memories():
    """Check that all profiles have memories built in the DB."""
    db_path = os.path.expanduser("~/.noesis/hot.db")
    if not os.path.exists(db_path):
        return False, "No hot.db found"
    db = sqlite3.connect(db_path)
    missing = []
    for p in PROFILES:
        n = db.execute("SELECT COUNT(*) FROM items WHERE user_id=?", (p["name"],)).fetchone()[0]
        if n == 0:
            missing.append(p["name"])
    db.close()
    if missing:
        return False, f"Missing memories for: {', '.join(missing[:5])}..."
    return True, f"All {len(PROFILES)} profiles have memories"


def run(model):
    print(f"\n{'=' * 64}")
    print(f"  Ollama A/B Experiment (English, n={len(PROFILES) * 5})")
    print(f"  Model: {model}")
    print(f"{'=' * 64}")

    # Check memories exist
    ok, msg = check_memories()
    if not ok:
        print(f"\n  ⚠️  {msg}")
        print(f"  Run ab_comparison_en.py first to build the memory store.")
        return None
    print(f"  {msg} ✓")

    # Check both services are up
    try:
        httpx.get("http://localhost:8080/v1/models", timeout=5)
        print("  Noesis proxy ✓")
    except Exception:
        print("  ⚠️  Noesis proxy not running on :8080. Start with: noesis start")
        return None

    try:
        httpx.get("http://localhost:11434/api/tags", timeout=5)
        print("  Ollama ✓")
    except Exception:
        print("  ⚠️  Ollama not running on :11434")
        return None

    results = []
    summary = {"with_hit": 0, "without_hit": 0, "total": 0, "errors": 0}
    q_num = 0

    for profile in PROFILES:
        uid = profile["name"]
        for q_info in profile["questions"]:
            q_num += 1
            q, expect = q_info["q"], q_info["expect"]

            # Group A: with memory (via proxy)
            ok_a, ans_a = ask_proxy(model, uid, q)
            hit_a = expect.lower() in ans_a.lower() if ok_a else False

            # Group B: without memory (direct)
            ok_b, ans_b = ask_direct(model, q)
            hit_b = expect.lower() in ans_b.lower() if ok_b else False

            if not ok_a or not ok_b:
                summary["errors"] += 1

            rec = {
                "profile": uid, "model": model, "question": q, "expect": expect,
                "with_mem_hit": hit_a, "without_mem_hit": hit_b,
                "with_mem_ans": ans_a[:150], "without_mem_ans": ans_b[:150],
            }
            results.append(rec)

            summary["total"] += 1
            if hit_a: summary["with_hit"] += 1
            if hit_b: summary["without_hit"] += 1

            symbol = "🟢" if (hit_a and not hit_b) else ("🟡" if hit_a == hit_b else "🔴")
            print(f"  [{q_num:3d}/{len(PROFILES)*5}] {symbol} {uid:10s} | "
                  f"with={'✅' if hit_a else '❌'} without={'✅' if hit_b else '❌'} "
                  f"('{expect}')")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "experiment": "ab_ollama_en", "model": model, "timestamp": ts,
        "n_profiles": len(PROFILES), "n_questions": len(PROFILES) * 5,
        "summary": summary, "details": results,
    }
    safe_model = model.replace(":", "_")
    path = RESULTS_DIR / f"ab_ollama_en_{safe_model}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with_rate = summary["with_hit"] / summary["total"] * 100 if summary["total"] else 0
    without_rate = summary["without_hit"] / summary["total"] * 100 if summary["total"] else 0

    print(f"\n  {'─' * 50}")
    print(f"  Model: {model} (n={summary['total']})")
    print(f"  With memory:    {summary['with_hit']}/{summary['total']} ({with_rate:.1f}%)")
    print(f"  Without memory: {summary['without_hit']}/{summary['total']} ({without_rate:.1f}%)")
    print(f"  Improvement:    +{with_rate - without_rate:.1f}pp")
    if summary["errors"]:
        print(f"  Errors: {summary['errors']}")
    print(f"  📄 {path}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ollama A/B experiment (English, n=150)")
    ap.add_argument("--model", required=True, help="Ollama model name, e.g. gemma3:4b")
    args = ap.parse_args()
    run(args.model)
