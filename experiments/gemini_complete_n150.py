#!/usr/bin/env python3
"""
Complete the Gemini A/B experiment to n=150.

The original run (ab_comparison_en_150_20260630.json) was rate-limited at n=111,
completing prof_1 through prof_23. This script runs the remaining prof_24-30
(35 questions) and merges with the existing 111 to produce a full n≈146 result.

Memories for prof_24-30 are already in the DB (built by the local model setup).
"""
import os, sys, json, time, sqlite3
import httpx
from datetime import datetime
from pathlib import Path

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
PROXY_URL = "http://127.0.0.1:8080/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MODEL = "gemini-flash-lite-latest"
RESULTS_DIR = Path("/Users/mac27ssd/Noesis/experiments/results")
EXISTING_FILE = RESULTS_DIR / "ab_comparison_en_150_20260630.json"

sys.path.insert(0, str(Path(__file__).parent))
from ab_comparison_en import PROFILES


def ask_proxy(user_id, question):
    for attempt in range(4):
        try:
            r = httpx.post(PROXY_URL, headers={
                "Authorization": f"Bearer {GEMINI_KEY}",
                "X-User-ID": user_id, "Content-Type": "application/json",
            }, json={"model": MODEL, "messages": [{"role": "user", "content": question}],
                     "max_tokens": 150, "stream": False}, timeout=60)
            if r.status_code == 200:
                return True, r.json()["choices"][0]["message"]["content"].strip()
            if r.status_code == 429:
                print(f"      429 rate limited, waiting {12*(attempt+1)}s...", flush=True)
                time.sleep(12 * (attempt + 1)); continue
            return False, f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            if attempt < 3: time.sleep(5); continue
            return False, f"ERR: {str(e)[:60]}"
    return False, "429 retries exhausted"


def ask_direct(question):
    for attempt in range(4):
        try:
            r = httpx.post(GEMINI_URL, headers={
                "Authorization": f"Bearer {GEMINI_KEY}", "Content-Type": "application/json",
            }, json={"model": MODEL, "messages": [{"role": "user", "content": question}],
                     "max_tokens": 150, "stream": False}, timeout=60)
            if r.status_code == 200:
                return True, r.json()["choices"][0]["message"]["content"].strip()
            if r.status_code == 429:
                print(f"      429 rate limited, waiting {12*(attempt+1)}s...", flush=True)
                time.sleep(12 * (attempt + 1)); continue
            return False, f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            if attempt < 3: time.sleep(5); continue
            return False, f"ERR: {str(e)[:60]}"
    return False, "429 retries exhausted"


def build_memory(user_id, statements):
    """Build memories for a profile via proxy (same as original experiment)."""
    for stmt in statements:
        ask_proxy(user_id, stmt)
        time.sleep(1.5)


def main():
    # Load existing results
    existing = json.loads(EXISTING_FILE.read_text())
    existing_details = existing.get("details", existing.get("results", []))
    completed_profiles = {d["profile"] for d in existing_details}
    print(f"Existing results: {len(existing_details)} questions, {len(completed_profiles)} profiles")

    # Find missing profiles
    missing_profiles = [p for p in PROFILES if p["name"] not in completed_profiles]
    print(f"Missing profiles: {[p['name'] for p in missing_profiles]}")
    print(f"Questions to run: {sum(len(p['questions']) for p in missing_profiles)}")
    print()

    # BUILD memories for missing profiles through Gemini (matching original methodology)
    for p in missing_profiles:
        uid = p["name"]
        print(f"  Building memory for {uid} ({len(p['build'])} stmts via Gemini)...", flush=True)
        build_memory(uid, p["build"])
        time.sleep(3)
        db = sqlite3.connect(os.path.expanduser("~/.noesis/hot.db"))
        n = db.execute("SELECT COUNT(*) FROM items WHERE user_id=?", (uid,)).fetchone()[0]
        db.close()
        print(f"    {uid}: {n} memories built ✓")

    # Run missing questions
    new_details = []
    q_num = 0
    total_missing = sum(len(p["questions"]) for p in missing_profiles)

    for profile in missing_profiles:
        uid = profile["name"]
        for q_info in profile["questions"]:
            q_num += 1
            q, expect = q_info["q"], q_info["expect"]

            ok_a, ans_a = ask_proxy(uid, q)
            hit_a = expect.lower() in ans_a.lower() if ok_a else False
            time.sleep(2)

            ok_b, ans_b = ask_direct(q)
            hit_b = expect.lower() in ans_b.lower() if ok_b else False
            time.sleep(2)

            new_details.append({
                "profile": uid, "question": q, "expect": expect,
                "with_mem_hit": hit_a, "without_mem_hit": hit_b,
                "with_mem_ans": ans_a[:100] if ok_a else ans_a,
                "without_mem_ans": ans_b[:100] if ok_b else ans_b,
            })

            symbol = "🟢" if (hit_a and not hit_b) else ("🟡" if hit_a == hit_b else "🔴")
            print(f"  [{q_num}/{total_missing}] {symbol} {uid:10s} | "
                  f"with={'✅' if hit_a else '❌'} without={'✅' if hit_b else '❌'} ('{expect}')",
                  flush=True)

    # Merge
    all_details = existing_details + new_details
    total = len(all_details)
    with_hits = sum(1 for d in all_details if d.get("with_mem_hit"))
    without_hits = sum(1 for d in all_details if d.get("without_mem_hit"))

    # McNemar
    b = sum(1 for d in all_details if d.get("with_mem_hit") and not d.get("without_mem_hit"))
    c = sum(1 for d in all_details if not d.get("with_mem_hit") and d.get("without_mem_hit"))
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0

    # Save merged result
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged = {
        "experiment": "ab_comparison_en_complete",
        "model": MODEL,
        "timestamp": ts,
        "num_questions": total,
        "summary": {
            "with_hit": with_hits, "without_hit": without_hits,
            "total": total,
        },
        "discordant": {"b": b, "c": c, "chi2": chi2},
        "details": all_details,
        "note": f"Merged from n={len(existing_details)} (original) + n={len(new_details)} (completion with Gemini-built memories)",
    }
    path = RESULTS_DIR / f"ab_comparison_en_n{total}_{ts}.json"
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))

    print(f"\n{'─' * 60}")
    print(f"  MERGED RESULT (Gemini Flash, n={total})")
    print(f"  With memory:    {with_hits}/{total} ({with_hits/total*100:.1f}%)")
    print(f"  Without memory: {without_hits}/{total} ({without_hits/total*100:.1f}%)")
    print(f"  Improvement:    +{(with_hits-without_hits)/total*100:.1f}pp")
    print(f"  McNemar χ²={chi2:.2f}  (b={b}, c={c})")
    print(f"  📄 {path}")


if __name__ == "__main__":
    main()
