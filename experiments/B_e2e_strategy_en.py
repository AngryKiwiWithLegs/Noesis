#!/usr/bin/env python3
"""
English B-e2e: 50-question end-to-end retrieval-strategy comparison
====================================================================
Same 10 profiles × 5 questions as the EN A/B run. Reuses built memories.
Four strategies, each injects a different system prompt to Gemini:
  - Noesis:        via proxy (full retrieval + confidence gating, ~1200 token budget)
  - naive-RAG:     all non-tentative memories of the user, raw
  - recent-window: last 5 non-tentative memories
  - no-memory:     empty system prompt (control)

Scoring: keyword hit (objective, reproducible).
Usage: GEMINI_API_KEY="..." python3 B_e2e_strategy_en.py
"""
import os, sys, json, time, sqlite3
import httpx
from datetime import datetime
from pathlib import Path

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
PROXY_URL  = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "gemini-flash-lite-latest"
HOT_DB = os.path.expanduser("~/.noesis/hot.db")
RESULTS_DIR = Path("/Users/mac27ssd/ZCodeProject/noesis_experiment/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROFILES = [
    {"name":"prof_1","questions":[
        {"q":"Recommend a database for me","expect":"postgres"},
        {"q":"What do I use for vector search?","expect":"sqlite"},
        {"q":"How do I feel about Python vs Java?","expect":"python"},
        {"q":"What is my view on microservices?","expect":"microservice"},
        {"q":"What is my job title?","expect":"backend"}]},
    {"name":"prof_2","questions":[
        {"q":"What do I use to process data?","expect":"pandas"},
        {"q":"What programming language am I learning?","expect":"rust"},
        {"q":"What do I use for notes?","expect":"obsidian"},
        {"q":"How do I feel about local dependency installs?","expect":"docker"},
        {"q":"What is my profession?","expect":"data"}]},
    {"name":"prof_3","questions":[
        {"q":"What frontend framework am I expert in?","expect":"react"},
        {"q":"What do I use for styling?","expect":"tailwind"},
        {"q":"What is my preference on TypeScript?","expect":"typescript"},
        {"q":"What is my view on GraphQL vs REST?","expect":"graphql"},
        {"q":"What kind of engineer am I?","expect":"frontend"}]},
    {"name":"prof_4","questions":[
        {"q":"What is my monitoring tool of choice?","expect":"grafana"},
        {"q":"What do I use for infrastructure?","expect":"terraform"},
        {"q":"What is my view on ArgoCD vs Flux?","expect":"argocd"},
        {"q":"What do I use for package management?","expect":"helm"},
        {"q":"What kind of engineer am I?","expect":"devops"}]},
    {"name":"prof_5","questions":[
        {"q":"What language do I prefer for iOS?","expect":"swift"},
        {"q":"What UI framework do I use for new projects?","expect":"swiftui"},
        {"q":"What is my view on Combine vs RxSwift?","expect":"combine"},
        {"q":"What do I use for mobile CI/CD?","expect":"fastlane"},
        {"q":"What kind of developer am I?","expect":"mobile"}]},
    {"name":"prof_6","questions":[
        {"q":"What deep learning framework do I prefer?","expect":"pytorch"},
        {"q":"What do I use for model deployment?","expect":"hugging"},
        {"q":"What is my view on LoRA vs full fine-tuning?","expect":"lora"},
        {"q":"What do I use for experiment tracking?","expect":"weights"},
        {"q":"What kind of engineer am I?","expect":"ml"}]},
    {"name":"prof_7","questions":[
        {"q":"What game engine do I use?","expect":"unity"},
        {"q":"What language do I prefer for scripting?","expect":"c#"},
        {"q":"What do I use for 3D modeling?","expect":"blender"},
        {"q":"What is my view on game architecture?","expect":"ecs"},
        {"q":"What kind of developer am I?","expect":"game"}]},
    {"name":"prof_8","questions":[
        {"q":"What endpoint tool do I prefer?","expect":"sentinelone"},
        {"q":"What do I use for secrets management?","expect":"vault"},
        {"q":"What is my security philosophy?","expect":"zero-trust"},
        {"q":"What do I use for runtime threat detection?","expect":"falco"},
        {"q":"What kind of engineer am I?","expect":"security"}]},
    {"name":"prof_9","questions":[
        {"q":"What language do I use for smart contracts?","expect":"solidity"},
        {"q":"What development tool do I prefer?","expect":"hardhat"},
        {"q":"What is my view on Ethereum scaling?","expect":"rollup"},
        {"q":"What do I use for data indexing?","expect":"graph"},
        {"q":"What kind of developer am I?","expect":"blockchain"}]},
    {"name":"prof_10","questions":[
        {"q":"What orchestration tool do I prefer?","expect":"airflow"},
        {"q":"What do I use for transformations?","expect":"dbt"},
        {"q":"What is my view on Snowflake vs BigQuery?","expect":"snowflake"},
        {"q":"What do I use for streaming?","expect":"kafka"},
        {"q":"What kind of engineer am I?","expect":"data"}]},
]

def get_all_memories(uid):
    db = sqlite3.connect(HOT_DB)
    rows = db.execute("SELECT text,type,status FROM items WHERE user_id=? AND status IN ('provisional','settled')",(uid,)).fetchall()
    db.close()
    return [{"text":r[0],"type":r[1],"status":r[2]} for r in rows]

def get_recent_memories(uid, n=5):
    db = sqlite3.connect(HOT_DB)
    rows = db.execute("SELECT text,type,status FROM items WHERE user_id=? AND status IN ('provisional','settled') ORDER BY created_at DESC LIMIT ?",(uid,n)).fetchall()
    db.close()
    return [{"text":r[0],"type":r[1],"status":r[2]} for r in rows]

def fmt_system(memories):
    if not memories: return ""
    lines = [f"- [{m['type']}·{m['status']}] {m['text']}" for m in memories]
    return "Here is what is known about the user:\n" + "\n".join(lines)

def call(url, headers, payload):
    for attempt in range(4):
        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return True, r.json()["choices"][0]["message"]["content"].strip()
            if r.status_code == 429:
                time.sleep(12 * (attempt + 1)); continue
            return False, f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            if attempt < 3: time.sleep(5); continue
            return False, f"ERR: {str(e)[:60]}"
    return False, "429 retries exhausted"

def main():
    print("="*72)
    print("  English B-e2e: retrieval-strategy comparison (50 questions)")
    print("="*72)
    print(f"  Profiles: 10, Questions: 50, Strategies: 4 (200 LLM calls)")
    print("="*72)

    STRATEGIES = ["Noesis", "naive-RAG", "recent-window", "no-memory"]
    summary = {s: {"hit":0,"total":0} for s in STRATEGIES}
    all_results = []

    for pi, profile in enumerate(PROFILES, 1):
        uid = profile["name"]
        print(f"\n[{pi}/10] {uid}")
        all_m = get_all_memories(uid); recent_m = get_recent_memories(uid, 5)
        sys_rag = fmt_system(all_m); sys_win = fmt_system(recent_m)
        print(f"  memories: all={len(all_m)} recent={len(recent_m)}")

        for q_info in profile["questions"]:
            q, expect = q_info["q"], q_info["expect"]
            row = {"profile":uid,"question":q,"expect":expect}
            for strat in STRATEGIES:
                if strat == "Noesis":
                    ok, ans = call(PROXY_URL, {"Authorization":f"Bearer {GEMINI_KEY}",
                        "X-User-ID":uid,"Content-Type":"application/json"},
                        {"model":MODEL,"messages":[{"role":"user","content":q}],"max_tokens":150,"stream":False})
                elif strat == "naive-RAG":
                    msgs = ([{"role":"system","content":sys_rag}] if sys_rag else []) + [{"role":"user","content":q}]
                    ok, ans = call(GEMINI_URL, {"Authorization":f"Bearer {GEMINI_KEY}","Content-Type":"application/json"},
                        {"model":MODEL,"messages":msgs,"max_tokens":150,"stream":False})
                elif strat == "recent-window":
                    msgs = ([{"role":"system","content":sys_win}] if sys_win else []) + [{"role":"user","content":q}]
                    ok, ans = call(GEMINI_URL, {"Authorization":f"Bearer {GEMINI_KEY}","Content-Type":"application/json"},
                        {"model":MODEL,"messages":msgs,"max_tokens":150,"stream":False})
                else:
                    ok, ans = call(GEMINI_URL, {"Authorization":f"Bearer {GEMINI_KEY}","Content-Type":"application/json"},
                        {"model":MODEL,"messages":[{"role":"user","content":q}],"max_tokens":150,"stream":False})
                hit = expect.lower() in ans.lower() if ok else False
                row[f"{strat}_hit"] = hit
                row[f"{strat}_ans"] = ans[:100] if ok else ans
                summary[strat]["total"] += 1
                if hit: summary[strat]["hit"] += 1
                time.sleep(2.5)
            all_results.append(row)
            print(f"    {q[:38]:40s} N{'✅' if row['Noesis_hit'] else '❌'} R{'✅' if row['naive-RAG_hit'] else '❌'} W{'✅' if row['recent-window_hit'] else '❌'} 0{'✅' if row['no-memory_hit'] else '❌'} ('{expect}')")

    print(f"\n{'='*72}\n  FINAL SUMMARY\n{'='*72}\n")
    print(f"  {'Strategy':16s} {'Hit':10s} {'Rate':8s}")
    print(f"  {'-'*36}")
    for s in STRATEGIES:
        h,t = summary[s]["hit"], summary[s]["total"]
        print(f"  {s:16s} {h}/{t:<8d} {100*h/t:.0f}%")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"B_e2e_strategy_en_{ts}.json"
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"experiment":"B_e2e_strategy_en","language":"en","model":MODEL,"timestamp":ts,
                   "summary":summary,"details":all_results}, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {path}")

if __name__ == "__main__":
    if not GEMINI_KEY:
        print("Error: set GEMINI_API_KEY"); sys.exit(1)
    main()
