#!/usr/bin/env python3
"""
ollama 本地模型 A/B 实验 (有记忆 vs 无记忆)
============================================
与 Gemini 实验完全相同的设计, 只是模型换成 ollama 本地模型。

优化:
  - 记忆已在之前 Gemini 实验建好 (zhang_wei/li_lei/han_meimei), 复用
  - 短超时 (60s)、小 max_tokens (100) 适配 CPU 推理
  - 失败重试 + 单模型运行 + 结果即时落盘 (防超时丢失)

用法:
    python3 ab_ollama.py --model gemma3:4b
    python3 ab_ollama.py --model qwen2.5:3b
"""
import os, sys, json, time, argparse, sqlite3
import httpx
from datetime import datetime
from pathlib import Path

PROXY_URL = "http://127.0.0.1:8080/v1/chat/completions"
OLLAMA_DIRECT = "http://localhost:11434/v1/chat/completions"
RESULTS_DIR = Path("/Users/mac27ssd/ZCodeProject/noesis_experiment/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 与 Gemini 版完全相同的画像和问题 ────────────────────────────────────────────
PROFILES = [
    {"name": "zhang_wei", "questions": [
        {"q": "帮我推荐一个数据库",                "expect": "postgres"},
        {"q": "我之前用什么做向量检索的?",         "expect": "sqlite"},
        {"q": "我对 Python 和 Java 的态度?",       "expect": "python"},
        {"q": "我对微服务的看法?",                 "expect": "微服务"},
        {"q": "我是什么职位的?",                   "expect": "后端"},
    ]},
    {"name": "li_lei", "questions": [
        {"q": "我平时用什么处理数据?",             "expect": "pandas"},
        {"q": "我在学什么编程语言?",               "expect": "rust"},
        {"q": "我用什么做笔记?",                   "expect": "obsidian"},
        {"q": "我对本地装依赖的态度?",             "expect": "docker"},
        {"q": "我的职业是什么?",                   "expect": "数据"},
    ]},
    {"name": "han_meimei", "questions": [
        {"q": "我擅长什么前端框架?",               "expect": "react"},
        {"q": "我用什么写样式?",                   "expect": "tailwind"},
        {"q": "我对 TypeScript 的偏好?",           "expect": "typescript"},
        {"q": "我对 GraphQL 和 REST 的看法?",      "expect": "graphql"},
        {"q": "我是什么工程师?",                   "expect": "前端"},
    ]},
]

def ask_proxy(model, user_id, question, retries=2):
    """A组: 经 Noesis 代理 (有记忆)。"""
    for attempt in range(retries + 1):
        try:
            r = httpx.post(PROXY_URL, headers={
                "Authorization": "Bearer dummy",
                "X-User-ID": user_id, "Content-Type": "application/json",
            }, json={"model": model, "messages": [{"role":"user","content":question}],
                    "max_tokens": 100, "stream": False}, timeout=90)
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
    """B组: 直连 ollama (无记忆)。"""
    for attempt in range(retries + 1):
        try:
            r = httpx.post(OLLAMA_DIRECT, headers={"Content-Type": "application/json"},
                json={"model": model, "messages": [{"role":"user","content":question}],
                      "max_tokens": 100, "stream": False}, timeout=90)
            if r.status_code == 200:
                return True, r.json()["choices"][0]["message"]["content"].strip()
            if attempt < retries: time.sleep(2); continue
            return False, f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            if attempt < retries: time.sleep(2); continue
            return False, f"ERR: {str(e)[:60]}"
    return False, "retries exhausted"

def run(model):
    print(f"\n{'='*64}")
    print(f"  ollama A/B 实验: {model}")
    print(f"{'='*64}")

    # 确认记忆还在
    db = sqlite3.connect(os.path.expanduser("~/.noesis/hot.db"))
    for uid in [p["name"] for p in PROFILES]:
        n = db.execute("SELECT COUNT(*) FROM items WHERE user_id=?", (uid,)).fetchone()[0]
        if n == 0:
            print(f"  ⚠️ {uid} 无记忆! 需先跑 Gemini 版建记忆")
            return None
    db.close()
    print("  记忆库就绪 ✓")

    results = []
    summary = {"with_hit": 0, "without_hit": 0, "total": 0}

    for profile in PROFILES:
        uid = profile["name"]
        print(f"\n  画像: {uid}")
        for q_info in profile["questions"]:
            q, expect = q_info["q"], q_info["expect"]

            ok_a, ans_a = ask_proxy(model, uid, q)
            hit_a = expect.lower() in ans_a.lower() if ok_a else False

            ok_b, ans_b = ask_direct(model, q)
            hit_b = expect.lower() in ans_b.lower() if ok_b else False

            rec = {"profile": uid, "model": model, "question": q, "expect": expect,
                   "with_mem_hit": hit_a, "without_mem_hit": hit_b,
                   "with_mem_ans": ans_a[:120], "without_mem_ans": ans_b[:120]}
            results.append(rec)

            summary["total"] += 1
            if hit_a: summary["with_hit"] += 1
            if hit_b: summary["without_hit"] += 1

            ma = "✅" if hit_a else "❌"
            mb = "✅" if hit_b else "❌"
            diff = "🟢" if (hit_a and not hit_b) else ("🟡" if hit_a==hit_b else "🔴")
            print(f"    [{diff}] {q[:22]:24s} 有记忆{ma} 无记忆{mb} ('{expect}')")

    # 落盘
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"experiment": "ab_ollama", "model": model, "timestamp": ts,
              "summary": summary, "details": results}
    path = RESULTS_DIR / f"ab_ollama_{model.replace(':','_')}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  {'─'*50}")
    print(f"  汇总: 有记忆 {summary['with_hit']}/{summary['total']} | "
          f"无记忆 {summary['without_hit']}/{summary['total']} | "
          f"提升 +{summary['with_hit']-summary['without_hit']}")
    print(f"  📄 {path}")
    return summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ollama 模型名, 如 gemma3:4b")
    args = ap.parse_args()
    run(args.model)
