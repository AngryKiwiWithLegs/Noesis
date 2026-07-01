#!/usr/bin/env python3
"""
实验: 有记忆(Noesis) vs 无记忆(直连) 端到端回答质量对比
==========================================================
核心问题: Noesis 的记忆注入让 AI 回答变好了多少?

设计:
  阶段1 建记忆: 每个用户画像通过 Noesis 代理说 N 句话, 建立记忆库
  阶段2 测试:   对每个用户问 M 个问题
    A组 (实验): 经 Noesis 代理 (有记忆注入)
    B组 (对照): 直连 LLM (无记忆)
  阶段3 评分:   关键词命中 (期望回答是否包含已知记忆的关键信息)

模型: Gemini (gemini-flash-lite-latest) + gemma3:4b (本地)

用法:
    GEMINI_API_KEY="你的key" python3 ab_comparison.py
"""
import os
import sys
import json
import time
import httpx
import sqlite3
import textwrap
from datetime import datetime
from pathlib import Path

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
PROXY_URL  = "http://127.0.0.1:8080/v1/chat/completions"
RESULTS_DIR = Path("/Users/mac27ssd/ZCodeProject/noesis_experiment/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 直连端点
DIRECT = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", GEMINI_KEY),
    "gemma3": ("http://localhost:11434/v1/chat/completions", "dummy"),
}
MODELS = {
    "gemini": "gemini-flash-lite-latest",
}
# 本地大模型可选 (CPU 慢, 默认注释; 取消注释启用)
# "gemma3": "gemma3:4b",

# ── 用户画像: 建记忆用的话 + 测试问题(及期望关键词) ────────────────────────────
# 每个画像: name, 建记忆的陈述句, 测试问答(问题+期望关键词)
PROFILES = [
    {
        "name": "zhang_wei",
        "build": [
            "我叫张伟, 是一名有8年经验的后端工程师",
            "我偏好用 PostgreSQL 而不是 MySQL",
            "我决定用 sqlite-vec 做向量检索, 嵌入式方案比 FAISS 更轻量",
            "我偏好用 Python 写后端, 觉得比 Java 更简洁",
            "我认为微服务架构比单体更适合快速迭代",
        ],
        "questions": [
            {"q": "帮我推荐一个数据库",                       "expect": "postgres"},
            {"q": "我之前用什么做向量检索的?",                "expect": "sqlite"},
            {"q": "我对 Python 和 Java 的态度?",             "expect": "python"},
            {"q": "我对微服务的看法?",                       "expect": "微服务"},
            {"q": "我是什么职位的?",                         "expect": "后端"},
        ],
    },
    {
        "name": "li_lei",
        "build": [
            "我叫李雷, 是一名数据科学家, 擅长机器学习",
            "我偏好用 pandas 做数据处理, 觉得比 Excel 强",
            "我决定用 Rust 学习系统编程, 觉得它比 C++ 更安全",
            "我喜欢用 Obsidian 做笔记, 觉得比 Notion 灵活",
            "我倾向于用 Docker 做开发环境, 不喜欢本地装依赖",
        ],
        "questions": [
            {"q": "我平时用什么处理数据?",                   "expect": "pandas"},
            {"q": "我在学什么编程语言?",                     "expect": "rust"},
            {"q": "我用什么做笔记?",                         "expect": "obsidian"},
            {"q": "我对本地装依赖的态度?",                   "expect": "docker"},
            {"q": "我的职业是什么?",                         "expect": "数据"},
        ],
    },
    {
        "name": "han_meimei",
        "name_zh": "韩梅梅",
        "build": [
            "我叫韩梅梅, 是一名前端工程师, 精通 React",
            "我偏好用 TypeScript 而不是 JavaScript",
            "我决定用 Tailwind CSS 写样式, 觉得比手写 CSS 高效",
            "我认为 GraphQL 比 REST 更适合复杂前端",
            "我喜欢用 Vim 快捷键, 觉得比鼠标高效",
        ],
        "questions": [
            {"q": "我擅长什么前端框架?",                     "expect": "react"},
            {"q": "我用什么写样式?",                         "expect": "tailwind"},
            {"q": "我对 TypeScript 的偏好?",                 "expect": "typescript"},
            {"q": "我对 GraphQL 和 REST 的看法?",            "expect": "graphql"},
            {"q": "我是什么工程师?",                         "expect": "前端"},
        ],
    },
]

# ── 通信函数 ────────────────────────────────────────────────────────────────────
def ask_proxy(model_key: str, user_id: str, question: str) -> tuple[bool, str]:
    """A组: 经 Noesis 代理 (有记忆注入)。"""
    model = MODELS[model_key]
    key = GEMINI_KEY if model_key == "gemini" else "dummy"
    try:
        r = httpx.post(PROXY_URL, headers={
            "Authorization": f"Bearer {key}",
            "X-User-ID": user_id,
            "Content-Type": "application/json",
        }, json={"model": model, "messages": [{"role":"user","content":question}],
                "max_tokens": 150, "stream": False}, timeout=120)
        if r.status_code == 200:
            return True, r.json()["choices"][0]["message"]["content"].strip()
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, f"ERR: {str(e)[:80]}"

def ask_direct(model_key: str, question: str) -> tuple[bool, str]:
    """B组: 直连 LLM (无记忆)。"""
    url, key = DIRECT[model_key]
    model = MODELS[model_key]
    try:
        r = httpx.post(url, headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }, json={"model": model, "messages": [{"role":"user","content":question}],
                "max_tokens": 150, "stream": False}, timeout=120)
        if r.status_code == 200:
            return True, r.json()["choices"][0]["message"]["content"].strip()
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, f"ERR: {str(e)[:80]}"

def build_memory(user_id: str, statements: list[str], model_key: str):
    """阶段1: 通过 Noesis 代理让用户说话, 建立记忆。"""
    for stmt in statements:
        ask_proxy(model_key, user_id, stmt)
        time.sleep(1.5)  # 避免速率限制 + 等 pipeline

def clear_user(user_id: str):
    """清空某用户的记忆, 保证干净实验。"""
    db = sqlite3.connect(os.path.expanduser("~/.noesis/hot.db"))
    db.execute("DELETE FROM items WHERE user_id=?", (user_id,))
    db.commit(); db.close()

def count_user_memory(user_id: str) -> int:
    db = sqlite3.connect(os.path.expanduser("~/.noesis/hot.db"))
    n = db.execute("SELECT COUNT(*) FROM items WHERE user_id=?", (user_id,)).fetchone()[0]
    db.close(); db = sqlite3.connect(os.path.expanduser("~/.noesis/hot.db")); db.close()
    return n

# ── 主流程 ──────────────────────────────────────────────────────────────────────
def run():
    print("=" * 72)
    print("  实验: 有记忆(Noesis) vs 无记忆(直连) 对比")
    print("=" * 72)
    print(f"  模型: {list(MODELS.values())}")
    print(f"  画像: {[p['name'] for p in PROFILES]}")
    print(f"  每画像: {len(PROFILES[0]['questions'])} 题 × 2 组 = {len(PROFILES[0]['questions'])*2} 次/模型")
    print("=" * 72)

    all_results = []
    summary = {}  # {model_key: {with_hit, without_hit, total}}

    for profile in PROFILES:
        uid = profile["name"]
        print(f"\n{'─'*72}")
        print(f"  画像: {uid}")
        print(f"{'─'*72}")

        # 清空 + 建记忆 (用 gemini 建, 记忆是模型无关的)
        clear_user(uid)
        print(f"  建记忆 ({len(profile['build'])} 句)...", end=" ", flush=True)
        build_memory(uid, profile["build"], "gemini")
        time.sleep(3)  # 等 pipeline 处理
        n = count_user_memory(uid)
        print(f"完成, 记忆库 {n} 条")

        # 对每个模型测试
        for model_key in MODELS:
            print(f"\n  ▸ 模型: {MODELS[model_key]}")
            for q_info in profile["questions"]:
                q, expect = q_info["q"], q_info["expect"]

                # A组: 有记忆 (代理)
                ok_a, ans_a = ask_proxy(model_key, uid, q)
                hit_a = expect.lower() in ans_a.lower() if ok_a else False
                time.sleep(1.2)

                # B组: 无记忆 (直连)
                ok_b, ans_b = ask_direct(model_key, q)
                hit_b = expect.lower() in ans_b.lower() if ok_b else False
                time.sleep(1.2)

                # 记录
                rec = {
                    "profile": uid, "model": model_key, "question": q,
                    "expect": expect,
                    "with_mem_hit": hit_a, "without_mem_hit": hit_b,
                    "with_mem_ans": ans_a[:150], "without_mem_ans": ans_b[:150],
                }
                all_results.append(rec)

                mark_a = "✅" if hit_a else "❌"
                mark_b = "✅" if hit_b else "❌"
                diff = "🟢有效" if (hit_a and not hit_b) else ("🟡相同" if hit_a==hit_b else "🔴异常")
                print(f"    [{diff}] Q: {q[:24]:26s} 有记忆{mark_a} 无记忆{mark_b} (期望'{expect}')")

                # 汇总
                key = model_key
                if key not in summary:
                    summary[key] = {"with_hit": 0, "without_hit": 0, "total": 0}
                summary[key]["total"] += 1
                if hit_a: summary[key]["with_hit"] += 1
                if hit_b: summary[key]["without_hit"] += 1

    # ── 最终汇总 ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  最终汇总")
    print(f"{'='*72}\n")
    print(f"  {'模型':28s} {'有记忆命中':12s} {'无记忆命中':12s} {'提升':8s}")
    print(f"  {'-'*60}")
    for mk, s in summary.items():
        wh = s["with_hit"]; woh = s["without_hit"]; tot = s["total"]
        lift = wh - woh
        print(f"  {MODELS[mk]:28s} {wh}/{tot:<10d} {woh}/{tot:<10d} +{lift}")
    print()

    # 保存报告
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "experiment": "ab_memory_vs_nomemory",
        "timestamp": ts,
        "models": MODELS,
        "summary": summary,
        "details": all_results,
    }
    json_path = RESULTS_DIR / f"ab_comparison_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  📄 完整数据: {json_path}")

if __name__ == "__main__":
    if not GEMINI_KEY:
        print("错误: 请设置 GEMINI_API_KEY 环境变量")
        sys.exit(1)
    run()
