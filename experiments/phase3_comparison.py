#!/usr/bin/env python3
"""
阶段 3 — 有无记忆效果对比

同一组问题, 分别在「通过 Noesis 代理(有记忆注入)」和「直连模型(无记忆)」下问,
对比回答是否体现用户历史偏好。

使用 Gemini (云端) 和 gemma3 (本地) 两个模型做对比。
"""
import subprocess
import sys
import os
import time
import signal
import httpx

# ── 配置 ────────────────────────────────────────────────────────────────────────
GEMINI_KEY = "YOUR_GEMINI_API_KEY_HERE"
PROXY_URL = "http://127.0.0.1:8080/v1/chat/completions"
NOESIS_DIR = "/Users/mac27ssd/Noesis"
USER_ID = "phase3_compare"

# 直连端点
DIRECT_ENDPOINTS = {
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key": GEMINI_KEY,
    },
    "gemma3": {
        "url": "http://localhost:11434/v1/chat/completions",
        "key": "dummy",
    },
}

# 测试问题: 和阶段 2 不同, 模拟一个全新用户
QUESTIONS = [
    {
        "q": "我叫李明, 我偏好用 Rust 写高性能服务, 觉得它比 Go 更适合计算密集型场景。",
        "expect": None,  # 建立记忆
        "phase": "build",
    },
    {
        "q": "帮我推荐一个高性能后端语言。",
        "expect": "rust",
        "phase": "test",
        "note": "有记忆时应推荐 Rust; 无记忆时可能推荐 Go/Java/Python",
    },
    {
        "q": "我对 Go 语言和 Rust 语言的看法分别是什么?",
        "expect": "rust",
        "phase": "test",
        "note": "有记忆时应体现 Rust 偏好; 无记忆时会泛泛而谈",
    },
]


def ask_proxy(model: str, msg: str, auth_key: str) -> tuple[int, str]:
    """通过 Noesis 代理问 (有记忆注入)。"""
    r = httpx.post(PROXY_URL, headers={
        "Authorization": f"Bearer {auth_key}",
        "X-User-ID": USER_ID,
        "Content-Type": "application/json",
    }, json={"model": model, "messages": [{"role": "user", "content": msg}],
            "max_tokens": 300, "stream": False}, timeout=120)
    if r.status_code == 200:
        return 200, r.json()["choices"][0]["message"]["content"].strip()
    return r.status_code, r.text[:200]


def ask_direct(model_key: str, model: str, msg: str) -> tuple[int, str]:
    """直连模型 (无记忆注入)。"""
    ep = DIRECT_ENDPOINTS[model_key]
    r = httpx.post(ep["url"], headers={
        "Authorization": f"Bearer {ep['key']}",
        "Content-Type": "application/json",
    }, json={"model": model, "messages": [{"role": "user", "content": msg}],
            "max_tokens": 300, "stream": False}, timeout=120)
    if r.status_code == 200:
        return 200, r.json()["choices"][0]["message"]["content"].strip()
    return r.status_code, r.text[:200]


# ── 主流程 ──────────────────────────────────────────────────────────────────────
print("=" * 72)
print("阶段 3: 有无记忆效果对比")
print("=" * 72)
print()

# 清空历史
import sqlite3
HOT_DB = os.path.expanduser("~/.noesis/hot.db")
db = sqlite3.connect(HOT_DB)
db.execute("DELETE FROM items WHERE user_id=?", (USER_ID,))
db.commit(); db.close()

# 启动代理
print("[启动] Noesis 代理...")
env = os.environ.copy()
env["GEMINI_API_KEY"] = GEMINI_KEY
proc = subprocess.Popen(["noesis", "start"], stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, cwd=NOESIS_DIR, env=env)
time.sleep(4)
if proc.poll() is not None:
    print(f"❌ 启动失败: {proc.stdout.read()[:500]}")
    sys.exit(1)
print("  ✅ 代理已启动\n")

# Phase 1: 建立记忆 (通过代理)
build_q = QUESTIONS[0]
model = "gemma3:4b"
auth_key = "dummy"
print(f"{'='*72}")
print("Phase 1: 建立记忆 (通过代理)")
print(f"{'='*72}")
print(f"  USER: {build_q['q']}")
s, r = ask_proxy(model, build_q['q'], auth_key)
print(f"  ASSISTANT: {r[:150]}...")
print()
time.sleep(3)  # 等 pipeline 处理

# 查看记忆状态
db = sqlite3.connect(HOT_DB)
items = db.execute("SELECT type,status,confidence,substr(text,1,60) FROM items WHERE user_id=?",
                    (USER_ID,)).fetchall()
db.close()
print("  记忆状态:")
for it in items:
    print(f"    type={it[0]:10s} status={it[1]:10s} conf={it[2]:.3f} | {it[3]}")
print()

# Phase 2: 对比测试
print(f"{'='*72}")
print("Phase 2: 有无记忆对比")
print(f"{'='*72}\n")

results = []
for q_info in QUESTIONS[1:]:
    q = q_info["q"]
    expect = q_info["expect"]
    note = q_info["note"]

    print(f"问题: {q}")
    print(f"  ({note})\n")

    # 有记忆 (通过代理)
    model_proxy = "gemini-flash-lite-latest"
    auth = GEMINI_KEY
    s1, r1 = ask_proxy(model_proxy, q, auth)
    has_memory = expect.lower() in r1.lower()

    # 无记忆 (直连)
    model_direct = "gemini-flash-lite-latest"
    s2, r2 = ask_direct("gemini", model_direct, q)
    no_memory = expect.lower() in r2.lower()

    print(f"  [有记忆] 代理→{model_proxy}:")
    print(f"    {r1[:200]}")
    hit_w = f"✅ 包含 {expect!r}" if has_memory else f"❌ 不包含 {expect!r}"
    hit_wo = f"✅ 包含 {expect!r}" if no_memory else f"❌ 不包含 {expect!r}"
    print(f"    → {hit_w}\n")

    print(f"  [无记忆] 直连→{model_direct}:")
    print(f"    {r2[:200]}")
    print(f"    → {hit_wo}\n")

    results.append({
        "question": q, "expect": expect,
        "with_memory": has_memory, "without_memory": no_memory,
    })
    print(f"  {'─'*60}\n")

# 关闭代理
proc.send_signal(signal.SIGINT)
try: proc.wait(timeout=5)
except: proc.kill(); proc.wait()

# 汇总
print(f"{'='*72}")
print("阶段 3 结果汇总")
print(f"{'='*72}")
print(f"{'问题':30s} {'有记忆':8s} {'无记忆':8s} {'差异':8s}")
print(f"{'─'*56}")
for r in results:
    w = "✅" if r["with_memory"] else "❌"
    wo = "✅" if r["without_memory"] else "❌"
    diff = "🟢 有效" if r["with_memory"] and not r["without_memory"] else (
        "🟡 相同" if r["with_memory"] == r["without_memory"] else "⚪")
    print(f"{r['question'][:28]:30s} {w:8s} {wo:8s} {diff:8s}")
print()

with_hit = sum(1 for r in results if r["with_memory"])
wo_hit = sum(1 for r in results if r["without_memory"])
print(f"有记忆命中率: {with_hit}/{len(results)}")
print(f"无记忆命中率: {wo_hit}/{len(results)}")
