#!/usr/bin/env python3
"""
阶段 2 — 跨模型记忆一致性验证

核心问题: 用户在模型 A 里说的话, 模型 B/C 能不能记得?

三模型:
  - gemini-flash-lite-latest  (云端)
  - gemma3:4b                 (本地 ollama)
  - qwen2.5:3b                (本地 ollama)

剧本: 用户先在一个模型里陈述立场, 然后换到其他模型提问,
      看回答是否体现了之前的记忆(通过 Noesis 代理注入 system prompt)。
"""
import subprocess
import sys
import os
import time
import signal
import sqlite3
import httpx

# ── 配置 ────────────────────────────────────────────────────────────────────────
GEMINI_KEY = "YOUR_GEMINI_API_KEY_HERE"
PROXY_URL = "http://127.0.0.1:8080/v1/chat/completions"
NOESIS_DIR = "/Users/mac27ssd/Noesis"
USER_ID = "phase2_cross"

MODELS = {
    "gemini": "gemini-flash-lite-latest",
    "gemma3": "gemma3:4b",
    "qwen":   "qwen2.5:3b",
}

# ── 剧本 ────────────────────────────────────────────────────────────────────────
# 每回合: (用哪个模型发, user 说什么, 期望回答里包含的关键词, 评分说明)
SCRIPT = [
    {
        "round": 1,
        "model_key": "gemma3",
        "user": "我叫张伟, 是一名后端工程师。我决定用 sqlite-vec 做向量检索方案, 嵌入式的方案比 FAISS 更适合我的轻量级场景。",
        "expect": None,  # 第1回合只是陈述, 无需验证回答
        "note": "陈述立场 → 应被存为 position/event, 达到 provisional",
    },
    {
        "round": 2,
        "model_key": "qwen",
        "user": "我之前跟你说过在用哪个向量检索库吗? 提醒我一下。",
        "expect": "sqlite-vec",
        "note": "换模型 B 查询 → 应从注入的记忆中答出 sqlite-vec",
    },
    {
        "round": 3,
        "model_key": "gemini",
        "user": "帮我推荐一个本地向量检索方案, 要轻量级的。",
        "expect": "sqlite",
        "note": "换模型 C → 回答应体现 sqlite-vec 偏好(记忆注入)",
    },
    {
        "round": 4,
        "model_key": "gemma3",
        "user": "我对向量库的选型态度是什么?",
        "expect": "sqlite",
        "note": "回到模型 A → 应延续立场 + 跨工具重复应提升 confidence",
    },
]


# ── 辅助 ────────────────────────────────────────────────────────────────────────
def send_via_proxy(model: str, user_msg: str, auth_key: str = "dummy") -> tuple[int, str]:
    """通过 Noesis 代理发送请求, 返回 (http_status, assistant_reply)。

    auth_key: 对 Gemini 必须传真实 key; 对 ollama 本地模型任意值即可。
    """
    try:
        r = httpx.post(
            PROXY_URL,
            headers={
                "Authorization": f"Bearer {auth_key}",
                "X-User-ID": USER_ID,
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": user_msg}],
                "max_tokens": 300,
                "stream": False,
            },
            timeout=180,  # gemma3:4b 冷启动需时间, 放宽到 3 分钟
        )
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            return 200, content.strip()
        return r.status_code, r.text[:200]
    except Exception as e:
        return -1, str(e)[:200]


def count_items(db_path):
    if not os.path.exists(db_path):
        return 0
    db = sqlite3.connect(db_path)
    try:
        return db.execute("SELECT COUNT(*) FROM items WHERE user_id=?", (USER_ID,)).fetchone()[0]
    except Exception:
        return 0
    finally:
        db.close()


def get_items_detail(db_path):
    if not os.path.exists(db_path):
        return []
    db = sqlite3.connect(db_path)
    try:
        return db.execute(
            "SELECT type, status, confidence, source_tool, substr(text,1,60) "
            "FROM items WHERE user_id=? ORDER BY created_at", (USER_ID,)
        ).fetchall()
    finally:
        db.close()


# ── 主流程 ──────────────────────────────────────────────────────────────────────
print("=" * 72)
print("阶段 2: 跨模型记忆一致性验证")
print(f"  用户: {USER_ID}")
print(f"  模型: {list(MODELS.values())}")
print("=" * 72)
print()

# 清空该用户的历史数据, 保证干净实验
HOT_DB = os.path.expanduser("~/.noesis/hot.db")
db = sqlite3.connect(HOT_DB)
db.execute("DELETE FROM items WHERE user_id=?", (USER_ID,))
db.commit(); db.close()
print("[准备] 已清空 phase2_cross 用户的历史数据")
print()

# 启动代理
print("[启动] 启动 Noesis 代理...")
env = os.environ.copy()
env["GEMINI_API_KEY"] = GEMINI_KEY
proc = subprocess.Popen(["noesis", "start"], stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, cwd=NOESIS_DIR, env=env)
time.sleep(4)
if proc.poll() is not None:
    print(f"❌ 代理启动失败:\n{proc.stdout.read()[:800]}")
    sys.exit(1)
print("  ✅ 代理已启动")
print()

results = []
for turn in SCRIPT:
    model_key = turn["model_key"]
    model = MODELS[model_key]
    # 本地模型用 dummy key, 云端 Gemini 用真实 key
    auth_key = GEMINI_KEY if model_key == "gemini" else "dummy"
    print(f"┌─ 回合 {turn['round']} [{model_key}] model={model}")
    print(f"│  USER: {turn['user']}")
    print(f"│  说明: {turn['note']}")

    status, reply = send_via_proxy(model, turn["user"], auth_key=auth_key)
    print(f"│  HTTP: {status}")

    if status != 200:
        print(f"│  ❌ 请求失败: {reply[:150]}")
        results.append({**turn, "pass": False, "reply": reply, "reason": "请求失败"})
        print(f"└─")
        print()
        time.sleep(2)
        continue

    # 截断显示
    show = reply[:200] + ("..." if len(reply) > 200 else "")
    print(f"│  ASSISTANT: {show}")

    # 等后台抽取
    time.sleep(2.5)

    # 评分
    if turn["expect"]:
        passed = turn["expect"].lower() in reply.lower()
        mark = "✅ 命中" if passed else "❌ 未命中"
        print(f"│  期望关键词: {turn['expect']!r} → {mark}")
        results.append({**turn, "pass": passed, "reply": reply})
    else:
        print(f"│  (陈述回合, 无需关键词验证)")
        results.append({**turn, "pass": True, "reply": reply})

    print(f"└─")
    print()

# 收集记忆状态
print("=" * 72)
print("最终记忆状态:")
print("=" * 72)
for row in get_items_detail(HOT_DB):
    print(f"  type={row[0]:10s} status={row[1]:10s} conf={row[2]:.3f} tool={row[3]:12s} text={row[4]}")
print()

# 关闭代理
proc.send_signal(signal.SIGINT)
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill(); proc.wait()

# 汇总
print("=" * 72)
print("阶段 2 结果汇总")
print("=" * 72)
scored = [r for r in results if r.get("expect")]
passed = sum(1 for r in scored if r["pass"])
total = len(scored)
for r in results:
    if r.get("expect"):
        mark = "✅" if r["pass"] else "❌"
        print(f"  {mark} 回合{r['round']} [{r['model_key']:7s}] 期望'{r['expect']}' → {'命中' if r['pass'] else '未命中'}")
    else:
        print(f"  ⚪ 回合{r['round']} [{r['model_key']:7s}] 陈述回合 (已存储)")
print()
print(f"关键词命中率: {passed}/{total}")
if passed == total:
    print("✅ 跨模型记忆一致性: 全部通过 — 不同模型能访问同一记忆")
elif passed > 0:
    print("🟡 部分通过 — 某些模型能访问记忆, 某些不能")
else:
    print("❌ 失败 — 记忆未跨模型注入")
