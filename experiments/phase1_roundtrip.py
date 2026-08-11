#!/usr/bin/env python3
"""
阶段 1.2 — Noesis 代理完整回路测试 (Gemini 单模型)

回路: 代理启动 → 通过代理发请求 → Gemini 返回 → 检查 hot.db 新增节点 → recall 检索

设计:
  - 在子进程中启动 noesis start
  - 用 httpx 向 localhost:8080 发请求 (模型 gemini-flash-lite-latest)
  - 带自定义测试话语, 方便后续检索验证
  - 等待后台抽取完成后, 检查 hot.db 和 recall 结果
  - 全部结果打印为结构化报告
"""
import subprocess
import sys
import os
import time
import tempfile
import sqlite3
import httpx
import signal

# ── 配置 ────────────────────────────────────────────────────────────────────────

GEMINI_KEY = "YOUR_GEMINI_API_KEY_HERE"
PROXY_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "gemini-flash-lite-latest"
USER_ID = "phase1_test"

# 测试话语: 包含明确的立场和身份信息, 方便后续检索验证
TEST_MESSAGES = [
    {"role": "user", "content": "我叫张伟, 是一名后端工程师, 最近在评估 sqlite-vec 做向量检索方案, 觉得嵌入式的方案非常适合我的场景"}
]

# Recall 测试查询
RECALL_QUERY = "向量检索方案"

# ── 辅助 ────────────────────────────────────────────────────────────────────────

def count_nodes(db_path):
    """统计 hot.db items 表中的节点数。"""
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM items").fetchone()
        return rows[0] if rows else 0
    except Exception:
        return -1
    finally:
        conn.close()

def get_status(db_path):
    """尝试获取 memory 状态。"""
    if not os.path.exists(db_path):
        return None
    try:
        from noesis.memory.main import Memory
        # 直接查数据库
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT status, COUNT(*) FROM items GROUP BY status").fetchall()
        conn.close()
        return dict(rows) if rows else {}
    except Exception as e:
        return {"error": str(e)}

# ── 主流程 ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("阶段 1.2: Noesis 代理完整回路测试")
print(f"  模型: {MODEL}")
print(f"  用户: {USER_ID}")
print("=" * 70)
print()

# Step 1: 记录启动前的节点数
HOT_DB = os.path.expanduser("~/.noesis/hot.db")
before_count = count_nodes(HOT_DB)
print(f"[Step 1] 启动前 hot.db 节点数: {before_count}")
print()

# Step 2: 启动 Noesis 代理 (后台)
print(f"[Step 2] 启动 Noesis 代理...")
# 用 venv 的 python 来确保环境一致
PYTHON = "/Users/mac27ssd/Noesis/.venv/bin/python"
NOESIS_DIR = "/Users/mac27ssd/Noesis"

env = os.environ.copy()
env["GEMINI_API_KEY"] = GEMINI_KEY
# 不设 OPENAI_API_KEY → 抽取器会回退 MockExtractor (这是预期的)

proc = subprocess.Popen(
    ["noesis", "start"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=NOESIS_DIR,
    env=env,
)

# 等代理就绪
time.sleep(4)
if proc.poll() is not None:
    out = proc.stdout.read()[:1500]
    print(f"❌ 代理启动失败 (exit {proc.returncode})")
    print(out)
    sys.exit(1)

print("  ✅ 代理已启动 (PID: %d)" % proc.pid)

# 读启动日志
startup_lines = []
# proc.stdout 是阻塞的, 先不等
print()

# Step 3: 通过代理发请求
print(f"[Step 3] 通过代理发送请求 (model={MODEL})...")
try:
    r = httpx.post(
        PROXY_URL,
        headers={
            "Authorization": f"Bearer {GEMINI_KEY}",
            "X-User-ID": USER_ID,
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": TEST_MESSAGES,
            "max_tokens": 200,
            "stream": False,
        },
        timeout=60,
    )
    proxy_status = "✅ 通" if r.status_code == 200 else f"❌ HTTP {r.status_code}"
    print(f"  代理返回: {proxy_status}")
    if r.status_code == 200:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        print(f"  Gemini 回答: {content[:150]}")
    else:
        print(f"  错误: {r.text[:300]}")
except Exception as e:
    print(f"  ❌ 请求异常: {e}")
    proxy_status = f"❌ 异常: {e}"

print()

# Step 4: 等待后台抽取完成
print("[Step 4] 等待后台抽取 pipeline 处理 (8 秒)...")
time.sleep(8)

# Step 5: 检查 hot.db 节点变化
after_count = count_nodes(HOT_DB)
new_nodes = after_count - before_count if before_count >= 0 and after_count >= 0 else "?"
print(f"  启动前: {before_count} → 现在: {after_count} (新增: {new_nodes})")
store_status = "✅ 有新增" if isinstance(new_nodes, int) and new_nodes > 0 else "⚠️ 无新增"
print(f"  {store_status}")

# 查看节点状态分布
statuses = get_status(HOT_DB)
if statuses:
    print(f"  状态分布: {statuses}")

print()

# Step 6: 用 recall 检索
print(f"[Step 6] Recall 测试 (query='{RECALL_QUERY}', user={USER_ID})...")
try:
    env2 = os.environ.copy()
    env2["GEMINI_API_KEY"] = GEMINI_KEY
    recall_proc = subprocess.run(
        [PYTHON, "-c", f"""
import sys; sys.path.insert(0, '{NOESIS_DIR}')
import os; os.environ['GEMINI_API_KEY'] = '{GEMINI_KEY}'
from noesis.memory.main import Memory
m = Memory.from_config_file(os.path.expanduser('~/.noesis/config.yaml'))
ctx = m.build_context('{RECALL_QUERY}', user_id='{USER_ID}')
if ctx:
    print(ctx)
else:
    print('(空 — 无相关记忆)')
"""],
        capture_output=True, text=True, timeout=30, env=env2,
    )
    recall_output = recall_proc.stdout.strip()
    if recall_output:
        print(f"  ✅ 检索到记忆:")
        for line in recall_output.split('\n'):
            print(f"    {line}")
    else:
        print(f"  ⚠️ Recall 返回空 (节点可能还在 tentative)")
        if recall_proc.stderr:
            err = recall_proc.stderr.strip().split('\n')[-1]
            if 'WARNING' not in err and 'MockExtractor' not in err:
                print(f"    stderr: {err[:120]}")
except Exception as e:
    print(f"  ❌ Recall 异常: {e}")

print()

# Step 7: 清理 — 关闭代理
print("[Step 7] 关闭代理...")
proc.send_signal(signal.SIGINT)
try:
    proc.wait(timeout=5)
    print("  ✅ 代理已关闭")
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
    print("  ⚠️ 代理被强制终止")

print()
print("=" * 70)
print("阶段 1.2 完成")
print("=" * 70)
