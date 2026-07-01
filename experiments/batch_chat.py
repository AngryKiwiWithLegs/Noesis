#!/usr/bin/env python3
"""
批量对话生成器 — 通过 Noesis API 代理与 Gemini 进行 50 轮真实对话。

每轮: 用户说一句有明确立场的话 → Gemini 真实回复 → Noesis 自动抽取存储。
记忆真实进入 hot.db, 刷新星图即可看到节点增长。

用法:
    GEMINI_API_KEY="你的key" python3 batch_chat.py
"""
import os
import sys
import time
import httpx

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
PROXY_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "gemini-flash-lite-latest"
USER_ID = "default"

if not GEMINI_KEY:
    print("错误: 请设置 GEMINI_API_KEY 环境变量")
    sys.exit(1)

# ── 50 句对话, 覆盖多种类型 ────────────────────────────────────────────────────
# 设计原则: 每句包含明确的身份/偏好/观点/事件/事实, 让 Noesis 能分类
CONVERSATIONS = [
    # 身份 identity (青)
    "我叫张伟, 是一名有 8 年经验的后端工程师",
    "我目前住在杭州, 在一家电商公司工作",
    "我的技术栈主要是 Python 和 Go, 偶尔写点 Rust",
    "我是计算机科学硕士毕业, 擅长分布式系统",
    "我在团队里负责基础设施和数据库架构",
    "我是 Linux 重度用户, 日常用 macOS 开发",
    # 偏好 preference (品红)
    "我偏好用 Python 做数据分析, 觉得 pandas 比 Excel 强得多",
    "我倾向于用 PostgreSQL 而不是 MySQL, 因为它的 JSON 支持更好",
    "我喜欢用 Docker Compose 做本地开发环境编排",
    "我偏好用 Vim 快捷键, 觉得比鼠标高效",
    "我决定用 sqlite-vec 做嵌入式向量检索, 比 FAISS 轻量",
    "我更愿意用 Tailwind CSS 写样式, 而不是手写 CSS",
    "我倾向选择强类型语言, 比如 TypeScript 而不是 JavaScript",
    "我喜欢用 Git 的 rebase 工作流, 保持提交历史整洁",
    # 观点 position (紫)
    "我认为微服务架构比单体更适合快速迭代的小团队",
    "我觉得 Rust 的内存安全机制让它优于 C++ 做系统编程",
    "我认为过早优化是万恶之源, 应该先写对再写快",
    "我觉得 code review 比 unit test 更能保证代码质量",
    "我认为关系型数据库在大多数业务场景下足够用了",
    "我觉得 GraphQL 比 REST 更适合复杂的前端需求",
    "我认为 Kafka 在高吞吐场景下比 RabbitMQ 更合适",
    "我觉得 serverless 不适合长连接和重计算任务",
    "我认为测试驱动开发能显著减少后期 bug",
    "我觉得 monorepo 比 multi-repo 更适合小团队",
    # 事件 event (金)
    "上周我把项目的数据库从 MySQL 迁移到了 PostgreSQL",
    "我决定这个季度开始学习 Rust 语言",
    "上个月我们团队上线了一个新的支付系统",
    "我最近在研究向量数据库, 打算用在推荐系统里",
    "昨天我修复了一个困扰团队两周的并发 bug",
    "我上周参加了 QCon 技术大会, 学到了很多",
    "我刚刚把博客从 WordPress 迁移到了 Hugo",
    # 事实 fact (绿)
    "我们团队有 5 个后端工程师和 2 个前端工程师",
    "公司的主要技术栈是 Python 后端加 React 前端",
    "我们的生产环境跑在 AWS 的 us-east-1 区域",
    "系统的日活用户大约是 50 万",
    "我们的代码库有超过 100 万行代码",
    "团队的 CI 流水线平均跑 12 分钟",
    # 提问/探索 (橙)
    "你觉得对于初创公司, 用 Kubernetes 是不是过度设计了?",
    "我想了解一下, 向量数据库选 Milvus 还是 Qdrant 更好?",
    "你能推荐几个适合后端工程师学习的系统设计书吗?",
    "我想知道, Redis 的持久化用 RDB 还是 AOF 更好?",
    # 更多身份和偏好 (丰富星图)
    "我是开源爱好者, 在 GitHub 上贡献过十几个项目",
    "我习惯早上 9 点开始工作, 晚上专注写代码",
    "我偏好用 Obsidian 做笔记, 觉得比 Notion 更灵活",
    "我用 Mechanical Keyboard 写代码, 觉得手感很重要",
    "我决定今年读完设计数据密集型应用这本书",
    "我倾向于用 pytest 而不是 unittest 做测试",
    "我觉得 PostgreSQL 的全文搜索比 Elasticsearch 更适合中小项目",
    "我认为 async/await 比 callback 更容易维护",
    "我偏好用 Click 而不是 argparse 写命令行工具",
    "我最近在学机器学习, 打算在推荐系统里实践",
    "我认为良好的文档比注释更重要",
]

def chat(user_msg: str, idx: int, log: list) -> bool:
    """发送一轮对话, 返回是否成功。完整对话存入 log。"""
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
                "messages": [{"role": "user", "content": user_msg}],
                "max_tokens": 200,
                "stream": False,
            },
            timeout=60,
        )
        if r.status_code == 200:
            reply = r.json()["choices"][0]["message"]["content"].strip()
            preview = reply[:50] + "..." if len(reply) > 50 else reply
            print(f"  [{idx:2d}] ✅ {user_msg[:32]:34s} → {preview}")
            # 保存完整对话到日志
            log.append({
                "index": idx,
                "user": user_msg,
                "assistant": reply,
                "status": "ok",
            })
            return True
        else:
            print(f"  [{idx:2d}] ❌ HTTP {r.status_code}: {r.text[:80]}")
            log.append({
                "index": idx,
                "user": user_msg,
                "assistant": "",
                "status": f"error_{r.status_code}",
                "error": r.text[:200],
            })
            return False
    except Exception as e:
        print(f"  [{idx:2d}] ❌ {type(e).__name__}: {str(e)[:60]}")
        log.append({
            "index": idx,
            "user": user_msg,
            "assistant": "",
            "status": "exception",
            "error": str(e)[:200],
        })
        return False

def count_memory() -> int:
    import sqlite3
    db = sqlite3.connect(os.path.expanduser("~/.noesis/hot.db"))
    try:
        return db.execute("SELECT COUNT(*) FROM items WHERE user_id=?", (USER_ID,)).fetchone()[0]
    finally:
        db.close()

# ── 主流程 ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  批量对话生成器 — 50 轮真实 Gemini 对话经 Noesis 代理")
print("=" * 70)
print(f"  模型: {MODEL}")
print(f"  用户: {USER_ID}")
print(f"  对话数: {len(CONVERSATIONS)}")
print(f"  开始前记忆数: {count_memory()}")
print("=" * 70)
print()

success = 0
failed = 0
conversation_log = []  # 保存完整对话
for i, msg in enumerate(CONVERSATIONS, 1):
    ok = chat(msg, i, conversation_log)
    if ok:
        success += 1
    else:
        failed += 1
    # 每轮间隔 1.5 秒, 避免触发速率限制, 也给后台 pipeline 处理时间
    time.sleep(1.5)

# 等待最后的后台抽取完成
print()
print("等待后台抽取 pipeline 处理最后几条...")
time.sleep(5)

# ── 保存完整对话日志 ────────────────────────────────────────────────────────────
import json
from datetime import datetime

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# JSON 完整版
json_path = os.path.join(LOG_DIR, f"conversations_{ts}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "model": MODEL,
        "user_id": USER_ID,
        "timestamp": ts,
        "total": len(conversation_log),
        "success": success,
        "conversations": conversation_log,
    }, f, ensure_ascii=False, indent=2)

# HTML 可读版 (像聊天界面)
html_path = os.path.join(LOG_DIR, f"conversations_{ts}.html")
html_rows = ""
for c in conversation_log:
    status_badge = "✅" if c["status"] == "ok" else f"❌ {c['status']}"
    user_bubble = c["user"].replace("<", "&lt;").replace(">", "&gt;")
    if c["status"] == "ok":
        ai_bubble = c["assistant"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        ai_html = f'<div class="msg ai"><span class="role">Gemini</span><div class="bubble">{ai_bubble}</div></div>'
    else:
        ai_html = f'<div class="msg ai error"><span class="role">Gemini</span><div class="bubble">{c.get("error","无回复")}</div></div>'
    html_rows += f'''
    <div class="turn">
      <div class="turn-no">{c["index"]}</div>
      <div class="msg user"><span class="role">你</span><div class="bubble">{user_bubble}</div></div>
      {ai_html}
    </div>'''

with open(html_path, "w", encoding="utf-8") as f:
    f.write(f'''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<title>Noesis 对话记录 {ts}</title>
<style>
  body {{ font-family: -apple-system,"PingFang SC",sans-serif; background:#f5f5f7; margin:0; padding:20px; }}
  h1 {{ color:#1d1d1f; font-size:22px; }}
  .summary {{ color:#86868b; font-size:13px; margin-bottom:24px; }}
  .turn {{ background:#fff; border-radius:16px; padding:16px 20px; margin-bottom:12px; max-width:800px; box-shadow:0 1px 3px rgba(0,0,0,0.06); display:flex; gap:12px; align-items:flex-start; }}
  .turn-no {{ color:#c7c7cc; font-size:13px; min-width:24px; padding-top:4px; }}
  .msg {{ flex:1; }}
  .msg + .msg {{ margin-top:8px; }}
  .role {{ font-size:11px; color:#86868b; font-weight:600; display:block; margin-bottom:3px; }}
  .bubble {{ font-size:15px; line-height:1.55; color:#1d1d1f; }}
  .msg.user .bubble {{ color:#0066cc; }}
  .msg.ai .bubble {{ color:#1d1d1f; }}
  .msg.ai.error .bubble {{ color:#ff3b30; font-size:13px; }}
</style></head><body>
<h1>🧠 Noesis 对话记录</h1>
<div class="summary">模型: {MODEL} · 用户: {USER_ID} · 时间: {ts} · 共 {len(conversation_log)} 轮 (成功 {success})</div>
{html_rows}
</body></html>''')

print()
print("=" * 70)
print("  完成!")
print("=" * 70)
print(f"  成功发送: {success}/{len(CONVERSATIONS)}")
print(f"  失败:     {failed}")
print(f"  当前记忆数: {count_memory()}")
print()
print(f"  📄 完整对话日志已保存:")
print(f"     HTML: {html_path}")
print(f"     JSON: {json_path}")
print()
print("  → 刷新星图页面 (Cmd+R) 查看所有节点")
print("     http://localhost:8787")
print("=" * 70)
