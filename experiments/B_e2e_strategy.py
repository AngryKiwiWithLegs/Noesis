#!/usr/bin/env python3
"""
B-e2e: 端到端检索策略对比
==========================
核心问题: 不同检索策略注入 system prompt, 最终 LLM 回答质量差多少?

同一问题、同一用户、同一 LLM (Gemini), 四种检索策略:
  - Noesis:   经代理, 走完整检索 + 置信度门控
  - 朴素RAG:  取该用户全部非tentative记忆塞 system
  - 最近窗口: 只取最近5条塞 system
  - 无记忆:   空 system (对照)

评分: 关键词命中 (与之前 A/B 实验一致)
"""
import os, sys, json, time, sqlite3
import httpx
from datetime import datetime
from pathlib import Path

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MODEL = "gemini-flash-lite-latest"
HOT_DB = os.path.expanduser("~/.noesis/hot.db")
RESULTS_DIR = Path("/Users/mac27ssd/ZCodeProject/noesis_experiment/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 复用 A/B 实验的画像和问题 ────────────────────────────────────────────────────
PROFILES = [
    {"name": "zhang_wei", "questions": [
        {"q": "帮我推荐一个数据库",            "expect": "postgres"},
        {"q": "我之前用什么做向量检索的?",     "expect": "sqlite"},
        {"q": "我对 Python 和 Java 的态度?",   "expect": "python"},
        {"q": "我对微服务的看法?",             "expect": "微服务"},
        {"q": "我是什么职位的?",               "expect": "后端"},
    ]},
    {"name": "li_lei", "questions": [
        {"q": "我平时用什么处理数据?",         "expect": "pandas"},
        {"q": "我在学什么编程语言?",           "expect": "rust"},
        {"q": "我用什么做笔记?",               "expect": "obsidian"},
        {"q": "我对本地装依赖的态度?",         "expect": "docker"},
        {"q": "我的职业是什么?",               "expect": "数据"},
    ]},
    {"name": "han_meimei", "questions": [
        {"q": "我擅长什么前端框架?",           "expect": "react"},
        {"q": "我用什么写样式?",               "expect": "tailwind"},
        {"q": "我对 TypeScript 的偏好?",       "expect": "typescript"},
        {"q": "我对 GraphQL 和 REST 的看法?",  "expect": "graphql"},
        {"q": "我是什么工程师?",               "expect": "前端"},
    ]},
]

# ── 检索策略: 构造 system prompt ─────────────────────────────────────────────────
def get_all_memories(uid):
    """取用户全部非tentative记忆。"""
    db = sqlite3.connect(HOT_DB)
    rows = db.execute(
        "SELECT text, type, status FROM items WHERE user_id=? "
        "AND status IN ('provisional','settled')", (uid,)
    ).fetchall()
    db.close()
    return [{"text": r[0], "type": r[1], "status": r[2]} for r in rows]

def get_recent_memories(uid, n=5):
    """取最近n条非tentative记忆。"""
    db = sqlite3.connect(HOT_DB)
    rows = db.execute(
        "SELECT text, type, status FROM items WHERE user_id=? "
        "AND status IN ('provisional','settled') "
        "ORDER BY created_at DESC LIMIT ?", (uid, n)
    ).fetchall()
    db.close()
    return [{"text": r[0], "type": r[1], "status": r[2]} for r in rows]

def fmt_system(memories):
    """把记忆列表格式化成 system prompt。"""
    if not memories:
        return ""
    lines = [f"- [{m['type']}·{m['status']}] {m['text']}" for m in memories]
    return "以下是关于用户的已知信息：\n" + "\n".join(lines)

# ── 四种策略 ────────────────────────────────────────────────────────────────────
def strategy_noesis(uid, question):
    """Noesis: 经代理 (走完整检索+门控)。"""
    for attempt in range(4):
        try:
            r = httpx.post("http://127.0.0.1:8080/v1/chat/completions", headers={
                "Authorization": f"Bearer {GEMINI_KEY}",
                "X-User-ID": uid, "Content-Type": "application/json",
            }, json={"model": MODEL, "messages": [{"role":"user","content":question}],
                    "max_tokens": 150, "stream": False}, timeout=60)
            if r.status_code == 200:
                return True, r.json()["choices"][0]["message"]["content"].strip()
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))  # 退避: 15/30/45s
                continue
            return False, f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            if attempt < 3: time.sleep(5); continue
            return False, f"ERR: {str(e)[:60]}"
    return False, "429 retries exhausted"

def strategy_custom(uid, question, sys_prompt):
    """朴素RAG/最近窗口/无记忆: 自定义system, 直连Gemini。"""
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": question})
    for attempt in range(4):
        try:
            r = httpx.post(GEMINI_URL, headers={
                "Authorization": f"Bearer {GEMINI_KEY}", "Content-Type": "application/json",
            }, json={"model": MODEL, "messages": messages, "max_tokens": 150, "stream": False},
               timeout=60)
            if r.status_code == 200:
                return True, r.json()["choices"][0]["message"]["content"].strip()
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            return False, f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            if attempt < 3: time.sleep(5); continue
            return False, f"ERR: {str(e)[:60]}"
    return False, "429 retries exhausted"

# ── 主流程 ──────────────────────────────────────────────────────────────────────
def run():
    print("=" * 72)
    print("  B-e2e: 端到端检索策略对比 (Gemini)")
    print("=" * 72)
    print(f"  画像: {[p['name'] for p in PROFILES]}")
    print(f"  策略: Noesis / 朴素RAG / 最近窗口 / 无记忆")
    print(f"  每画像 {len(PROFILES[0]['questions'])} 题 × 4 策略 = 20 次/画像")
    print("=" * 72)

    STRATEGIES = ["Noesis", "朴素RAG", "最近窗口", "无记忆"]
    summary = {s: {"hit": 0, "total": 0} for s in STRATEGIES}
    all_results = []

    for profile in PROFILES:
        uid = profile["name"]
        print(f"\n{'─'*72}")
        print(f"  画像: {uid}")
        # 预取记忆 (朴素RAG和最近窗口共用)
        all_mems = get_all_memories(uid)
        recent_mems = get_recent_memories(uid, 5)
        sys_rag = fmt_system(all_mems)
        sys_window = fmt_system(recent_mems)
        print(f"  记忆: 全部 {len(all_mems)} 条 | 最近 {len(recent_mems)} 条")

        for q_info in profile["questions"]:
            q, expect = q_info["q"], q_info["expect"]
            row = {"profile": uid, "question": q, "expect": expect}
            print(f"\n    Q: {q}  (期望: {expect})")

            for strat in STRATEGIES:
                if strat == "Noesis":
                    ok, ans = strategy_noesis(uid, q)
                elif strat == "朴素RAG":
                    ok, ans = strategy_custom(uid, q, sys_rag)
                elif strat == "最近窗口":
                    ok, ans = strategy_custom(uid, q, sys_window)
                else:  # 无记忆
                    ok, ans = strategy_custom(uid, q, "")

                hit = expect.lower() in ans.lower() if ok else False
                row[f"{strat}_hit"] = hit
                row[f"{strat}_ans"] = ans[:100] if ok else ans
                summary[strat]["total"] += 1
                if hit:
                    summary[strat]["hit"] += 1
                mark = "✅" if hit else "❌"
                print(f"      {strat:8s} {mark}  {ans[:60]}")
                time.sleep(3)  # 避免速率限制 (429)

            all_results.append(row)

    # 汇总
    print(f"\n{'='*72}")
    print("  最终汇总")
    print(f"{'='*72}\n")
    print(f"  {'策略':12s} {'命中':10s} {'命中率':8s}")
    print(f"  {'-'*34}")
    for s in STRATEGIES:
        h, t = summary[s]["hit"], summary[s]["total"]
        print(f"  {s:12s} {h}/{t:<8d} {100*h/t:.0f}%")

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"experiment": "B_e2e_strategy_comparison", "model": MODEL,
              "timestamp": ts, "summary": summary, "details": all_results}
    path = RESULTS_DIR / f"B_e2e_strategy_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 {path}")

if __name__ == "__main__":
    if not GEMINI_KEY:
        print("错误: 请设置 GEMINI_API_KEY"); sys.exit(1)
    run()
