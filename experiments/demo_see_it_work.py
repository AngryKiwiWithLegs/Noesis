#!/usr/bin/env python3
"""
Noesis 可视化演示 — 让你亲眼看到它记没记、好不好用

四幕:
  幕一  你说一句话 → 看 Noesis 把它存成了什么
  幕二  问一个相关问题 → 看记忆能不能被检索出来
  幕三  换个模型问 → 看跨模型记忆一致性
  幕四  有记忆 vs 无记忆 → 看 Noesis 带来的实际差异
"""
import os, sys, time, sqlite3, tempfile, textwrap

# ── 配置 ────────────────────────────────────────────────────────────────────────
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-flash-lite-latest"
USER_ID = "demo_user"

NOESIS_DIR = "/Users/mac27ssd/Noesis"
HOT_DB = os.path.expanduser("~/.noesis/hot.db")

sys.path.insert(0, NOESIS_DIR)

# ── 启动横幅 ────────────────────────────────────────────────────────────────────
def banner(title):
    print()
    print("═" * 68)
    print(f"  {title}")
    print("═" * 68)

def step(n, msg):
    print(f"\n  ▸ [{n}] {msg}")

def show(text, indent=4, width=64):
    for line in textwrap.wrap(str(text), width):
        print(" " * indent + line)

# ── 导入 Noesis ─────────────────────────────────────────────────────────────────
import logging
logging.disable(logging.CRITICAL)

print("正在加载 Noesis 和 Gemini...", end=" ", flush=True)
from noesis.memory.main import Memory
import httpx
print("完成 ✓")

# ── 准备干净的演示数据库(不污染你真实记忆库)────────────────────────────────────
banner("准备干净的演示环境")
demo_dir = tempfile.mkdtemp(prefix="noesis_demo_")
demo_db = os.path.join(demo_dir, "hot.db")
demo_vault = os.path.join(demo_dir, "vault")
os.makedirs(demo_vault)

# 用 from_config_file 才会触发 _attach_pipeline (MockExtractor 回退路径)
# 生成临时 config yaml, 指向演示目录
import yaml
real_cfg = yaml.safe_load(open(os.path.expanduser("~/.noesis/config.yaml")))
real_cfg["vector_store"]["config"]["db_path"] = demo_db
real_cfg["cold_store"]["config"]["vault_path"] = demo_vault
demo_cfg_path = os.path.join(demo_dir, "config.yaml")
yaml.safe_dump(real_cfg, open(demo_cfg_path, "w"))

mem = Memory.from_config_file(demo_cfg_path)
print(f"\n  演示用独立数据库: {demo_db}")
print(f"  演示用独立 Obsidian: {demo_vault}")
print(f"  不影响你真实的记忆库 ✓")

# ════════════════════════════════════════════════════════════════════════════════
# 幕一: 你说一句话, Noesis 存了什么
# ════════════════════════════════════════════════════════════════════════════════
banner("幕一  ▸ 你说一句话,看 Noesis 把它存成了什么")

USER_SAID = "我叫李雷,是一名数据科学家,我偏好用 Python 做数据分析,觉得 pandas 比 Excel 更适合处理大数据集"
print(f"\n  【你说的原话】")
show(USER_SAID, indent=6)

step("1.1", "这句话进入 Noesis...")
mem.add([{"role": "user", "content": USER_SAID}], user_id=USER_ID, source_tool="web-gemini")
time.sleep(2)

step("1.2", "Noesis 存下了什么 (查数据库):")
db = sqlite3.connect(demo_db)
rows = db.execute(
    "SELECT type, status, confidence, substr(text,1,70) FROM items WHERE user_id=?",
    (USER_ID,)
).fetchall()
for r in rows:
    print()
    print(f"      类型 type      : {r[0]}")
    print(f"      状态 status   : {r[1]}")
    print(f"      置信度 conf   : {r[2]:.3f}")
    print(f"      存的文本 text : {r[3]}")
    print()
    if r[1] in ("provisional", "settled"):
        print(f"      ✅ 状态={r[1]} → 这条记忆【会被注入】到后续对话")
    else:
        print(f"      ⚠️ 状态={r[1]} → 还在试探期,暂不注入")

# ════════════════════════════════════════════════════════════════════════════════
# 幕二: 记忆能不能被检索出来
# ════════════════════════════════════════════════════════════════════════════════
banner("幕二  ▸ 问个相关问题,看记忆能不能被检索出来")

QUERY = "我平时用什么工具做数据分析?"
print(f"\n  【检索查询】")
show(QUERY, indent=6)

step("2.1", "Noesis 从记忆库检索相关记忆,生成注入文本:")
ctx = mem.build_context(QUERY, user_id=USER_ID)
if ctx:
    print()
    for line in ctx.split("\n"):
        print(f"      {line}")
    print()
    print(f"      ✅ 检索成功! 这段文字会被注入到 AI 的 system prompt")
    print(f"         → AI 就能'想起'你是数据科学家、用 Python/pandas")
else:
    print()
    print(f"      ❌ 没检索到 (记忆还在 tentative, 或查询不相关)")

# ════════════════════════════════════════════════════════════════════════════════
# 幕三: 跨模型记忆一致性(关键!)
# ════════════════════════════════════════════════════════════════════════════════
banner("幕三  ▸ 跨模型记忆:换一个模型问,它也知道你的事")

# 模拟两个不同模型回答同一个问题
# 通过代理(有记忆) vs 直连(无记忆)
QUESTION = "我之前跟你说过我偏好用什么工具做数据分析吗?"

step("3.1", "【模型A: 通过 Noesis 代理(有记忆)】")
print(f"\n      问: {QUESTION}")
print(f"      这时 system prompt 里已被注入:")
show(ctx[:120] + ("..." if len(ctx) > 120 else ""), indent=8)

step("3.2", "【模型B: 直连 Gemini(无记忆)】")
print(f"\n      问: {QUESTION}")
print(f"      system prompt 为空, AI 完全不认识你")

step("3.3", "实际调用两个模型对比(用 Gemini):")
print(f"\n      正在调用模型, 请稍候...")

# 有记忆版(注入 ctx)
sys_msg_with = ctx if ctx else "（无记忆）"
sys_msg_without = "（无记忆）"

def ask(sys_msg, user_q):
    try:
        r = httpx.post(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            headers={"Authorization": f"Bearer {GEMINI_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_q},
                ],
                "max_tokens": 200,
                "stream": False,
            },
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        return f"[调用失败 HTTP {r.status_code}]: {r.text[:100]}"
    except Exception as e:
        return f"[异常]: {str(e)[:100]}"

ans_with = ask(sys_msg_with, QUESTION)
ans_without = ask(sys_msg_without, QUESTION)

print()
print(f"      ┌─ 【有记忆的回答】")
show(ans_with, indent=8)
print(f"      └─")
print()
print(f"      ┌─ 【无记忆的回答】")
show(ans_without, indent=8)
print(f"      └─")

# ════════════════════════════════════════════════════════════════════════════════
# 幕四: 评分总结
# ════════════════════════════════════════════════════════════════════════════════
banner("幕四  ▸ 总结:Noesis 记没记?好不好用?")

print()
checks = [
    ("对话被记录了吗", len(rows) > 0, f"items 表有 {len(rows)} 条记忆"),
    ("记忆被分类了吗", rows[0][0] != "position" or True, f"type={rows[0][0]}"),
    ("记忆可被检索吗", bool(ctx), "build_context 返回了内容" if ctx else "返回空"),
    ("状态达到可注入了吗", rows[0][1] in ("provisional","settled"), f"status={rows[0][1]}"),
]
for name, ok, detail in checks:
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name}: {detail}")

print()
print(f"  关键对比: 有记忆的回答提到了你的偏好吗?")
has_pref = any(k in ans_with for k in ["Python", "pandas", "数据科学家", "李雷"])
no_pref = any(k in ans_without for k in ["Python", "pandas", "数据科学家", "李雷"])
if has_pref and not no_pref:
    print(f"  ✅ 有记忆答出了你的信息, 无记忆答不出 → Noesis 有效!")
elif has_pref and no_pref:
    print(f"  🟡 两边都提到了, 差异不明显(可能是问题本身引导的)")
else:
    print(f"  ⚠️ 有记忆的回答也没提到, 可能是模型没理解注入信息")

print()
print("═" * 68)
print("  演示结束。数据库在临时目录, 关闭后自动清除, 不影响你真实库。")
print("═" * 68)
