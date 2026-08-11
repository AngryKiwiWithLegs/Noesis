#!/usr/bin/env python3
"""
阶段 1.1c — 确认 gemini-flash-lite-latest 走 OpenAI 兼容端点 (Noesis 实际用的) 可用。
"""
import httpx

GEMINI_KEY = "YOUR_GEMINI_API_KEY_HERE"

print("=== 测试 gemini-flash-lite-latest 走 OpenAI 兼容端点 ===\n")

# 这正是 Noesis 代理会转发到的端点和模型
r = httpx.post(
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    headers={"Authorization": f"Bearer {GEMINI_KEY}"},
    json={
        "model": "gemini-flash-lite-latest",
        "messages": [{"role":"user","content":"用一个词回答: 你好"}],
        "max_tokens": 20,
    },
    timeout=30,
)
print(f"HTTP {r.status_code}")
if r.status_code == 200:
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    print(f"✅ 通 | 回答: {content!r}")
    print(f"   模型: {data.get('model','?')}")
else:
    print(f"❌ | {r.text[:300]}")
