#!/usr/bin/env python3
"""
阶段 1.1 — Gemini 直连探测 (新 key)。
"""
import httpx

GEMINI_KEY = "YOUR_GEMINI_API_KEY_HERE"

print("=" * 60)
print("阶段 1.1: Gemini 新 key 直连探测")
print("=" * 60)
print()

# 测试两种端点: OpenAI 兼容端点 (Noesis 用的) + 原生端点
print("--- 测试 A: OpenAI 兼容端点 (generativelanguage.../v1beta/openai) ---")
try:
    r = httpx.post(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        headers={"Authorization": f"Bearer {GEMINI_KEY}"},
        json={"model": "gemini-2.0-flash",
              "messages": [{"role":"user","content":"用一个词回答: 你好"}],
              "max_tokens": 20},
        timeout=30,
    )
    print(f"HTTP {r.status_code}")
    if r.status_code == 200:
        content = r.json()["choices"][0]["message"]["content"]
        print(f"✅ 通 | 回答: {content!r}")
    else:
        print(f"❌ | {r.text[:300]}")
except Exception as e:
    print(f"❌ 异常: {e}")

print()
print("--- 测试 B: 原生 Gemini 端点 (用 x-goog-api-key) ---")
try:
    r = httpx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        headers={"x-goog-api-key": GEMINI_KEY},
        json={"contents":[{"parts":[{"text":"用一个词回答: 你好"}]}],
              "generationConfig":{"maxOutputTokens":20}},
        timeout=30,
    )
    print(f"HTTP {r.status_code}")
    if r.status_code == 200:
        content = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"✅ 通 | 回答: {content!r}")
    else:
        print(f"❌ | {r.text[:300]}")
except Exception as e:
    print(f"❌ 异常: {e}")

print()
print("=" * 60)
