#!/usr/bin/env python3
"""
阶段 1.1 — 直连探测: 确认 3 个 API key 各自有效、网络通。
绕过 Noesis 代理，直接打各家 API。只发一个最便宜的请求。
"""
import sys
import os

# key 只在当前进程内存生效,不写入任何文件
OPENAI_KEY    = "YOUR_OPENAI_API_KEY_HERE"
GEMINI_KEY    = "YOUR_GEMINI_API_KEY_HERE"
DEEPSEEK_KEY  = "YOUR_DEEPSEEK_API_KEY_HERE"

import httpx

results = {}

def probe_openai():
    """OpenAI: 直连, 发一个 1-token 请求。"""
    try:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            json={"model": "gpt-4o-mini", "messages": [{"role":"user","content":"hi"}], "max_tokens": 5},
            timeout=30,
        )
        if r.status_code == 200:
            return "✅ 通", f"{r.json()['choices'][0]['message']['content'][:30]!r}"
        return "❌ 失败", f"HTTP {r.status_code}: {r.text[:150]}"
    except Exception as e:
        return "❌ 异常", str(e)[:150]

def probe_gemini():
    """Gemini: 走 OpenAI 兼容端点 (Noesis 用的就是这个)。"""
    try:
        r = httpx.post(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            headers={"Authorization": f"Bearer {GEMINI_KEY}"},
            json={"model": "gemini-2.0-flash", "messages": [{"role":"user","content":"hi"}], "max_tokens": 5},
            timeout=30,
        )
        if r.status_code == 200:
            return "✅ 通", f"{r.json()['choices'][0]['message']['content'][:30]!r}"
        return "❌ 失败", f"HTTP {r.status_code}: {r.text[:150]}"
    except Exception as e:
        return "❌ 异常", str(e)[:150]

def probe_deepseek():
    """DeepSeek: 直连官方 API。"""
    try:
        r = httpx.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"},
            json={"model": "deepseek-chat", "messages": [{"role":"user","content":"hi"}], "max_tokens": 5},
            timeout=30,
        )
        if r.status_code == 200:
            return "✅ 通", f"{r.json()['choices'][0]['message']['content'][:30]!r}"
        return "❌ 失败", f"HTTP {r.status_code}: {r.text[:150]}"
    except Exception as e:
        return "❌ 异常", str(e)[:150]

print("=" * 60)
print("阶段 1.1: 直连 API key 探测 (绕过 Noesis)")
print("=" * 60)
print()
for name, fn in [("OpenAI (gpt-4o-mini)", probe_openai),
                 ("Gemini (gemini-2.0-flash)", probe_gemini),
                 ("DeepSeek (deepseek-chat)", probe_deepseek)]:
    status, detail = fn()
    print(f"{name:30s} {status}")
    print(f"    └─ {detail}")
    print()

print("=" * 60)
