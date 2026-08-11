#!/usr/bin/env python3
"""测试多个 Gemini 模型, 找出哪个还有配额。"""
import httpx

GEMINI_KEY = "YOUR_GEMINI_API_KEY_HERE"

candidates = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-flash-lite-latest",
    "gemma-4-26b-a4b-it",
]

print("=== 逐个模型探测配额 ===\n")
for model in candidates:
    try:
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": GEMINI_KEY},
            json={"contents":[{"parts":[{"text":"用一个词回答: 你好"}]}],
                  "generationConfig":{"maxOutputTokens":20}},
            timeout=30,
        )
        if r.status_code == 200:
            content = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            print(f"✅ {model:30s} HTTP 200 | 回答: {content!r}")
        else:
            err = r.json().get("error",{})
            # 提取 limit 信息
            msg = err.get("message","")[:80]
            print(f"❌ {model:30s} HTTP {r.status_code} | {msg}")
    except Exception as e:
        print(f"⚠️  {model:30s} 异常: {str(e)[:60]}")
