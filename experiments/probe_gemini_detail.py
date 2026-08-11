#!/usr/bin/env python3
"""读取 Gemini 429 完整错误信息, 区分配额类型。"""
import httpx, json

GEMINI_KEY = "YOUR_GEMINI_API_KEY_HERE"

r = httpx.post(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
    headers={"x-goog-api-key": GEMINI_KEY},
    json={"contents":[{"parts":[{"text":"hi"}]}]},
    timeout=30,
)
print(f"HTTP {r.status_code}")
print(json.dumps(r.json(), indent=2, ensure_ascii=False))
