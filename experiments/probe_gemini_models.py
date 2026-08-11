#!/usr/bin/env python3
"""列出这个 Gemini key 能访问的模型, 看是否有任何模型可用。"""
import httpx, json

GEMINI_KEY = "YOUR_GEMINI_API_KEY_HERE"

print("=== 该 key 可访问的模型列表 ===")
r = httpx.get(
    "https://generativelanguage.googleapis.com/v1beta/models",
    headers={"x-goog-api-key": GEMINI_KEY},
    timeout=30,
)
print(f"HTTP {r.status_code}")
if r.status_code == 200:
    models = r.json().get("models", [])
    print(f"共 {len(models)} 个模型")
    for m in models[:15]:
        name = m.get("name","")
        methods = m.get("supportedGenerationMethods", [])
        print(f"  {name:45s} {methods}")
else:
    print(r.text[:500])
