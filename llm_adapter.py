import os, json, logging, httpx
from typing import Optional

logger = logging.getLogger(__name__)

BACKENDS = [
    {
        "name": "rapid-mlx",
        "base_url": os.getenv("RAPID_MLX_URL", "http://localhost:8000/v1"),
        "model": "default",
        "timeout": 60,
        "enabled": True,
    },
    {
        "name": "ollama",
        "base_url": os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        "timeout": 120,
        "enabled": True,
    },
]

def _chat(system, user, json_mode=False, temperature=0.1, max_tokens=1000):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    for backend in BACKENDS:
        if not backend.get("enabled", True):
            continue
        try:
            payload["model"] = backend["model"]
            resp = httpx.post(
                f"{backend['base_url']}/chat/completions",
                json=payload, timeout=backend["timeout"],
                headers={"Authorization": "Bearer not-needed"},
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            logger.debug(f"[llm_adapter] used {backend['name']}")
            return content
        except Exception as e:
            logger.warning(f"[llm_adapter] {backend['name']} failed: {e}")
            continue
    return None

def extract_json(system, user):
    raw = _chat(system=system, user=user, json_mode=True)
    if raw is None:
        return None
    clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"[llm_adapter] JSON parse failed: {e}")
        return None

def generate_text(system, user, temperature=0.3):
    return _chat(system=system, user=user, json_mode=False, temperature=temperature)

def name_cluster(description):
    result = generate_text(
        system="Reply with ONLY a 2-4 word topic label, no punctuation.",
        user=f"Name this topic cluster:\n{description}",
    )
    return result.strip().split("\n")[0][:40] if result else "未命名主题"

def check_backends():
    results = {}
    for backend in BACKENDS:
        if not backend.get("enabled", True):
            results[backend["name"]] = "disabled"
            continue
        try:
            resp = httpx.get(f"{backend['base_url']}/models", timeout=5,
                             headers={"Authorization": "Bearer not-needed"})
            resp.raise_for_status()
            results[backend["name"]] = "✓ reachable"
        except Exception as e:
            results[backend["name"]] = f"✗ {e}"
    return results

if __name__ == "__main__":
    print("Checking backends...")
    for name, result in check_backends().items():
        print(f"  {name}: {result}")
    print("\nTesting extraction...")
    result = extract_json(
        system=("Extract a thought node from this text. Return JSON with keys: "
                "title (str), type (position/question/synthesis/preference/identity/event/contradiction), "
                "summary (str), confidence (float 0-1)."),
        user="I think transformers fundamentally changed how we approach NLP problems.",
    )
    if result:
        print("Success:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Failed — check that Rapid-MLX or Ollama is running.")
