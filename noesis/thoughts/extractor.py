"""
noesis/thoughts/extractor.py

THE CRITICAL FILE. Quality here determines everything downstream.

The extractor turns raw conversation turns into typed ThoughtCandidates.
Bad extraction = bad confidence scores = bad injection = useless system.

Design decisions:
- English prompt works for Chinese input (Claude/GPT handle it fine)
- Returns structured JSON; strip/retry on parse failure
- MockExtractor for tests and offline development
- Supports Anthropic (default) and OpenAI
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from .types import ThoughtCandidate, ThoughtType

logger = logging.getLogger(__name__)


# ── Extraction prompt ─────────────────────────────────────────────────────────
# This prompt is the hardest-to-get-right piece of the entire project.
# Iterate on it based on real failures captured in tests/test_extraction_quality.py

_SYSTEM_PROMPT = """You are a thought extraction system for a personal knowledge base.
Given a conversation, identify only the USER's genuine thoughts worth preserving long-term.

## Extract when the user:
- Takes a clear position ("I think X is better than Y")
- Decides something ("I've decided to use sqlite-vec")
- Expresses a real preference ("I prefer local models over cloud")
- Reveals who they are ("I'm a senior ML engineer")
- Raises a question they genuinely want to explore (not just asking for help)

## DO NOT extract:
- Requests for help or tasks ("can you help me write...")
- Simple acknowledgements ("ok", "got it", "thanks")
- Restatements of what the AI just said
- Highly speculative throwaway comments
- Questions that are purely asking for information (vs exploring ideas)

## Output rules:
- Rewrite the thought as a clean, third-person declarative statement
- Keep it concise (1-2 sentences max)
- Preserve the user's actual stance — don't soften or harden it
- topic_cluster: 2-4 word kebab-case label ("rag-retrieval", "python-tooling", "career-goals")

## assertion_strength scale:
0.9 = "I've decided" / "I'm certain" / "I definitely" / "I always"
0.7 = "I think" / "I believe" / "I feel" / "I prefer" / "I want"
0.5 = neutral statement of observation or fact
0.2 = "maybe" / "perhaps" / "I wonder if" / "not sure"
0.1 = highly speculative or self-questioning

## Types:
position    — a stance or belief about how things are or should be
question    — a genuine open question driving exploration (status: open)
event       — something that happened or a decision made
preference  — how the user likes things done
identity    — who the user is (role, background, stable facts)

Return ONLY valid JSON, no preamble, no markdown fences:
{"thoughts": [{"type": "...", "text": "...", "assertion_strength": 0.7,
               "initial_confidence": 0.35, "topic_cluster": "..."}]}
If nothing worth capturing: {"thoughts": []}"""

_USER_TEMPLATE = """Conversation to analyse:

{conversation}

Extract only the user's genuine thoughts (ignore assistant turns)."""


# ── Abstract base ─────────────────────────────────────────────────────────────

class AbstractExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        messages: list[dict] | str,
        source_tool: str = "",
        session_id:  str = "",
    ) -> list[ThoughtCandidate]:
        ...


# ── Cloud LLM extractor ───────────────────────────────────────────────────────

class CloudLLMExtractor(AbstractExtractor):
    """
    Calls Anthropic (default) or OpenAI to extract thoughts.
    Uses the cheapest fast models — Haiku / GPT-4o-mini.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model:    str = "claude-haiku-4-5-20251001",
        api_key:  str | None = None,
        max_retries: int = 2,
    ):
        self.provider    = provider
        self.model       = model
        self.api_key     = api_key
        self.max_retries = max_retries
        self._client: Any = None

    @property
    def client(self):
        if self._client is None:
            if self.provider == "anthropic":
                try:
                    import anthropic
                    self._client = anthropic.Anthropic(
                        api_key=self.api_key  # None → reads ANTHROPIC_API_KEY
                    )
                except ImportError as e:
                    raise ImportError("pip install anthropic") from e
            elif self.provider == "openai":
                try:
                    import openai
                    self._client = openai.OpenAI(api_key=self.api_key)
                except ImportError as e:
                    raise ImportError("pip install openai") from e
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        return self._client

    def extract(
        self,
        messages: list[dict] | str,
        source_tool: str = "",
        session_id:  str = "",
    ) -> list[ThoughtCandidate]:
        conversation = _format_conversation(messages)
        if not conversation.strip():
            return []

        user_msg = _USER_TEMPLATE.format(conversation=conversation)

        for attempt in range(self.max_retries + 1):
            try:
                raw = self._call_llm(user_msg)
                candidates = _parse_response(raw)
                return [
                    ThoughtCandidate(
                        type=c["type"],
                        text=c["text"],
                        initial_confidence=float(c.get("initial_confidence", 0.3)),
                        assertion_strength=float(c.get("assertion_strength", 0.5)),
                        source_tool=source_tool,
                        source_session=session_id,
                        topic_cluster=c.get("topic_cluster", ""),
                    )
                    for c in candidates
                    if _valid_type(c.get("type", ""))
                ]
            except Exception as e:
                logger.warning(f"Extraction attempt {attempt+1} failed: {e}")
                if attempt == self.max_retries:
                    logger.error("All extraction attempts failed; returning empty")
                    return []

        return []

    def _call_llm(self, user_msg: str) -> str:
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            return resp.content[0].text

        elif self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",  "content": _SYSTEM_PROMPT},
                    {"role": "user",    "content": user_msg},
                ],
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content

        raise ValueError(f"Unknown provider: {self.provider}")


# ── Mock extractor (tests + offline dev) ─────────────────────────────────────

class MockExtractor(AbstractExtractor):
    """
    Deterministic extractor for tests and offline development (no API key).

    Unlike the cloud extractor, this runs entirely locally with rule-based
    classification. To make offline experiments viable, it:
      - Stores ONLY the user's statements (assistant response is discarded),
        normalised into a clean third-person declarative form
      - Tags each candidate with an elevated assertion_strength so the
        ConfidenceScorer can promote single strong statements to provisional
        (see mock_boost in confidence.py)

    Rules (in order):
      - "随口" or "也许" in text → return [] (weak assertion, skip)
      - "我叫" or "I am" in text → identity
      - "决定" or "确定" in text → event
      - "偏好" or "prefer" in text → preference
      - "?" or "？" at end → question
      - default → position
    """

    def extract(
        self,
        messages: list[dict] | str,
        source_tool: str = "",
        session_id:  str = "",
    ) -> list[ThoughtCandidate]:
        # In mock mode we only care about what the user actually said.
        user_text = _extract_user_only(messages)
        if not user_text.strip():
            return []

        t = user_text.lower()

        # Weak assertions → skip
        if any(w in t for w in ["随口", "也许", "maybe", "perhaps", "不确定"]):
            return []

        # Determine type + assertion strength
        if any(w in t for w in ["我叫", "i am", "i'm a", "我是"]):
            thought_type: ThoughtType = "identity"
            strength, conf = 0.95, 0.8
        elif any(w in t for w in ["决定", "decided", "确定", "i've chosen", "我选择"]):
            thought_type = "event"
            strength, conf = 0.95, 0.8
        elif any(w in t for w in ["偏好", "prefer", "喜欢用", "倾向", "倾向用"]):
            thought_type = "preference"
            strength, conf = 0.85, 0.7
        elif user_text.rstrip().endswith(("?", "？")):
            thought_type = "question"
            strength, conf = 0.6, 0.5
        else:
            thought_type = "position"
            strength, conf = 0.8, 0.65

        clean = _normalise_thought(user_text)

        return [ThoughtCandidate(
            type=thought_type,
            text=clean,
            initial_confidence=conf,
            assertion_strength=strength,
            source_tool=source_tool,
            source_session=session_id,
            topic_cluster=_infer_cluster(clean),
        )]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_conversation(messages: list[dict] | str) -> str:
    if isinstance(messages, str):
        return messages
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role    = m.get("role", "")
        content = m.get("content", "")
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _extract_user_only(messages: list[dict] | str) -> str:
    """Return only the user's utterance(s), discarding assistant/system turns.

    Used by MockExtractor so the stored thought is the user's own words,
    not the full conversation transcript.
    """
    if isinstance(messages, str):
        return messages
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
    return "\n".join(parts)


def _normalise_thought(user_text: str) -> str:
    """Trim to a single clean statement, drop role prefixes.

    The proxy forwards turns as [{'role':'user','content':...}] but the raw
    text can still contain 'USER:' prefixes from upstream formatting. We also
    collapse whitespace and cap length so injected context stays compact.
    """
    line = user_text.strip().split("\n")[0]            # first statement only
    line = re.sub(r"^(user|用户)\s*[:：]\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\s+", " ", line).strip()
    return line[:240]


def _infer_cluster(text: str) -> str:
    """Topic-cluster guess from keywords. Fast (no embedding call) — the
    default for the hot write path. Bilingual EN+ZH domain map.

    The map is intentionally broad: ~12 domains covering the technology /
    engineering / career space the experiment corpus actually occupies.
    The first match wins, so more specific domains are listed first.
    """
    t = (text or "").lower()
    for cluster, keywords in _DOMAIN_KEYWORDS:
        if any(k in t for k in keywords):
            return cluster
    return "general"


# Bilingual keyword map. (cluster, [keywords]) — order matters: first match
# wins, so specific/overlapping domains go before general ones.
_DOMAIN_KEYWORDS: list[tuple[str, list[str]]] = [
    ("vector-store",  ["sqlite-vec", "faiss", "milvus", "qdrant", "向量",
                       "vector", "embedding", "嵌入", "retrieval", "检索"]),
    ("database",      ["postgres", "mysql", "redis", "mongo", "sqlite",
                       "数据库", "关系型", "rdb", "aof"]),
    ("languages",     ["python", "rust", "java", "golang", "typescript",
                       "javascript", "swift", "kotlin", "c++", "编程",
                       "语言", "技术栈"]),
    ("frontend",      ["react", "vue", "svelte", "tailwind", "css", "前端",
                       "typescript", "javascript"]),
    ("api-design",    ["graphql", "rest", "grpc", "endpoint", "openapi",
                       "接口"]),
    ("cloud-infra",   ["kubernetes", "docker", "serverless", "lambda", "aws",
                       "azure", "gcp", "微服务", "monorepo", "ci/cd", "argocd",
                       "架构", "分布式", "基础设施"]),
    ("messaging",     ["kafka", "rabbitmq", "队列", "queue", "消息"]),
    ("devtools",      ["git", "rebase", "vim", "neovim", "linux", "macos",
                       "hugo", "wordpress", "gitbook", "obsidian", "工具",
                       "编辑器"]),
    ("llm-choice",    ["llm", "gemini", "gpt", "claude", "deepseek", "模型",
                       "transformer", "fine-tune", "finetune", "微调"]),
    ("data-science",  ["pandas", "numpy", "scikit", "machine learning",
                       "机器学习", "数据科学", "scientist", "数据分析"]),
    ("engineering",   ["code review", "unit test", "refactor", "测试驱动",
                       "重构", "性能优化", "ci pipeline", "tech debt"]),
    ("career",        ["工作", "团队", "工程师", "developer", "engineer",
                       "硕士", "毕业", "面试", "职业", "公司", "项目经验",
                       "i am", "my name", "我叫", "我是", "经验"]),
]


def _infer_cluster_embedding(
    text: str,
    embedding_model,
    existing: list[tuple[str, list[float]]],
    threshold: float = 0.55,
) -> str:
    """Embedding-aware cluster assignment. Slower (one embed call) but far
    more accurate than keywords — used by the `recluster` command.

    `existing` is a list of (cluster_label, embedding_vector) for already-
    classified nodes. We assign `text` to the cluster of its most similar
    existing neighbor, IF that similarity >= threshold. Otherwise we fall
    back to keyword inference, and only return "general" if that also fails.

    This makes clusters self-organizing: as the corpus grows, new nodes
    join the cluster their content most resembles, without any pre-set
    keyword list needing to anticipate every topic.
    """
    # Keyword first-pass: if it clearly matches a domain, trust it (fast + exact)
    kw = _infer_cluster(text)
    if kw != "general":
        return kw

    if embedding_model is None or not existing:
        return "general"

    import math
    vec = embedding_model.embed(text)
    if not any(vec):
        return "general"

    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    best_cluster, best_sim = "general", 0.0
    for cluster, evec in existing:
        sim = _cos(vec, evec)
        if sim > best_sim:
            best_sim, best_cluster = sim, cluster

    return best_cluster if best_sim >= threshold else "general"


def _parse_response(raw: str) -> list[dict]:
    """Parse LLM JSON output. Strips markdown fences if present."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the JSON object
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in: {raw[:200]}")
    data = json.loads(m.group())
    thoughts = data.get("thoughts", [])
    if not isinstance(thoughts, list):
        return []
    return thoughts


_VALID_TYPES = {"position", "question", "event", "preference", "identity",
                "synthesis", "contradiction"}

def _valid_type(t: str) -> bool:
    return t in _VALID_TYPES
