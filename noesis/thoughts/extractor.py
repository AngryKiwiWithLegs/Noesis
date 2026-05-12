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
    Deterministic extractor for tests.
    Rules (in order):
      - "随口" or "也许" in text → return [] (weak assertion, skip)
      - "决定" or "确定" in text → position with high confidence
      - "我叫" or "I am" in text → identity
      - "偏好" or "prefer" in text → preference
      - "?" or "？" at end → question
      - default → position with medium confidence
    """

    def extract(
        self,
        messages: list[dict] | str,
        source_tool: str = "",
        session_id:  str = "",
    ) -> list[ThoughtCandidate]:
        text = _format_conversation(messages)
        if not text.strip():
            return []

        t = text.lower()

        # Weak assertions → skip
        if any(w in t for w in ["随口", "也许", "maybe", "perhaps", "不确定"]):
            return []

        # Determine type
        if any(w in t for w in ["我叫", "i am", "i'm a", "我是"]):
            thought_type: ThoughtType = "identity"
            strength, conf = 0.9, 0.7
        elif any(w in t for w in ["决定", "decided", "确定", "i've chosen"]):
            thought_type = "event"
            strength, conf = 0.9, 0.6
        elif any(w in t for w in ["偏好", "prefer", "喜欢用", "prefer to"]):
            thought_type = "preference"
            strength, conf = 0.7, 0.4
        elif text.rstrip().endswith(("?", "？")):
            thought_type = "question"
            strength, conf = 0.5, 0.3
        else:
            thought_type = "position"
            strength, conf = 0.6, 0.35

        return [ThoughtCandidate(
            type=thought_type,
            text=text[:200].strip(),
            initial_confidence=conf,
            assertion_strength=strength,
            source_tool=source_tool,
            source_session=session_id,
            topic_cluster="test-topic",
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
