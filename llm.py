"""Real model gateway: the ModelGateway port over any OpenAI-compatible chat-completions API.

Settings come from the environment or from .env next to this file (the environment wins):
  OPENAI_API_KEY   required
  OPENAI_BASE_URL  optional, for another OpenAI-compatible provider (DeepSeek, Qwen, OpenRouter ...)
  LLM_MODEL        optional, default gpt-4.1-mini
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI

from contract import ModelReply, ToolCall

DEFAULT_MODEL = "gpt-4.1-mini"


def load_env(path: Path = Path(__file__).with_name(".env")) -> None:
    """KEY=value lines from .env into os.environ; blank values and comments are skipped."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if sep and value and not key.startswith("#"):
            os.environ.setdefault(key, value)


def to_openai(m: dict) -> dict:
    """Our chat message -> a chat-completions message."""
    if m["role"] != "assistant" or not m.get("tool_calls"):
        return {k: v for k, v in m.items() if k != "tool_calls"}
    return {"role": "assistant", "content": m["content"] or None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}}
                           for c in m["tool_calls"]]}


def to_openai_tool(t: dict) -> dict:
    params = t["parameters"] if (t.get("parameters") or {}).get("type") else {"type": "object", "properties": {}}
    return {"type": "function", "function": {"name": t["name"], "description": t["description"] or "",
                                             "parameters": params}}


class OpenAIGateway:
    """Calls the real model. Records each call like FakeModel does, so tests can look at what was sent,
    plus the reply, the seconds it took and the (prompt, completion) tokens."""

    def __init__(self, model: str | None = None, client: AsyncOpenAI | None = None):
        load_env()
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.client = client or AsyncOpenAI()  # reads OPENAI_API_KEY and OPENAI_BASE_URL
        self.calls: list[dict] = []

    async def chat(self, agent: str, messages: list[dict], tools: list[dict]) -> ModelReply:
        record = {"agent": agent, "messages": messages, "tools": [t["name"] for t in tools]}
        self.calls.append(record)
        extra = {"tools": [to_openai_tool(t) for t in tools]} if tools else {}
        start = time.perf_counter()
        resp = await self.client.chat.completions.create(
            model=self.model, messages=[to_openai(m) for m in messages], **extra)
        msg = resp.choices[0].message
        calls = [c for c in msg.tool_calls or [] if c.type == "function"]
        reply = ModelReply(text=msg.content or "", tool_calls=tuple(
            ToolCall(c.id, c.function.name, json.loads(c.function.arguments or "{}")) for c in calls))
        usage = resp.usage
        record.update(reply=reply, seconds=time.perf_counter() - start,
                      tokens=(usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0))
        return reply
