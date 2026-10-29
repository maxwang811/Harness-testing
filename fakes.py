"""Fake ports for offline tests: a scripted model and tools that record their calls."""
from __future__ import annotations

from contract import ModelReply


class FakeModel:
    """Returns the scripted replies for each agent in order. An unexpected extra call fails the test."""

    def __init__(self, script: dict[str, list[ModelReply]]):
        self.script = {agent: list(replies) for agent, replies in script.items()}
        self.calls: list[dict] = []

    async def chat(self, agent: str, messages: list[dict], tools: list[dict]) -> ModelReply:
        self.calls.append({"agent": agent, "messages": messages, "tools": [t["name"] for t in tools]})
        replies = self.script.get(agent, [])
        if not replies:
            raise AssertionError(f"unexpected model call for {agent}")
        return replies.pop(0)


class FakeTools:
    def __init__(self, outputs: dict[str, str]):
        self.outputs = outputs
        self.calls: list[tuple[str, dict, str]] = []

    async def call(self, name: str, args: dict, idempotency_key: str) -> str:
        self.calls.append((name, args, idempotency_key))
        return self.outputs[name]
