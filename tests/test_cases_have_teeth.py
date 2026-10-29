"""Mutation tests: break the reference loop on purpose and check the right case catches it.
A case that still passes against a broken implementation is not testing anything."""
import asyncio
import uuid

import pytest

import fakes.context
import impls.reference_loop as rl
from evals.case import load_cases
from evals.runner import evaluate
from harness.ports import BaseMiddleware


def failing_checks(case_id, mutate=None):
    case = load_cases([case_id])[0]
    if mutate:
        case = mutate(case)
    r = asyncio.run(evaluate(case, "loop-ref"))
    return {c.name for c in r.checks if not c.ok}


def test_e04_catches_missing_injection_guard(monkeypatch):
    monkeypatch.setattr(fakes.context, "InjectionGuard", BaseMiddleware)
    assert "untrusted_wrapped" in failing_checks("E04")


def test_e04_catches_random_idempotency_keys(monkeypatch):
    original = rl._Run.execute

    async def random_key(self, tc, key, *a, **kw):
        return await original(self, tc, f"{key}:{uuid.uuid4().hex}", *a, **kw)

    monkeypatch.setattr(rl._Run, "execute", random_key)
    assert "side_effects_once" in failing_checks("E04")


def test_e06_catches_asking_twice(monkeypatch):
    original = rl._Run.apply_resume

    async def forgetful(self, *a, **kw):
        await original(self, *a, **kw)
        self.asked.clear()

    monkeypatch.setattr(rl._Run, "apply_resume", forgetful)
    assert "never_asked_twice" in failing_checks("E06")


def test_e11_catches_thinking_left_on():
    off = failing_checks("E11", lambda c: c.model_copy(update={"agent_patch": {"model_policy": {"reasoning_effort": None}}}))
    assert {"provider_flag", "call_latency_ms_max", "ttft_ms_max"} <= off


def test_e07_catches_reader_cancelling_the_run(monkeypatch):
    from harness.runtime import QueueRunHandle

    original = QueueRunHandle.events

    async def cancelling_events(self, after_seq=0):
        try:
            async for e in original(self, after_seq):
                yield e
        finally:  # mimic a handler that ties the run to the request: reader leaves -> run cancelled
            if not self._done.is_set():
                self._task.cancel()

    monkeypatch.setattr(QueueRunHandle, "events", cancelling_events)
    assert "reattach_exact" in failing_checks("E07")
