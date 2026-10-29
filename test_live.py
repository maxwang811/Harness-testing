"""Goal 3, live: E01, E02, E04 and E06 on each slice against a real model (llm.OpenAIGateway).

Tools stay fake, so nothing is really searched or emailed. Skipped until OPENAI_API_KEY is set
(in the environment or in .env). A real model words things its own way, so these check the
contract rules and the shape of each case, not exact text.

    pytest test_live.py -v -s     # -s prints every run's events
"""
import asyncio
import dataclasses
import json
import os

import pytest

from contract import RunContext, RunRequest
from fakes import FakeTools
from llm import OpenAIGateway, load_env
from test_cases import ANALYST, ASSISTANT, check_basics, harness, types  # noqa: F401 (harness is a fixture)

load_env()
pytestmark = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="paste OPENAI_API_KEY into .env")

# The offline specs, plus what a real model needs to know to play each case.
LIVE_ANALYST = dataclasses.replace(ANALYST, instructions=(
    "You are the business analyst. Answer briefly, in Chinese. "
    "Known data: last month's 7-day retention dropped by 6 percentage points."))
LIVE_ASSISTANT = dataclasses.replace(ASSISTANT, instructions=(
    "You are the team's assistant. Answer briefly, in Chinese. "
    "This week the team finished refactoring the login module. "
    "When the user @-mentions the analyst (@分析师), hand over with relay_to_analyst and do not answer yourself. "
    "Use web_search for facts you do not know. "
    "When asked to send an email, call send_email right away; the system asks the user to approve it."))
TOOL_OUTPUTS = {"web_search": "竞品X 基础版 ¥99/月", "send_email": "sent"}


def run(harness, tools, run_id="r1", **kw):
    """One run on a fresh gateway (asyncio.run closes the loop its HTTP client was bound to)."""
    model = OpenAIGateway()
    req = RunRequest(run_id=run_id, agent=LIVE_ASSISTANT, roster={ANALYST.ref: LIVE_ANALYST}, **kw)
    events = asyncio.run(harness.run(req, RunContext(model, tools)))
    for e in events:
        print(f"{harness.name} {run_id} #{e.seq} {e.type} [{e.agent}] {json.dumps(e.data, ensure_ascii=False)[:200]}")
    check_basics(events)
    return events, model


def test_e01_home_chat(harness):
    ev, model = run(harness, FakeTools(TOOL_OUTPUTS), input="总结本周进展")
    assert ev[-1].data["outcome"] == "completed"
    assert ev[-2].type == "message.completed" and ev[-2].agent == ASSISTANT.ref and ev[-2].data["text"].strip()
    assert "handoff" not in types(ev) and "interrupt.requested" not in types(ev)
    assert model.calls[0]["messages"][-1] == {"role": "user", "content": "总结本周进展"}


def test_e02_relay(harness):
    ev, _ = run(harness, FakeTools(TOOL_OUTPUTS), input="@分析师 看一下留存数据")
    assert "handoff" in types(ev), "the assistant did not relay"
    at = types(ev).index("handoff")
    assert ev[at].data["to"] == ANALYST.ref
    after = ev[at + 1:-1]
    assert any(e.type == "message.completed" for e in after)
    assert all(e.agent == ANALYST.ref for e in after)  # the target speaks, the caller does not speak again
    assert ev[-1].data["outcome"] == "completed"


def test_e04_tool_call(harness):
    tools = FakeTools(TOOL_OUTPUTS)
    ev, model = run(harness, tools, input="查一下竞品 X 的定价")
    called = [e.data for e in ev if e.type == "tool.called"]
    assert called and all(c["tool"] == "web_search" for c in called)
    assert [c[2] for c in tools.calls] == [f"r1:{c['call_id']}" for c in called]  # once each, with the idempotency key
    assert any(m["role"] == "tool" and "¥99" in m["content"] for m in model.calls[-1]["messages"])
    assert ev[-2].type == "message.completed" and "99" in ev[-2].data["text"]
    assert ev[-1].data["outcome"] == "completed"


def test_e06_approval(harness):
    tools = FakeTools(TOOL_OUTPUTS)
    first, _ = run(harness, tools, input="把周报发给 boss@corp.com")
    cards = [e.data for e in first if e.type == "interrupt.requested"]
    fin = first[-1].data
    assert [c["tool"] for c in cards] == ["send_email"] and cards[0]["args"]["to"] == "boss@corp.com"
    assert cards[0]["assignee"] and cards[0]["default"] and cards[0]["expires_at"]  # who answers, default, deadline
    assert fin["outcome"] == "interrupted" and fin["interrupt_ids"] == [cards[0]["interrupt_id"]]
    assert tools.calls == []  # nothing is sent before approval

    state = json.loads(json.dumps(fin["resume_state"]))  # must survive storage as JSON
    second, _ = run(harness, tools, run_id="r2", resume_state=state, resume_inputs={cards[0]["interrupt_id"]: True})
    assert types(second)[1:3] == ["tool.called", "tool.result"]
    assert [c[0] for c in tools.calls] == ["send_email"]  # sent exactly once, never asked twice
    assert second[-2].type == "message.completed" and second[-1].data["outcome"] == "completed"
