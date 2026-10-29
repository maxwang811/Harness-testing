"""Goal 3, offline: E01, E02, E04 and E06 on each framework slice that is installed.

    pip install -r requirements.txt
    pytest
"""
import asyncio
import importlib
import json

import pytest

from contract import AgentSpec, ModelReply, RunContext, RunRequest, Tool, ToolCall, relay_tool_name
from fakes import FakeModel, FakeTools

IMPLS = {  # name -> (framework package, module, class)
    "pydantic-ai": ("pydantic_ai", "slice_pydantic_ai", "PydanticAISlice"),
    "openai-agents": ("agents", "slice_openai_agents", "OpenAIAgentsSlice"),
}

ANALYST = AgentSpec("staff:analyst@1", "You are the business analyst.")
ASSISTANT = AgentSpec(
    "staff:assistant@1",
    "You are the team's assistant.",
    tools=(
        Tool("web_search", "Search the web",
             {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
        Tool("send_email", "Send an email",
             {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}},
              "required": ["to", "subject"]},
             needs_approval=True),
    ),
    relays=(ANALYST.ref,),
)


@pytest.fixture(params=list(IMPLS))
def harness(request):
    framework, module, cls = IMPLS[request.param]
    pytest.importorskip(framework)
    return getattr(importlib.import_module(module), cls)()


def run(harness, model, tools, run_id="r1", **kw):
    req = RunRequest(run_id=run_id, agent=ASSISTANT, roster={ANALYST.ref: ANALYST}, **kw)
    events = asyncio.run(harness.run(req, RunContext(model, tools)))
    check_basics(events)
    return events


def check_basics(events):
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert events[0].type == "run.started" and events[-1].type == "run.finished"
    called = [e.data["call_id"] for e in events if e.type == "tool.called"]
    results = [e.data["call_id"] for e in events if e.type == "tool.result"]
    assert sorted(called) == sorted(results) and len(set(called)) == len(called), "one result per tool call"


def types(events):
    return [e.type for e in events]


def test_e01_home_chat(harness):
    model = FakeModel({ASSISTANT.ref: [ModelReply("本周完成了登录模块重构。")]})
    ev = run(harness, model, FakeTools({}), input="总结本周进展")
    assert types(ev) == ["run.started", "message.completed", "run.finished"]
    assert ev[1].agent == ASSISTANT.ref and ev[1].data["text"] == "本周完成了登录模块重构。"
    assert ev[-1].data["outcome"] == "completed"
    assert model.calls[0]["messages"][-1] == {"role": "user", "content": "总结本周进展"}


def test_e02_relay(harness):
    model = FakeModel({
        ASSISTANT.ref: [ModelReply(tool_calls=(ToolCall("c1", relay_tool_name(ANALYST.ref), {}),))],
        ANALYST.ref: [ModelReply("上个月 7 日留存下降了 6 个百分点。")],
    })
    ev = run(harness, model, FakeTools({}), input="@分析师 看一下留存数据")
    assert types(ev) == ["run.started", "handoff", "message.completed", "run.finished"]
    assert ev[1].data["to"] == ANALYST.ref
    assert ev[2].agent == ANALYST.ref  # the target speaks; FakeModel fails if the assistant is called again
    assert relay_tool_name(ANALYST.ref) in model.calls[0]["tools"]


def test_e04_tool_call(harness):
    model = FakeModel({ASSISTANT.ref: [
        ModelReply(tool_calls=(ToolCall("t1", "web_search", {"q": "竞品X 定价"}),)),
        ModelReply("竞品 X 基础版每月 99 元。"),
    ]})
    tools = FakeTools({"web_search": "竞品X 基础版 ¥99/月"})
    ev = run(harness, model, tools, input="查一下竞品 X 的定价")
    assert types(ev) == ["run.started", "tool.called", "tool.result", "message.completed", "run.finished"]
    assert tools.calls == [("web_search", {"q": "竞品X 定价"}, "r1:t1")]  # once, with the idempotency key
    last = model.calls[1]["messages"][-1]
    assert last["role"] == "tool" and last["tool_call_id"] == "t1" and "¥99" in last["content"]


def test_e06_approval(harness):
    model = FakeModel({ASSISTANT.ref: [
        ModelReply(tool_calls=(ToolCall("e1", "send_email", {"to": "boss@corp.com", "subject": "周报"}),)),
        ModelReply("已发送。"),
    ]})
    tools = FakeTools({"send_email": "sent"})

    first = run(harness, model, tools, input="把周报发给 boss@corp.com")
    assert types(first) == ["run.started", "interrupt.requested", "run.finished"]
    card, fin = first[1].data, first[-1].data
    assert card["assignee"] and card["default"] and card["expires_at"]  # who answers, default, deadline
    assert fin["outcome"] == "interrupted" and fin["interrupt_ids"] == [card["interrupt_id"]]
    assert tools.calls == []  # nothing is sent before approval

    state = json.loads(json.dumps(fin["resume_state"]))  # must survive storage as JSON
    second = run(harness, model, tools, run_id="r2", resume_state=state,
                 resume_inputs={card["interrupt_id"]: True})
    assert types(second) == ["run.started", "tool.called", "tool.result", "message.completed", "run.finished"]
    assert [c[0] for c in tools.calls] == ["send_email"]  # sent exactly once, never asked twice
