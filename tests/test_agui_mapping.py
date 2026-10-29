import asyncio

from evals.case import load_cases
from evals.runner import execute
from harness.mapping.agui import translate_all


def agui(case_id, impl="loop-ref", run=0):
    cr = asyncio.run(execute(load_cases([case_id])[0], impl))
    r = cr.runs[run]
    return translate_all(r.request.thread_id, r.events)


def types(out):
    return [o["type"] for o in out]


def test_home_chat_maps_to_text_message_sequence():
    t = types(agui("E01"))
    assert t[0] == "RUN_STARTED" and t[-1] == "RUN_FINISHED"
    assert t.index("TEXT_MESSAGE_START") < t.index("TEXT_MESSAGE_CONTENT") < t.index("TEXT_MESSAGE_END")


def test_usage_never_reaches_clients():
    assert not any("usage" in str(o.get("name", "")) for o in agui("E01"))


def test_interrupt_ends_run_with_interrupt_outcome():
    out = agui("E06")
    fin = out[-1]
    assert fin["type"] == "RUN_FINISHED" and fin["outcome"]["type"] == "interrupt"
    item = fin["outcome"]["interrupts"][0]
    assert item["reason"] == "tool_call" and item["toolCallId"] == "e1" and item["expiresAt"]
    assert item["metadata"]["captain"]["assignee"] and item["metadata"]["captain"]["defaultAction"] == "deny"


def test_continuation_points_at_parent_run():
    out = agui("E06", run=1)
    assert out[0]["parentRunId"] == "run_E06_1"


def test_relay_maps_to_subagent_events_and_tags_children():
    out = agui("E02")
    start = next(o for o in out if o["type"] == "SUBAGENT_STARTED")
    child_text = [o for o in out if o["type"] == "TEXT_MESSAGE_CONTENT"]
    assert child_text and all(o.get("subagentRunId") == start["subagentRunId"] for o in child_text)
    assert any(o["type"] == "CUSTOM" and o["name"] == "captain.handoff" for o in out)


def test_plan_is_snapshot_then_json_patch():
    out = agui("E03")
    states = [o for o in out if o["type"] in ("STATE_SNAPSHOT", "STATE_DELTA")]
    assert states[0]["type"] == "STATE_SNAPSHOT"
    assert all(o["type"] == "STATE_DELTA" for o in states[1:])
    assert any(op["op"] == "replace" and op["path"].endswith("/status") for o in states[1:] for op in o["delta"])


def test_failure_maps_to_run_error_with_code():
    out = agui("E08")
    assert out[-1]["type"] == "RUN_ERROR" and out[-1]["code"] == "credit_exhausted"
