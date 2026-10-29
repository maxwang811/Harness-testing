import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.events import EVENT_PAYLOADS, HarnessEvent, make_payload, parse_event
from harness.types import AgentSpec, RunRequest

ROOT = Path(__file__).resolve().parent.parent


def ev(type_, payload, seq=1):
    return {"v": 1, "type": type_, "run_id": "run_1", "seq": seq, "ts": "2026-01-01T00:00:00.000Z", "payload": payload}


def test_known_event_parses_to_typed_payload():
    e = parse_event(ev("tool.result", {"call_id": "c1", "ok": True, "summary": "read 3 files"}))
    assert e.payload.call_id == "c1" and e.known


def test_consumer_ignores_unknown_fields_and_types():
    e = parse_event(ev("tool.result", {"call_id": "c1", "ok": True, "added_in_v1_3": 1}) | {"new_envelope_field": 2})
    assert e.payload.ok
    u = parse_event(ev("subagent.paused", {"anything": 1}))
    assert not u.known and u.payload == {"anything": 1}


def test_malformed_known_event_is_an_error():
    with pytest.raises(ValidationError):
        parse_event(ev("tool.result", {"ok": True}))  # call_id missing


def test_producer_rejects_unknown_fields():
    with pytest.raises(TypeError):
        make_payload("tool.result", call_id="c1", ok=True, sumary="typo")


def test_approval_card_fields_are_required():
    with pytest.raises(ValidationError):
        make_payload("interrupt.requested", interrupt_id="i1", kind="approval", reason="tool_approval", prompt="",
                     fingerprint="f")  # no assignee / default_action / expires_at


def test_seq_must_be_positive():
    with pytest.raises(ValidationError):
        HarnessEvent(type="notice", run_id="r", seq=0, ts="x", payload={"code": "x"})


def test_catalogue_covers_the_split_design_events():
    for t in ["run.started", "run.finished", "step.started", "step.finished", "message.started", "message.delta",
              "message.completed", "reasoning.summary", "tool.called", "tool.result", "plan.updated", "handoff",
              "subagent.started", "subagent.finished", "artifact.created", "interrupt.requested", "usage"]:
        assert t in EVENT_PAYLOADS


def test_committed_schemas_match_code():
    r = subprocess.run([sys.executable, "-m", "tools.gen_schemas", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_run_request_roundtrips_as_json():
    from evals.case import load_cases

    req = load_cases(["E02"])[0].request("run_x")
    again = RunRequest.model_validate_json(req.model_dump_json())
    assert again == req and isinstance(again.agent, AgentSpec) and again.roster
