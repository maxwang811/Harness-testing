import copy
import json
from pathlib import Path

from tools.schema_compat import breaking_changes

SCHEMA = json.loads((Path(__file__).resolve().parent.parent / "schemas" / "harness_event.v1.json").read_text())


def test_identical_is_compatible():
    assert breaking_changes(SCHEMA, SCHEMA) == []


def test_adding_an_optional_field_is_additive():
    new = copy.deepcopy(SCHEMA)
    new["$defs"]["ToolResult"]["properties"]["duration_ms"] = {"type": "integer"}
    assert breaking_changes(SCHEMA, new) == []


def test_removing_a_field_is_breaking():
    new = copy.deepcopy(SCHEMA)
    del new["$defs"]["ToolResult"]["properties"]["summary"]
    assert any("summary" in p for p in breaking_changes(SCHEMA, new))


def test_newly_required_is_breaking():
    new = copy.deepcopy(SCHEMA)
    new["$defs"]["ToolResult"]["required"].append("summary")
    assert any("newly required" in p for p in breaking_changes(SCHEMA, new))


def test_removing_an_event_type_is_breaking():
    new = copy.deepcopy(SCHEMA)
    del new["discriminator"]["mapping"]["handoff"]
    assert any("handoff" in p for p in breaking_changes(SCHEMA, new))


def test_removing_an_enum_value_is_breaking():
    new = copy.deepcopy(SCHEMA)
    prop = new["$defs"]["RunFinished"]["properties"]["outcome"]
    prop["enum"] = [v for v in prop["enum"] if v != "cancelled"]
    assert any("enum" in p for p in breaking_changes(SCHEMA, new))
