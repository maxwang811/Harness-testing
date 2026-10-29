"""Within a major version, schemas may only grow. This checker compares two schema files
(old = main, new = the PR) and lists breaking changes. Wire it into PR CI next to --check.

    python -m tools.schema_compat schemas_main/harness_event.v1.json schemas/harness_event.v1.json

Breaking: a removed model, property, enum value or event type; a property newly required;
a changed type. Additive (allowed): a new optional property, model, enum value or event type.
"""
from __future__ import annotations

import json
import sys
from typing import Any


def _sig(prop: dict[str, Any]) -> str:
    """A coarse type signature: type / $ref / anyOf members / const."""
    if "$ref" in prop:
        return "ref:" + prop["$ref"].rsplit("/", 1)[-1]
    if "const" in prop:
        return "const:" + json.dumps(prop["const"])
    if "anyOf" in prop:
        return "anyOf(" + "|".join(sorted(_sig(p) for p in prop["anyOf"])) + ")"
    t = prop.get("type", "any")
    if t == "array":
        return "array[" + _sig(prop.get("items", {})) + "]"
    return str(t)


def _enum(prop: dict[str, Any]) -> set[str]:
    vals = set(json.dumps(v) for v in prop.get("enum", []))
    for p in prop.get("anyOf", []):
        vals |= _enum(p)
    return vals


def _models(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = dict(schema.get("$defs", {}))
    if "properties" in schema:
        models["<root>"] = schema
    return models


def _event_types(schema: dict[str, Any]) -> set[str]:
    mapping = schema.get("discriminator", {}).get("mapping", {})
    return set(mapping)


def breaking_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for t in sorted(_event_types(old) - _event_types(new)):
        problems.append(f"event type removed: {t}")
    om, nm = _models(old), _models(new)
    for name, o in om.items():
        n = nm.get(name)
        if n is None:
            if "enum" in o:
                problems.append(f"enum removed: {name}")
            else:
                problems.append(f"model removed: {name}")
            continue
        if "enum" in o:
            gone = set(json.dumps(v) for v in o["enum"]) - set(json.dumps(v) for v in n.get("enum", []))
            if gone:
                problems.append(f"{name}: enum values removed {sorted(gone)}")
            continue
        op, np_ = o.get("properties", {}), n.get("properties", {})
        for prop in op:
            if prop not in np_:
                problems.append(f"{name}.{prop}: property removed")
            elif _sig(op[prop]) != _sig(np_[prop]):
                problems.append(f"{name}.{prop}: type changed {_sig(op[prop])} -> {_sig(np_[prop])}")
            else:
                gone = _enum(op[prop]) - _enum(np_[prop])
                if gone:
                    problems.append(f"{name}.{prop}: enum values removed {sorted(gone)}")
        newly_required = set(n.get("required", [])) - set(o.get("required", []))
        for prop in sorted(newly_required):
            problems.append(f"{name}.{prop}: newly required")
    return problems


def main(argv: list[str]) -> int:
    old, new = (json.load(open(p)) for p in argv[:2])
    problems = breaking_changes(old, new)
    for p in problems:
        print("BREAKING:", p)
    if problems:
        print("A breaking change needs a new major version (harness_event.v2.json) and a joint review.")
        return 1
    print("compatible (additive only)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
