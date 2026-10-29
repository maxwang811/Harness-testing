"""Generate the contract's JSON Schemas into schemas/, or check them (PR CI contract-drift check).

    python -m tools.gen_schemas           # write
    python -m tools.gen_schemas --check   # exit 1 if committed schemas differ from the code
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from harness.events import event_json_schema
from harness.types import AgentSpec, ResumeInput, RunRequest

OUT = Path(__file__).resolve().parent.parent / "schemas"


def _model_schema(model, name: str) -> dict:
    s = TypeAdapter(model).json_schema(ref_template="#/$defs/{model}")
    s["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    s["$id"] = f"https://captain.internal/schemas/{name}.v1.json"
    return s


def build() -> dict[str, dict]:
    return {
        "harness_event.v1.json": event_json_schema(),
        "run_request.v1.json": _model_schema(RunRequest, "run_request"),
        "agent_spec.v1.json": _model_schema(AgentSpec, "agent_spec"),
        "resume_input.v1.json": _model_schema(ResumeInput, "resume_input"),
    }


def main(argv: list[str]) -> int:
    check = "--check" in argv
    OUT.mkdir(exist_ok=True)
    drift = []
    for name, schema in build().items():
        text = json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        path = OUT / name
        if check:
            if not path.exists() or path.read_text() != text:
                drift.append(name)
        else:
            path.write_text(text)
    if check and drift:
        print("contract drift: regenerate with `python -m tools.gen_schemas` and review:", ", ".join(drift))
        return 1
    print("schemas", "up to date" if check else f"written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
