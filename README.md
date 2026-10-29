# Harness-testing

Phase-1 test bed for the Captain backend–harness split: the **contract** (Goal 1), the
**SDK comparison slices** (Goal 2) and the **offline evaluation cases** (Goal 3). Everything
here is written so it can be copied into the Captain repo as `backend/harness/` (contract),
`backend/harness/testing/` (fakes) and `tests/eval/` (cases + runner) without changes to the
design.

## Run it

```bash
uv venv && uv pip install -e ".[dev]"          # pydantic, pyyaml, jsonschema, pytest
pytest                                          # contract, schema-compat, AG-UI mapping, all cases, mutation tests
python -m evals.runner                          # case table: legacy-replay vs loop-ref
python -m tools.gen_schemas --check             # contract-drift check (PR CI)
python -m tools.schema_compat old.json new.json # additive-only check between two schema versions

uv pip install -e ".[slices]"                   # Goal 2: pydantic-ai-slim 2.x + openai-agents 0.24
pytest tests/test_slices.py                     # same cases on both framework slices
python -m evals.runner --impl all               # every implementation side by side
python -m tools.loc_report                      # code size per implementation (scorecard input)
```

## Layout

| Path | Goal | What it is |
|---|---|---|
| `harness/types.py` | 1 | RunRequest, AgentSpec (data), InputItem, FeatureSet, ResumeState/ResumeInput, Capabilities |
| `harness/events.py` | 1 | Event envelope + 19 typed payloads; tolerant parser; JSON Schema builder |
| `harness/ports.py` | 1 | ModelGateway, ToolExecutor, MemoryPort, ThreadReader, CheckpointStore, BudgetGate, Tracer, EventSink, Clock, Middleware, RunContext |
| `harness/api.py` | 1 | `Harness` / `RunHandle` protocols and the lifecycle rules |
| `harness/errors.py` | 1 | Error codes with default retryability |
| `harness/runtime.py` | 1 | Shared emitter (seq, write-before-yield) and queue-backed RunHandle (re-attach, readers never cancel) |
| `harness/mapping/agui.py` | 1 | Harness events -> AG-UI events (the full mapping table as code) |
| `harness/mapping/legacy_sse.py` | 1 | Legacy SSE -> harness events (what the legacy wrappers do) — payload fields assumed until the SSE inventory |
| `schemas/*.v1.json` | 1 | Generated JSON Schemas (committed; CI checks drift) |
| `tools/gen_schemas.py`, `tools/schema_compat.py` | 1 | Drift check and additive-only check |
| `impls/reference_loop.py` | 2 | `loop-ref`: the slice with no framework ("write our own" option) |
| `impls/pydantic_ai_slice.py` | 2 | Candidate A — **drafted, not yet run** |
| `impls/openai_agents_slice.py` | 2 | Candidate B — **drafted, not yet run** |
| `impls/common.py` | 2 | Glue both slices share (events, steps, usage, idempotency, resume validation) |
| `impls/legacy_replay.py` | 3 | Replays a golden legacy SSE stream: the baseline every case is calibrated on |
| `fakes/` | 3 | Track-A fake ports with fault injection and a virtual clock |
| `evals/cases/E*.yaml` | 3 | E01–E13 (+ variants E08b, E09b, E11b): fixtures, faults, checks, staging steps |
| `evals/assertions.py`, `evals/runner.py` | 3 | Machine-checkable assertions and the offline runner |
| `router/` | 3 | Executable spec of the routing switch used by E13 |
| `tests/test_cases_have_teeth.py` | 3 | Mutation tests: a deliberately broken implementation must fail the right case |

## What is verified and what is not

* Verified here (Python 3.13, pydantic 2.13): the contract types and schemas, the AG-UI mapping,
  every Track-A case on `loop-ref` and the calibration of every case on `legacy-replay`
  (core flows pass on legacy; incident cases fail on the legacy stand-in), and 5 mutation tests.
* Not verified: the two framework slices (PyPI was unreachable where this was written), and every
  `legacy` stream and model script — they are **synthetic** (`synthetic: true` in each case) until
  replaced by recordings from staging.
