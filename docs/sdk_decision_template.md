# SDK approach: decision (one page)

Date: ____ · Author: ____ · Reviewers: harness group, PX group (Victor for the cut points)

## Decision

We build loop-v1 on **____** (Pydantic AI / OpenAI Agents SDK / our own loop). The contract in
`backend/harness/` stays ours; the framework is one implementation of `Harness` behind the switch.

## Hard gates (all must pass, else the candidate is out)

| Gate | Pydantic AI | OpenAI Agents SDK | loop-ref |
|---|---|---|---|
| Every model call goes through our ModelGateway (no direct provider client) | | | |
| Approval -> process restart -> continuation works from the serialized state alone | | | |
| Offline cases E01 E02 E04 E05 E06 E07 E08 E08b E12 pass (`pytest tests/test_slices.py`) | | | |
| No network or API key needed in PR CI | | | |

## Scorecard (weights agreed before measuring)

| Criterion (weight) | How it is measured | Pydantic AI | OpenAI Agents SDK | loop-ref |
|---|---|---|---|---|
| Gateway integration cost (25) | `tools/loc_report.py` adapter lines; # framework types we construct; usage/route visible? | | | |
| Contract fidelity (20) | events we had to synthesize outside the framework; author/handoff/steps available? | | | |
| Approval and resume (20) | state size (bytes) after E06 run 1; JSON-serializable; survives a version bump? | | | |
| Weight (10) | added wheels + MB (`uv pip install` diff); import time (`python -X importtime`) | | | |
| Eval tooling (10) | can our cases run in-process with fakes? pydantic-evals / tracing usable? | | | |
| Behavior + size (15) | cases passed; slice LOC (excluding `impls/common.py`) | | | |
| **Total (100)** | | | | |

Decision rule (goals doc, open question 1): Pydantic AI unless the OpenAI Agents SDK scores
at least 15 points higher or Pydantic AI fails a hard gate.

## What we borrow either way

LangGraph semantics for checkpoints, interrupt/resume and Last-Event-ID resume; ____

## Risks and follow-ups
