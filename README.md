# Harness-testing

A small test for the harness Phase-1 goals: one contract, the same slice on two frameworks,
four test cases. The cases run offline on a scripted model, and live on a real model once an API
key is set. Nothing here talks to a database or the Captain backend.

| File | Goal | What it is |
|---|---|---|
| `contract.py` | 1 | The contract: request, 7 event types, 2 ports (model gateway, tools), the `Harness` entry point |
| `slice_pydantic_ai.py` | 2 | The slice on Pydantic AI |
| `slice_openai_agents.py` | 2 | The slice on the OpenAI Agents SDK |
| `fakes.py` | 3 | Scripted model and recording tools |
| `llm.py` | 3 | The real model gateway (any OpenAI-compatible chat-completions API) |
| `test_cases.py` | 3 | E01 home chat, E02 @-relay, E04 tool call, E06 approval, on both slices, scripted model |
| `live_eval.py` | 3 | Live comparison: 12 scenarios (the four above + eight harder ones), a stateful fake tool world, the checks, and the side-by-side scorecard |
| `test_live.py` | 3 | Every `live_eval.py` scenario once per slice under pytest, real model, fake tools |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Real model

Paste your key into `.env` (`OPENAI_API_KEY=sk-...`; copy `.env.example` if `.env` is missing).
`OPENAI_BASE_URL` and `LLM_MODEL` in the same file switch to another OpenAI-compatible provider or
model (default `gpt-4.1-mini`). Then:

```bash
pytest test_live.py -v -s
```

`-s` prints every run's events. Without a key the live tests are skipped. Tools stay fake, so no
search or email really happens. A gap a slice is known to have is an expected failure (xfail); a
run where the model took another path than the scenario is about is skipped.

## Live comparison

One run says little about a real model, so `live_eval.py` runs every scenario several times on each
slice and prints a scorecard side by side (also written to `results/`, with the raw events):

```bash
python live_eval.py                  # 12 scenarios x 2 slices x 3 runs, about 2 minutes
python live_eval.py -n 5 -k L06,L10  # some scenarios, 5 runs each
python live_eval.py -v               # also print every run's events
```

Each scenario targets one place where the slices could differ:

| | Scenario | What it probes |
|---|---|---|
| L01–L04 | home chat, @-relay, tool call, approval | baseline (the four offline cases) |
| L05 | vendor price research | multi-step + parallel tool calls: round trips and tokens |
| L06 | 12-page incident log | a 12-step tool chain: the loop's turn budget |
| L07 | search, then email the result | a tool that ran before the pause must not run again on resume |
| L08 | approve one, deny one | how a denial is told to the model: what it does next |
| L09 | card nobody answers | an unanswered card: its default (deny) applies on resume |
| L10 | fetch, then relay | relay after work: does the target see what was gathered? |
| L11 | approval inside a relay | a relay target asks for approval: pause and resume across the handoff |
| L12 | tool raises once | a tool error: told to the model, or the run dies? |

Every run is checked on the contract rules plus three more (every side effect shows up as a
`tool.called` event, the tool gets the args the model sent, every event names the staff member
that produced it), then on the scenario's own checks. A run counts as n/a when the model did not
take the path the scenario is about (judged from what the model asked for, not what the harness
ran), and as infra on a provider error. The cost table gives model calls, tokens, wall time and
resume-state size per run.

Status (gpt-4.1-mini, 3 runs per cell, 2026-10-29): `test_cases.py` passes on both slices. The live
baseline L01–L05 and L07 pass on both. The known gaps, which are the expected failures in
`test_live.py`:

| | pydantic-ai | openai-agents |
|---|---|---|
| L06 | passes | stops at the SDK's default `max_turns=10` (MaxTurnsExceeded) |
| L08 | the model gets "The tool call was denied." and tells the user the send failed or could not be done, not that they declined it | the model gets "Tool execution was not approved." and asks for approval again |
| L09 | resume raises: `DeferredToolResults` needs an answer for every call | passes (unanswered = rejected) |
| L10 | the relay target gets only the user input, not the tool results gathered before | passes (handoff carries the conversation) |
| L11 | resume raises: the continuation does not reach the relay target | resumes; but the card names the assistant, not the analyst who asked (both slices emit cards as `req.agent`) |
| L12 | the tool exception ends the run (no `tool.result`, no `run.finished`) | same |

The resume state is about 4× larger on openai-agents (about 10 KB against 2.4 KB for one approval).
