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
| `test_live.py` | 3 | The same four cases on both slices, real model, fake tools |

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
search or email really happens. The live checks are on the contract rules and the shape of each
case, not on exact wording.

Status: `test_cases.py` passes on both slices. `test_live.py` has been checked against a local
stand-in for the OpenAI endpoint, not yet against a real model.
