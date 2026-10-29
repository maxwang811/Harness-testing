# Harness-testing

A small offline test for the harness Phase-1 goals: one contract, the same slice on two
frameworks, four test cases. Nothing here talks to a real model, database or the Captain backend.

| File | Goal | What it is |
|---|---|---|
| `contract.py` | 1 | The contract: request, 7 event types, 2 ports (model gateway, tools), the `Harness` entry point |
| `slice_pydantic_ai.py` | 2 | The slice on Pydantic AI |
| `slice_openai_agents.py` | 2 | The slice on the OpenAI Agents SDK |
| `fakes.py` | 3 | Scripted model and recording tools |
| `test_cases.py` | 3 | E01 home chat, E02 @-relay, E04 tool call, E06 approval, run on both slices |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Status: the contract, fakes and tests were checked against a throwaway implementation. The two
slices are written from the framework docs but have not been run yet; making `pytest` pass on
both is the first step of Goal 2.
