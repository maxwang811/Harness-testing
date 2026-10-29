"""Goal 3, live comparison: harder scenarios that tell the two slices apart, on a real model.

Each scenario is built around one place where the slices could differ (the "probes" column), and
runs several times per slice because a real model varies from run to run. The tools are a fake but
stateful world (World), so every side effect is visible to the checks. Every run is checked on:
  - the contract rules in contract.py, plus three the first live tests did not check: every side
    effect shows up as a tool.called event, the tool gets the args the model sent, and every event
    names the staff member that produced it;
  - the scenario's own checks: did the user get what they asked for, sent once, never asked twice.
A run where the model took another path than the one the scenario is about counts as n/a, and a
provider error (rate limit, network) as infra; neither counts as a pass or a fail.

    python live_eval.py                     # every scenario, 3 runs per slice, prints the scorecard
    python live_eval.py -n 5 -k L06,L10     # some scenarios, 5 runs each
    python live_eval.py -v                  # also print every run's events

test_live.py runs every scenario once per slice under pytest.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import argparse
import asyncio
import datetime as dt
import hashlib
import importlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import openai

from contract import AgentSpec, Event, RunContext, RunRequest, Tool, relay_tool_name
from llm import DEFAULT_MODEL, OpenAIGateway, load_env
from test_cases import IMPLS

# ---------------------------------------------------------------- the world the tools act on

PRICES = {  # cloud vendor -> price history; Arbor rose most (+25%)
    "Lumen": "Lumen 对象存储：2025 年 ¥0.12/GB，2026 年 ¥0.13/GB",
    "Arbor": "Arbor 云主机：2025 年 ¥480/台/月，2026 年 ¥600/台/月",
    "Kite": "Kite CDN：2025 年 ¥0.20/GB，2026 年 ¥0.21/GB",
}
FUNNEL = "上周注册漏斗：访问 18,342，注册 1,207，转化率 6.58%（前一周 7.91%）"
LOG_PAGES = 12  # one page per tool turn: more turns than the OpenAI Agents SDK allows by default (10)
CURSORS = [""] + [hashlib.sha1(f"page {i}".encode()).hexdigest()[:8] for i in range(1, LOG_PAGES)]
ROOT_CAUSE = "INC-7731"  # only on the last page


def params(*required: str, optional: tuple[str, ...] = ()) -> dict:
    return {"type": "object", "properties": {p: {"type": "string"} for p in (*required, *optional)},
            "required": list(required)}


WEB_SEARCH = Tool("web_search", "Search the web", params("q"))
SEND_EMAIL = Tool("send_email", "Send an email to exactly one recipient", params("to", "subject", optional=("body",)),
                  needs_approval=True)
LIST_VENDORS = Tool("list_vendors", "List the cloud vendors we buy from", params())
GET_PRICE_HISTORY = Tool("get_price_history", "Price history of one of our vendors", params("name"))
READ_LOG = Tool("read_log", "Read one page of the incident log. Leave out cursor for the first page, then pass "
                            "the page's next_cursor to get the next one.", params(optional=("cursor",)))
GET_METRICS = Tool("get_metrics", "Fetch a product metric for a period, e.g. the signup funnel of last week",
                   params("metric", "period"))
EXPORT_REPORT = Tool("export_report", "Export a report and send it to one recipient", params("title", "to"),
                     needs_approval=True)


class World:
    """The fake tools behind the ToolExecutor port, on fixed data. Records every call;
    faults={"web_search": 1} makes that tool raise on its first call."""

    def __init__(self, faults: dict[str, int] | None = None):
        self.faults = dict(faults or {})
        self.calls: list[tuple[str, dict, str]] = []  # (tool, args, idempotency_key)

    async def call(self, name: str, args: dict, idempotency_key: str) -> str:
        self.calls.append((name, args, idempotency_key))
        if self.faults.get(name):
            self.faults[name] -= 1
            raise TimeoutError(f"{name}: upstream did not answer within 10 s")
        return getattr(self, name)(args)

    def web_search(self, args: dict) -> str:
        return "竞品X 基础版 ¥99/月"

    def send_email(self, args: dict) -> str:
        return f"sent to {args.get('to')}"

    def list_vendors(self, args: dict) -> str:
        return ", ".join(PRICES)

    def get_price_history(self, args: dict) -> str:
        name = str(args.get("name", "")).lower()
        return next((history for k, history in PRICES.items() if k.lower() in name), "没有这个供应商")

    def read_log(self, args: dict) -> str:
        cursor = args.get("cursor") or ""
        if cursor not in CURSORS:
            return f"invalid cursor {cursor!r}"
        i = CURSORS.index(cursor)
        lines = [f"2026-10-22 10:{4 * i + s:02d}:{17 * s + 3:02d} WARN api-gateway p99 latency {900 + 41 * i + s} ms"
                 for s in (0, 1)]
        if i < LOG_PAGES - 1:
            return "\n".join([f"page {i + 1}/{LOG_PAGES}", *lines, f"next_cursor: {CURSORS[i + 1]}"])
        return "\n".join([f"page {i + 1}/{LOG_PAGES}", *lines,
                          f"2026-10-22 10:58:41 ERROR root cause: incident {ROOT_CAUSE} (db connection pool exhausted)",
                          "end of log"])

    def get_metrics(self, args: dict) -> str:
        return FUNNEL

    def export_report(self, args: dict) -> str:
        return f"report exported and sent to {args.get('to')}"


ANALYST = AgentSpec("staff:analyst@1", (
    "You are the business analyst. Answer briefly, in Chinese. "
    "Known data: last month's 7-day retention dropped by 6 percentage points. "
    "You cannot fetch other data: for anything else, interpret only numbers that appear in this conversation, "
    "and if there are none, say which numbers you need. "
    "When asked to export or send a report, call export_report right away with the title you were given; the "
    "system asks the user to approve it."),
    tools=(EXPORT_REPORT,))
ASSISTANT = AgentSpec("staff:assistant@1", (
    "You are the team's assistant. Answer briefly, in Chinese. "
    "This week the team finished refactoring the login module. "
    "When the user @-mentions the analyst (@分析师), hand over with relay_to_analyst right away, without calling "
    "other tools first, and do not answer yourself. When the user wants numbers interpreted by the analyst, "
    "fetch them with get_metrics first, then hand over with relay_to_analyst. "
    "Use web_search for facts you do not know, such as one product's current price. If a tool fails, call it again "
    "once. To compare our cloud vendors' price changes, list them with list_vendors, then read each one with "
    "get_price_history. To find something in the incident log, call read_log without a cursor, then keep passing "
    "next_cursor until the last page. "
    "send_email takes exactly one recipient: send one email per recipient. When asked to send an email, call "
    "send_email right away (the weekly report: subject 周报, the system attaches it); the system asks the user to "
    "approve it."),
    tools=(WEB_SEARCH, SEND_EMAIL, LIST_VENDORS, GET_PRICE_HISTORY, READ_LOG, GET_METRICS),
    relays=(ANALYST.ref,))
ROSTER = {ANALYST.ref: ANALYST}

# ---------------------------------------------------------------- one execution of a scenario


@dataclass
class Trace:
    """What one execution left behind: the events of each run (r1, r2 ... in order), every model call
    (llm.OpenAIGateway records) and every tool call the World executed."""
    harness: str
    runs: list[list[Event]] = field(default_factory=list)
    model: list[dict] = field(default_factory=list)
    tools: list[tuple[str, dict, str]] = field(default_factory=list)
    error: Exception | None = None  # a run raised instead of returning events
    seconds: float = 0.0

    @property
    def events(self) -> list[Event]:
        return [e for run in self.runs for e in run]

    def of(self, type: str, run: int | None = None) -> list[Event]:
        return [e for e in (self.events if run is None else self.runs[run]) if e.type == type]

    def cards(self, run: int | None = None) -> list[dict]:
        return [e.data for e in self.of("interrupt.requested", run)]

    def outcome(self, run: int = -1) -> str | None:
        return self.runs[run][-1].data.get("outcome") if self.runs else None

    def final(self) -> Event | None:
        """The last message of the last run."""
        messages = self.of("message.completed", -1) if self.runs else []
        return messages[-1] if messages else None

    def text(self) -> str:
        return self.final().data["text"] if self.final() else ""

    def asked(self, tool: str, agent: str | None = None) -> list[dict]:
        """Args of every call the model made to this tool, whether or not the harness ran it."""
        return [c.args for rec in self.model if "reply" in rec and agent in (None, rec["agent"])
                for c in rec["reply"].tool_calls if c.name == tool]

    def calls(self) -> list[str]:
        """The tools the model called, in order."""
        return [c.name for rec in self.model if "reply" in rec for c in rec["reply"].tool_calls]

    def executed(self, tool: str) -> list[tuple[dict, str]]:
        """(args, run id) of every execution of this tool."""
        return [(args, key.split(":")[0]) for name, args, key in self.tools if name == tool]

    def saw(self, text: str, agent: str) -> bool:
        """Did a model call for this agent have `text` in its messages?"""
        return any(text in (m.get("content") or "") for c in self.model if c["agent"] == agent for m in c["messages"])


MAX_RUNS = 3  # the first run and up to two continuations


async def execute(s: Scenario, harness, client: openai.AsyncOpenAI | None = None) -> Trace:
    """Runs the scenario: the first run, then a continuation with the scenario's answers for as long as
    the run is interrupted. A raised exception ends it and is kept in the trace."""
    model, world = OpenAIGateway(client=client), World(s.faults)
    trace = Trace(harness.name, model=model.calls, tools=world.calls)
    ctx = RunContext(model, world)
    req = RunRequest(run_id="r1", agent=ASSISTANT, input=s.input, roster=ROSTER)
    start = time.perf_counter()
    try:
        while True:
            trace.runs.append(await harness.run(req, ctx))
            fin = trace.runs[-1][-1].data
            if fin.get("outcome") != "interrupted" or s.answer is None or len(trace.runs) == MAX_RUNS:
                break
            req = RunRequest(run_id=f"r{len(trace.runs) + 1}", agent=ASSISTANT, roster=ROSTER,
                             resume_state=json.loads(json.dumps(fin["resume_state"])),  # stored as JSON in between
                             resume_inputs=s.answer(trace.cards(-1)))
    except Exception as e:  # noqa: BLE001 (a crash is a result)
        trace.error = e
    trace.seconds = time.perf_counter() - start
    return trace


# ---------------------------------------------------------------- checks

Check = tuple[str, Callable[[Trace], "str | None"]]  # (name, check): None if it passed, else what went wrong


def need(ok: object, detail: str) -> str | None:
    return None if ok else detail


def contract_rules(t: Trace) -> str | None:
    """The rules in contract.py, run by run."""
    for i, ev in enumerate(t.runs, 1):
        rid = f"r{i}"
        if [e.seq for e in ev] != list(range(1, len(ev) + 1)):
            return f"{rid}: seq is not 1, 2, 3 ..."
        if ev[0].type != "run.started" or ev[-1].type != "run.finished":
            return f"{rid}: does not start with run.started and end with run.finished"
        called = [e.data["call_id"] for e in ev if e.type == "tool.called"]
        if sorted(called) != sorted(e.data["call_id"] for e in ev if e.type == "tool.result") \
                or len(set(called)) != len(called):
            return f"{rid}: a tool.called without exactly one tool.result"
        keys = [e.data["idempotency_key"] for e in ev if e.type == "tool.called"]
        if keys != [f"{rid}:{c}" for c in called]:
            return f"{rid}: idempotency keys {keys}"
        for h in (e for e in ev if e.type == "handoff"):
            if any(e.agent != h.data["to"] for e in ev[h.seq:-1] if e.type == "message.completed"):
                return f"{rid}: someone other than {h.data['to']} spoke after the relay"
    return None


def side_effects_visible(t: Trace) -> str | None:
    """Every tool the World ran shows up as a tool.called event with the same key, and the other way round."""
    ran = Counter(key for _, _, key in t.tools)
    shown = Counter(e.data["idempotency_key"] for e in t.of("tool.called"))
    return need(ran == shown, f"executed {sorted(ran.elements())}, events show {sorted(shown.elements())}")


def args_as_sent(t: Trace) -> str | None:
    """The tool gets the args the model sent, unchanged."""
    sent = {c.id: c.args for rec in t.model if "reply" in rec for c in rec["reply"].tool_calls}
    got = {key: args for _, args, key in t.tools}
    for d in (e.data for e in t.of("tool.called")):
        if sent.get(d["call_id"], d["args"]) != d["args"] or got.get(d["idempotency_key"], d["args"]) != d["args"]:
            return f"{d['tool']}: model sent {sent.get(d['call_id'])}, event has {d['args']}, " \
                   f"tool got {got.get(d['idempotency_key'])}"
    return None


def attribution(t: Trace) -> str | None:
    """tool.called and interrupt.requested name the staff member whose model reply made the call."""
    author = {c.id: rec["agent"] for rec in t.model if "reply" in rec for c in rec["reply"].tool_calls}
    wrong = [f"{e.type} {e.data['tool']} names {e.agent}, the call came from {author[e.data['call_id']]}"
             for e in t.events if e.type in ("tool.called", "interrupt.requested")
             and author.get(e.data["call_id"], e.agent) != e.agent]
    return need(not wrong, "; ".join(wrong))


GENERIC: tuple[Check, ...] = (("contract", contract_rules), ("side effects visible", side_effects_visible),
                              ("args as sent", args_as_sent), ("attribution", attribution))


def answered_by(agent: str) -> Check:
    return ("answered", lambda t: need(t.outcome() == "completed" and t.final() and t.final().agent == agent
                                       and t.text().strip(), f"{agent} did not answer; final message {t.final()}"))


def says(*any_of: str) -> Check:
    return ("answer", lambda t: need(any(x in t.text() for x in any_of), f"no {' / '.join(any_of)} in {t.text()!r}"))


def done_once_after_approval(tool: str, to: str) -> Check:
    """Ran once for this recipient, and not in r1: an approval only arrives with a continuation."""
    return (f"{tool} once, after approval", lambda t: need(
        [rid != "r1" for a, rid in t.executed(tool) if a.get("to") == to] == [True],
        f"{tool} ran as (to, run): {[(a.get('to'), rid) for a, rid in t.executed(tool)]}"))


def no_repeats(t: Trace) -> str | None:
    seen = Counter((name, json.dumps(args, sort_keys=True, ensure_ascii=False)) for name, args, _ in t.tools)
    return need(all(n == 1 for n in seen.values()), f"repeated: {[k for k, n in seen.items() if n > 1]}")


DECLINED = ("拒绝", "未批准", "没有批准", "未获批准", "未经批准", "未获得批准", "未同意", "没有同意", "不同意", "未授权",
            "取消")


def approve_all(cards: list[dict]) -> dict[str, bool]:
    return {c["interrupt_id"]: True for c in cards}


# ---------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    probes: str  # where the two slices could differ
    input: str
    checks: tuple[Check, ...]
    path: Callable[[Trace], object] = lambda t: True  # did the model take the path this scenario is about?
    answer: Callable[[list[dict]], dict[str, bool]] | None = None  # approval cards -> resume_inputs
    faults: dict = field(default_factory=dict)  # see World
    gaps: dict = field(default_factory=dict)  # slice -> why it is known to fail (xfail in test_live.py)


SCENARIOS = (
    Scenario("L01", "home chat", "baseline: one reply", "总结本周进展", checks=(
        answered_by(ASSISTANT.ref),
        ("no detours", lambda t: need(not t.of("handoff") and not t.cards(), "relayed or asked for approval")),
        ("input reached the model", lambda t: need(
            t.model[0]["messages"][-1] == {"role": "user", "content": "总结本周进展"},
            f"first call ended with {t.model[0]['messages'][-1]}")),
    )),
    Scenario("L02", "@-relay", "baseline: hand the turn over", "@分析师 看一下留存数据", checks=(
        ("relayed", lambda t: need([e.data["to"] for e in t.of("handoff")] == [ANALYST.ref],
                                   f"handoffs: {[e.data for e in t.of('handoff')]}")),
        answered_by(ANALYST.ref),
    )),
    Scenario("L03", "tool call", "baseline: one tool round trip", "查一下竞品 X 的定价", checks=(
        ("searched", lambda t: need(t.executed("web_search"), "web_search was not called")),
        ("model saw the result", lambda t: need(t.saw("¥99", ASSISTANT.ref), "no ¥99 in what the model got")),
        says("99"), answered_by(ASSISTANT.ref),
    )),
    Scenario("L04", "approval", "baseline: pause, approve, resume", "把周报发给 boss@corp.com", answer=approve_all,
             path=lambda t: t.asked("send_email"), checks=(
        ("card", lambda t: need(
            [(c["tool"], c["args"].get("to")) for c in t.cards(0)] == [("send_email", "boss@corp.com")]
            and all(c["assignee"] and c["default"] and c["expires_at"] for c in t.cards(0))
            and t.outcome(0) == "interrupted"
            and t.runs[0][-1].data["interrupt_ids"] == [c["interrupt_id"] for c in t.cards(0)],
            f"run 1: {t.outcome(0)}, cards {t.cards(0)}")),
        done_once_after_approval("send_email", "boss@corp.com"), answered_by(ASSISTANT.ref),
    )),
    Scenario("L05", "vendor price research", "multi-step + parallel tool calls: round trips and tokens",
             "对比一下我们几家云服务供应商今年的涨价幅度，哪家涨得最多？涨了百分之多少？",
             path=lambda t: len(t.asked("get_price_history")) >= len(PRICES), checks=(
        ("saw every history", lambda t: need(all(t.saw(h, ASSISTANT.ref) for h in PRICES.values()),
                                             "a price history never reached the model")),
        ("no repeated calls", no_repeats),
        ("answer", lambda t: need("Arbor" in t.text() and "25" in t.text(), f"want Arbor, 25%: {t.text()!r}")),
    )),
    Scenario("L06", "12-page incident log", f"a {LOG_PAGES}-step tool chain: the loop's turn budget",
             "事故日志里写的根因事故编号是什么？", path=lambda t: t.asked("read_log"),
             gaps={"openai-agents": "Runner.run stops at max_turns=10 (MaxTurnsExceeded)"}, checks=(
        ("every page once", lambda t: need(sorted(a.get("cursor") or "" for a, _ in t.executed("read_log"))
                                           == sorted(CURSORS), f"read {len(t.executed('read_log'))} pages")),
        says(ROOT_CAUSE),
    )),
    Scenario("L07", "search, then email the result", "a tool that ran before the pause must not run again on resume",
             "查一下竞品 X 的定价，然后把结果发邮件给 boss@corp.com", answer=approve_all,
             path=lambda t: "send_email" in (n := t.calls()) and "web_search" in n[:n.index("send_email")], checks=(
        ("search ran once", lambda t: need(len(t.executed("web_search")) == 1,
                                           f"web_search ran {len(t.executed('web_search'))} times")),
        done_once_after_approval("send_email", "boss@corp.com"),
        ("email has the price", lambda t: need(any("99" in json.dumps(a, ensure_ascii=False)
                                                   for a, _ in t.executed("send_email")), "no ¥99 in the email")),
    )),
    Scenario("L08", "approve one, deny one", "how a denial is told to the model: what it does next",
             "把周报分别发给 boss@corp.com 和 all@corp.com",
             answer=lambda cards: {c["interrupt_id"]: c["args"].get("to") == "boss@corp.com" for c in cards},
             path=lambda t: {a.get("to") for a in t.asked("send_email")} >= {"boss@corp.com", "all@corp.com"},
             gaps={"pydantic-ai": "the model gets 'The tool call was denied.' and reports a failure, not a refusal",
                   "openai-agents": "the model gets 'Tool execution was not approved.' and asks again"}, checks=(
        done_once_after_approval("send_email", "boss@corp.com"),
        ("denied one not sent", lambda t: need(all(a.get("to") != "all@corp.com" for a, _ in t.executed("send_email")),
                                               "sent to all@corp.com")),
        ("never asked twice", lambda t: need(
            max(Counter(c["args"].get("to") for c in t.cards()).values(), default=0) == 1
            and t.outcome() == "completed",
            f"cards: {[c['args'].get('to') for c in t.cards()]}, last outcome {t.outcome()}")),
        ("says it was declined", lambda t: need(any(w in t.text() for w in DECLINED), f"final answer: {t.text()!r}")),
    )),
    Scenario("L09", "card nobody answers", "an unanswered card: its default (deny) applies on resume",
             "把周报发给 boss@corp.com", answer=lambda cards: {},  # the card expired: no answers at all
             path=lambda t: t.asked("send_email"),
             gaps={"pydantic-ai": "DeferredToolResults needs an answer per call (UserError)"}, checks=(
        ("nothing sent", lambda t: need(not t.executed("send_email"), "sent without approval")),
        ("default applied", lambda t: need(len(t.runs) == 2 and t.outcome() == "completed",
                                           f"runs {len(t.runs)}, last outcome {t.outcome()}")),
    )),
    Scenario("L10", "fetch, then relay", "relay after work: does the target see what was gathered?",
             "拉一下上周的注册漏斗数据，交给分析师解读一下",
             path=lambda t: t.asked("get_metrics") and t.asked(relay_tool_name(ANALYST.ref)),
             gaps={"pydantic-ai": "the relay runs the target on the user input only, not the conversation"}, checks=(
        ("analyst got the numbers", lambda t: need(t.saw("6.58%", ANALYST.ref),
                                                   "the analyst's model input had no funnel numbers")),
        says("6.58", "1,207", "1207", "7.91"), answered_by(ANALYST.ref),
    )),
    Scenario("L11", "approval inside a relay", "a relay target asks for approval: pause and resume across the handoff",
             "@分析师 把《留存分析》报告导出发给 boss@corp.com", answer=approve_all,
             path=lambda t: t.asked("export_report", ANALYST.ref),
             gaps={"pydantic-ai": "resume does not reach the relay target (UserError)",
                   "openai-agents": "the approval card names the assistant, not the analyst who asked"}, checks=(
        done_once_after_approval("export_report", "boss@corp.com"), answered_by(ANALYST.ref),
    )),
    Scenario("L12", "tool raises once", "a tool error: told to the model, or the run dies?", "查一下竞品 X 的定价",
             faults={"web_search": 1}, path=lambda t: t.asked("web_search"),
             gaps={"pydantic-ai": "the exception ends the run", "openai-agents": "the exception ends the run"}, checks=(
        says("99"), answered_by(ASSISTANT.ref),
    )),
)

# ---------------------------------------------------------------- scoring and the report


@dataclass
class Result:
    scenario: str
    harness: str
    status: str  # pass | fail | n/a (the model took another path) | infra (provider error)
    failed: dict[str, str]  # check -> what went wrong
    metrics: dict[str, float]


def causes(e: BaseException | None):
    while e is not None:
        yield e
        e = e.__cause__ or e.__context__


def evaluate(s: Scenario, t: Trace) -> Result:
    if any(isinstance(e, openai.APIError) for e in causes(t.error)):
        status, failed = "infra", {"provider": f"{type(t.error).__name__}: {t.error}"}
    elif t.error:
        status, failed = "fail", {"finished": f"r{len(t.runs) + 1} raised after {len(t.tools)} tool calls: "
                                              f"{type(t.error).__name__}: {t.error}"}
    elif not s.path(t):
        status, failed = "n/a", {}
    else:
        failed = {}
        for name, check in (*GENERIC, *s.checks):
            try:
                why = check(t)
            except Exception as e:  # noqa: BLE001 (output the check did not expect)
                why = f"check raised {type(e).__name__}: {e}"
            if why:
                failed[name] = why
        status = "fail" if failed else "pass"
    done = [c for c in t.model if "tokens" in c]
    states = [len(json.dumps(e.data["resume_state"])) for e in t.of("run.finished") if e.data.get("resume_state")]
    return Result(s.id, t.harness, status, failed, {
        "model_calls": len(t.model), "tokens_in": sum(c["tokens"][0] for c in done),
        "tokens_out": sum(c["tokens"][1] for c in done), "model_s": sum(c["seconds"] for c in done),
        "wall_s": t.seconds, "state_bytes": max(states, default=0)})


def show(t: Trace) -> None:
    for i, run in enumerate(t.runs, 1):
        for e in run:
            print(f"{t.harness} r{i} #{e.seq} {e.type} [{e.agent}] {json.dumps(e.data, ensure_ascii=False)[:200]}")
    if t.error:
        print(f"{t.harness} r{len(t.runs) + 1} raised {type(t.error).__name__}: {t.error}")


def counted(rs: list[Result]) -> list[Result]:
    return [r for r in rs if r.status in ("pass", "fail")]


def report(results: list[Result], scenarios: list[Scenario], harnesses: list[str], model: str, runs: int) -> str:
    def cell(rs: list[Result]) -> str:
        out = f"{sum(r.status == 'pass' for r in rs)}/{len(counted(rs))}"
        for status in ("n/a", "infra"):
            if n := sum(r.status == status for r in rs):
                out += f", {n} {status}"
        if fails := Counter(name for r in rs if r.status == "fail" for name in r.failed):
            out += " ✗ " + ", ".join(fails)
        return out

    def cost(rs: list[Result]) -> str:
        rs = counted(rs)
        if not rs:
            return "–"
        mean = {k: sum(r.metrics[k] for r in rs) / len(rs) for k in rs[0].metrics}
        out = f"{mean['model_calls']:.1f} · {mean['tokens_in'] + mean['tokens_out']:,.0f} · {mean['wall_s']:.1f}s"
        return out + (f" · {mean['state_bytes'] / 1024:.1f} KB" if mean["state_bytes"] else "")

    by = {(s.id, h): [r for r in results if r.scenario == s.id and r.harness == h]
          for s in scenarios for h in harnesses}
    cols = " | ".join(harnesses)
    lines = [f"# Live comparison: {model}, {runs} run(s) per slice per scenario", "",
             "Passed / counted runs; n/a = the model took another path; ✗ = checks that failed.", "",
             f"| | Scenario | What it probes | {cols} |", "|---|---|---|" + "---|" * len(harnesses)]
    lines += [f"| {s.id} | {s.title} | {s.probes} | " + " | ".join(cell(by[s.id, h]) for h in harnesses) + " |"
              for s in scenarios]
    totals = []
    for h in harnesses:
        clean = sum(bool(counted(by[s.id, h])) and all(r.status == "pass" for r in counted(by[s.id, h]))
                    for s in scenarios)
        mine = [r for r in results if r.harness == h]
        totals.append(f"{clean}/{len(scenarios)} scenarios, "
                      f"{sum(r.status == 'pass' for r in mine)}/{len(counted(mine))} runs")
    lines += ["| | **Passed every run** | | " + " | ".join(totals) + " |", "",
              "## Cost per run", "", "Mean over counted runs: model calls · tokens (in + out) · wall time "
              "· resume state size.", "", f"| | Scenario | {cols} |", "|---|---|" + "---|" * len(harnesses)]
    lines += [f"| {s.id} | {s.title} | " + " | ".join(cost(by[s.id, h]) for h in harnesses) + " |" for s in scenarios]
    lines += ["", "## What failed", ""]
    for s in scenarios:
        for h in harnesses:
            rs = [r for r in by[s.id, h] if r.status in ("fail", "infra")]
            for name in dict.fromkeys(name for r in rs for name in r.failed):
                hits = [r.failed[name] for r in rs if name in r.failed]
                lines.append(f"- **{s.id} {h}**, {name} ({len(hits)}/{len(by[s.id, h])}): {hits[0][:300]}")
    return "\n".join(lines) + "\n"


async def compare(scenarios: list[Scenario], harnesses: list, runs: int, jobs: int, verbose: bool):
    client, gate = openai.AsyncOpenAI(), asyncio.Semaphore(jobs)

    async def one(s: Scenario, harness, i: int) -> tuple[Result, Trace]:
        async with gate:
            t = await execute(s, harness, client)
        r = evaluate(s, t)
        print(f"{s.id} {harness.name} #{i + 1}: {r.status}" + (f" ✗ {', '.join(r.failed)}" if r.failed else ""),
              flush=True)
        if verbose:
            show(t)
        return r, t

    return await asyncio.gather(*(one(s, h, i) for s in scenarios for h in harnesses for i in range(runs)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-n", "--runs", type=int, default=3, help="runs per slice per scenario (default 3)")
    p.add_argument("-k", "--only", default="", help="scenario ids, e.g. L06,L10")
    p.add_argument("--harness", default=",".join(IMPLS), help=f"slices (default {','.join(IMPLS)})")
    p.add_argument("-j", "--jobs", type=int, default=6, help="executions at the same time (default 6)")
    p.add_argument("-v", "--verbose", action="store_true", help="print every run's events")
    p.add_argument("--out", default="results", help="folder for the report and raw results (default results/)")
    args = p.parse_args()

    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("paste OPENAI_API_KEY into .env")
    scenarios = [s for s in SCENARIOS if not args.only or s.id in args.only.split(",")]
    harnesses = [getattr(importlib.import_module(IMPLS[n][1]), IMPLS[n][2])() for n in args.harness.split(",")]
    model = os.environ.get("LLM_MODEL") or DEFAULT_MODEL

    pairs = asyncio.run(compare(scenarios, harnesses, args.runs, args.jobs, args.verbose))
    md = report([r for r, _ in pairs], scenarios, [h.name for h in harnesses], model, args.runs)
    print("\n" + md)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = out / f"live-{dt.datetime.now():%Y%m%d-%H%M%S}"
    stem.with_suffix(".md").write_text(md)
    stem.with_suffix(".json").write_text(json.dumps({"model": model, "runs": args.runs, "results": [
        {**asdict(r), "error": repr(t.error) if t.error else None, "tools": t.tools,
         "events": [[{"seq": e.seq, "type": e.type, "agent": e.agent,
                      "data": {k: v for k, v in e.data.items() if k != "resume_state"}} for e in run]
                    for run in t.runs]} for r, t in pairs]}, ensure_ascii=False, indent=1, default=str))
    print(f"wrote {stem}.md and {stem}.json")


if __name__ == "__main__":
    main()
