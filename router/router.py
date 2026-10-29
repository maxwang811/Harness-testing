"""Minimal HarnessRouter, just enough to pin down E13's semantics offline. The real one is
Victor's (cut points) plus the harness_routes table; this is the executable spec for it.

Priority (split design): 1 internal header > 2 kill switch > 3.. rules > default legacy.
Rules come from the database through a 30-second cache; decisions are pinned per run.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Requester:
    user_id: str
    workspace_id: str
    is_internal: bool = False
    header_impl: str | None = None  # X-Harness header value


@dataclass(frozen=True)
class RouteDecision:
    primary: str
    shadows: tuple[str, ...] = ()
    rule_id: str = "default"


@dataclass(frozen=True)
class Rule:
    id: str
    scenario: str
    primary: str | None = None  # None = keep legacy as primary
    shadows: tuple[str, ...] = ()
    canary_workspaces: tuple[str, ...] = ()
    percent: int | None = None  # by stable workspace bucket


@dataclass
class RulesSnapshot:
    rules: list[Rule] = field(default_factory=list)
    kill_switch: set[str] = field(default_factory=set)  # scenarios forced back to legacy


LEGACY = {"assist": "legacy-assist", "delegate": "legacy-delegation", "home": "legacy-home", "oneshot": "legacy-oneshot"}


def bucket(workspace_id: str) -> int:
    return int(hashlib.sha256(workspace_id.encode()).hexdigest(), 16) % 100


class HarnessRouter:
    def __init__(self, load_rules: Callable[[], RulesSnapshot], now_ms: Callable[[], int], cache_ttl_ms: int = 30_000,
                 legacy: dict[str, str] | None = None):
        self._load = load_rules
        self._now = now_ms
        self._ttl = cache_ttl_ms
        self._snap: RulesSnapshot | None = None
        self._loaded_at = -10**12
        self.legacy = legacy or LEGACY
        self.pinned: dict[str, RouteDecision] = {}  # run_id -> decision, written to the run record

    def _rules(self) -> RulesSnapshot:
        if self._snap is None or self._now() - self._loaded_at >= self._ttl:
            self._snap, self._loaded_at = self._load(), self._now()
        return self._snap

    def route(self, run_id: str, scenario: str, who: Requester, continues_run_id: str | None = None) -> RouteDecision:
        if continues_run_id and continues_run_id in self.pinned:  # continuation / failover: same implementation
            d = self.pinned[continues_run_id]
            self.pinned[run_id] = RouteDecision(d.primary, (), "pinned:" + d.rule_id)
            return self.pinned[run_id]
        d = self._decide(scenario, who)
        self.pinned[run_id] = d
        return d

    def _decide(self, scenario: str, who: Requester) -> RouteDecision:
        legacy = self.legacy[scenario]
        if who.is_internal and who.header_impl:
            return RouteDecision(who.header_impl, (), "header")
        snap = self._rules()
        if scenario in snap.kill_switch:
            return RouteDecision(legacy, (), "kill_switch")
        for r in snap.rules:
            if r.scenario != scenario:
                continue
            if r.canary_workspaces and who.workspace_id not in r.canary_workspaces:
                continue
            if r.percent is not None and bucket(who.workspace_id) >= r.percent:
                continue
            return RouteDecision(r.primary or legacy, r.shadows, r.id)
        return RouteDecision(legacy, (), "default")
