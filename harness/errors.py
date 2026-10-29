"""Error codes shared by events, ports and the backend.

Every failure that reaches a client carries a code and whether a retry can help.
The codes below come from the four live paths plus the incidents behind E07-E12.
Adding a code is an additive (minor) change; renaming or removing one is breaking.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class ErrorCode(str, Enum):
    # model gateway (raised by the ModelGateway port after its own fallback is exhausted)
    MODEL_RATE_LIMITED = "model_rate_limited"        # 429 on every usable route (E11)
    MODEL_TIMEOUT = "model_timeout"                  # E11
    MODEL_UNAVAILABLE = "model_unavailable"          # one provider down, no fallback configured
    ALL_ROUTES_UNAVAILABLE = "all_routes_unavailable"  # every gateway dead (E09)
    CREDIT_EXHAUSTED = "credit_exhausted"            # provider key out of credit, 402 (E08)
    UNSUPPORTED_REQUEST = "unsupported_request"      # route cannot serve the request (E10)
    # budget / billing (BudgetGate)
    BUDGET_DENIED = "budget_denied"                  # workspace out of Captain credit (legacy credit_limit)
    # tools
    TOOL_FAILED = "tool_failed"
    TOOL_DENIED = "tool_denied"                      # permission or tenant scope
    # run lifecycle
    INVALID_REQUEST = "invalid_request"
    RESUME_INVALID = "resume_invalid"                # unknown interrupt id, incomplete resume, bad payload
    INTERRUPT_EXPIRED = "interrupt_expired"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


#: Default retryability per code. A producer may override per instance.
RETRYABLE_BY_DEFAULT: dict[ErrorCode, bool] = {
    ErrorCode.MODEL_RATE_LIMITED: True,
    ErrorCode.MODEL_TIMEOUT: True,
    ErrorCode.MODEL_UNAVAILABLE: True,
    ErrorCode.ALL_ROUTES_UNAVAILABLE: True,
    ErrorCode.CREDIT_EXHAUSTED: False,
    ErrorCode.UNSUPPORTED_REQUEST: False,
    ErrorCode.BUDGET_DENIED: False,
    ErrorCode.TOOL_FAILED: True,
    ErrorCode.TOOL_DENIED: False,
    ErrorCode.INVALID_REQUEST: False,
    ErrorCode.RESUME_INVALID: False,
    ErrorCode.INTERRUPT_EXPIRED: False,
    ErrorCode.MAX_STEPS_EXCEEDED: False,
    ErrorCode.CANCELLED: False,
    ErrorCode.INTERNAL: True,
}


class ErrorInfo(BaseModel):
    """The error object carried by run.finished and tool.result. No UI copy: the
    experience layer turns ``code`` (+ ``params``) into localized text."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    code: ErrorCode
    retryable: bool
    message: str = ""          # developer-facing, never shown to users
    route: str | None = None   # model route or tool that failed, for diagnosis
    params: dict = {}

    @classmethod
    def of(cls, code: ErrorCode, message: str = "", **kw) -> "ErrorInfo":
        return cls(code=code, retryable=kw.pop("retryable", RETRYABLE_BY_DEFAULT[code]), message=message, **kw)


class HarnessError(Exception):
    """Raised by ports. Implementations convert it into run.finished(failed) or tool.result(ok=false)."""

    def __init__(self, info: ErrorInfo):
        super().__init__(f"{info.code.value}: {info.message}")
        self.info = info


class GatewayError(HarnessError):
    pass


class BudgetDenied(HarnessError):
    pass


class NotSupported(Exception):
    """An optional contract capability the implementation does not declare."""
