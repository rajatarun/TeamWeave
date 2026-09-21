"""How long this invocation has left, so a call cannot outlive its caller.

The worker ran the whole pipeline with a 300 s Lambda timeout while its
Bedrock client was configured to wait 1800 s for a single response. Six times
the function's entire budget. That is not a slow path, it is an unreachable
one: the SDK can never raise a read timeout, because Lambda kills the process
first. Every stall therefore arrived as

    Sandbox.Timedout: Task timed out after 300.00 seconds

which names no step, no agent and no cause -- and that is what a real pipeline
run produced the first time one was ever attempted.

Giving each call a slice of the time that actually remains means a stall
raises `ReadTimeoutError` with enough left for the worker to say which step
hung and fail properly. The rule is simply that a client's read timeout must
fit inside its caller's remaining budget, and `tests/test_deadline.py` holds
the template to it.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from .logger import get_logger

log = get_logger("deadline")

# Left for the worker to log the failure, write the step record and return a
# StepFailed that names the step. Without it the handled error is raised so
# late that Lambda kills the process during the handling.
RESERVE_SECONDS = 20.0

# Floor: below this a call is not worth starting, and a one-second timeout
# would fail calls that were about to succeed.
MIN_CALL_SECONDS = 10.0

# Used when nothing has told us the deadline -- a local run, a test, or a
# caller that never set one. Deliberately well under any Lambda timeout here.
DEFAULT_CALL_SECONDS = float(os.environ.get("AGENT_INVOKE_READ_TIMEOUT_SECONDS", "240"))

_deadline_epoch: Optional[float] = None


def set_deadline_from_context(context: Any) -> None:
    """Record when this invocation must be finished.

    Lambda's context knows exactly how long is left, which is better than any
    configured constant: it accounts for time already spent on earlier steps
    of the same pipeline.
    """
    global _deadline_epoch
    remaining_ms = getattr(context, "get_remaining_time_in_millis", None)
    if not callable(remaining_ms):
        _deadline_epoch = None
        return
    try:
        _deadline_epoch = time.monotonic() + (float(remaining_ms()) / 1000.0)
    except (TypeError, ValueError):
        _deadline_epoch = None


def clear_deadline() -> None:
    global _deadline_epoch
    _deadline_epoch = None


def remaining_seconds() -> Optional[float]:
    """Seconds left in this invocation, or None if nothing set a deadline."""
    if _deadline_epoch is None:
        return None
    return _deadline_epoch - time.monotonic()


def budget_for_call() -> float:
    """How long one outbound call may wait for a response.

    Never more than the time that remains, so the SDK's own error wins the
    race against Lambda's kill -- which is the whole point.
    """
    remaining = remaining_seconds()
    if remaining is None:
        return max(DEFAULT_CALL_SECONDS, MIN_CALL_SECONDS)
    usable = remaining - RESERVE_SECONDS
    if usable < MIN_CALL_SECONDS:
        log.warning(
            "agent_call_budget_at_floor; the invocation is nearly out of time",
            extra={"remaining_s": round(remaining, 1), "using_s": MIN_CALL_SECONDS},
        )
        return MIN_CALL_SECONDS
    return min(usable, DEFAULT_CALL_SECONDS)
