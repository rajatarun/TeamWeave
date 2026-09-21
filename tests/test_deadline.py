"""A call must not be allowed to outlive the invocation that made it.

The worker's Lambda timeout was 300 s and its Bedrock client was configured to
wait 1800 s for one response. Six times the function's entire budget, which
makes the SDK's read timeout unreachable: Lambda kills the process first, so
every stall arrives as

    Sandbox.Timedout: Task timed out after 300.00 seconds

naming no step, no agent and no cause. That is what the first real pipeline
run ever attempted produced, and no amount of logging inside the worker would
have helped, because the worker never got to run its error path.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src.orchestrator import deadline

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_TEXT = (REPO / "infra" / "template.yaml").read_text()


class FakeContext:
    def __init__(self, remaining_ms):
        self._remaining = remaining_ms

    def get_remaining_time_in_millis(self):
        return self._remaining


@pytest.fixture(autouse=True)
def _clean():
    deadline.clear_deadline()
    yield
    deadline.clear_deadline()


def test_a_call_gets_less_than_the_invocation_has_left():
    deadline.set_deadline_from_context(FakeContext(120_000))
    budget = deadline.budget_for_call()
    assert budget < 120, "a call may not use the whole invocation"
    assert budget >= deadline.MIN_CALL_SECONDS


def test_the_reserve_is_what_lets_the_worker_report_the_failure():
    # Without it the handled error is raised too late to log which step hung.
    deadline.set_deadline_from_context(FakeContext(100_000))
    assert deadline.budget_for_call() <= 100 - deadline.RESERVE_SECONDS


def test_the_budget_shrinks_as_the_pipeline_spends_it():
    """Later agents get less, which is the point of reading the context.

    A client built once for the first agent would hand the last one a timeout
    computed when the function was fresh — and that call is exactly the one
    that outlives the invocation.
    """
    deadline.set_deadline_from_context(FakeContext(800_000))
    early = deadline.budget_for_call()
    deadline.set_deadline_from_context(FakeContext(60_000))
    late = deadline.budget_for_call()
    assert late < early


def test_almost_no_time_left_still_yields_a_usable_timeout():
    # A sub-second timeout would fail calls that were about to succeed.
    deadline.set_deadline_from_context(FakeContext(1_000))
    assert deadline.budget_for_call() == deadline.MIN_CALL_SECONDS


def test_a_negative_remaining_time_does_not_produce_a_negative_timeout():
    deadline.set_deadline_from_context(FakeContext(-5_000))
    assert deadline.budget_for_call() > 0


def test_with_no_deadline_set_it_falls_back_to_a_bounded_default():
    # Local runs and tests have no Lambda context; the default must still be
    # far below any Lambda timeout rather than the old 1800.
    assert deadline.remaining_seconds() is None
    assert 0 < deadline.budget_for_call() <= deadline.DEFAULT_CALL_SECONDS


def test_a_context_without_the_method_is_tolerated():
    deadline.set_deadline_from_context(object())
    assert deadline.remaining_seconds() is None
    assert deadline.budget_for_call() > 0


# ── The invariant, against the template ──────────────────────────────────────

def function_timeouts() -> dict[str, int]:
    """Each Serverless function's Timeout, from the template."""
    out: dict[str, int] = {}
    for match in re.finditer(r"^  (\w+):\n    Type: AWS::Serverless::Function\n", TEMPLATE_TEXT, re.M):
        name = match.group(1)
        block = TEMPLATE_TEXT[match.end(): match.end() + 2500]
        # Stop at the next top-level resource so a Timeout is not borrowed.
        block = re.split(r"\n  \w+:\n    Type: ", block)[0]
        found = re.search(r"^      Timeout: (\d+)$", block, re.M)
        if found:
            out[name] = int(found.group(1))
    return out


def test_the_scan_finds_the_worker():
    timeouts = function_timeouts()
    assert "WorkerFunction" in timeouts, f"found: {sorted(timeouts)}"
    assert len(timeouts) >= 3


def test_the_worker_can_outlast_a_multi_agent_pipeline():
    """One invocation runs every step, so its budget is the sum of them all.

    At 300 s a three-agent team could not finish. This is also the ceiling on
    how large a team the current architecture serves: past it, the pipeline
    needs a Step Functions state per step rather than a loop inside one
    function.
    """
    assert function_timeouts()["WorkerFunction"] >= 900


def test_no_call_timeout_exceeds_the_function_that_makes_it():
    # The invariant the 1800/300 mismatch broke. A read timeout larger than
    # the Lambda timeout is unreachable by construction.
    worker = function_timeouts()["WorkerFunction"]
    assert deadline.DEFAULT_CALL_SECONDS < worker, (
        f"a call may wait {deadline.DEFAULT_CALL_SECONDS}s inside a {worker}s function, "
        f"so Lambda kills the process before the SDK can raise"
    )


def test_no_hardcoded_timeout_outlives_the_worker():
    """Nothing in the orchestrator may configure a wait longer than the Lambda.

    Derived rather than listed, so a new client with a generous timeout is
    caught here instead of turning some future stall into Sandbox.Timedout.
    """
    worker = function_timeouts()["WorkerFunction"]
    offenders = []
    for path in (REPO / "src" / "orchestrator").rglob("*.py"):
        for match in re.finditer(r"read_timeout=(\d+)", path.read_text()):
            if int(match.group(1)) >= worker:
                offenders.append(f"{path.name}: read_timeout={match.group(1)}")
    assert not offenders, (
        f"these wait at least as long as the whole function lives, so their errors "
        f"are unreachable: {offenders}"
    )


def test_the_worker_records_the_deadline_before_doing_anything():
    """Without this call the whole mechanism is inert.

    `budget_for_call()` falls back to a default when no deadline is set, so
    removing the wiring breaks nothing visibly: calls still go out, bounded by
    a constant rather than by the time actually left. The pipeline's last step
    is then free to outlive the invocation again — the original bug, silently
    restored. Structural rather than a grep, so a call in a comment or a
    different function does not satisfy it.
    """
    import ast

    source = (REPO / "src" / "orchestrator" / "worker_handler.py").read_text()
    tree = ast.parse(source)
    handler = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "handler"),
        None,
    )
    assert handler is not None, "worker_handler has no handler() — has it been renamed?"

    calls = [
        n for n in ast.walk(handler)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "set_deadline_from_context"
    ]
    assert calls, "handler() never records the invocation deadline"
    assert calls[0].args, "set_deadline_from_context is called without the context"
