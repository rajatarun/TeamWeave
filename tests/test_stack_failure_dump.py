"""The on-failure dump has to describe *this* run.

Filtered only by status, it printed the previous three deploys' rollbacks into
the log of a run whose own deploy had succeeded. That reads exactly like the
current run failing, and sends you to debug something already fixed -- it cost
a full cycle on the first real failure of the agent-registry step, where the
deploy was fine and every AgentCore row shown was an hour old.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from scripts import dump_stack_failures as dump  # noqa: E402

NOW = datetime(2026, 9, 20, 18, 0, 0, tzinfo=timezone.utc)


def event(minutes: int, status: str, logical: str = "AgentCoreRuntime"):
    return {
        "Timestamp": NOW + timedelta(minutes=minutes),
        "ResourceStatus": status,
        "LogicalResourceId": logical,
        "ResourceType": "AWS::BedrockAgentCore::Runtime",
        "ResourceStatusReason": "because",
    }


def test_a_failure_from_an_earlier_run_is_not_reported():
    # The whole defect: an hour-old CREATE_FAILED shown under a green deploy.
    assert dump.failures_since([event(-90, "CREATE_FAILED")], NOW) == []


def test_a_failure_from_this_run_is_reported():
    assert len(dump.failures_since([event(+5, "CREATE_FAILED")], NOW)) == 1


def test_a_failure_exactly_at_the_start_counts():
    # The run's first event shares its start timestamp; excluding it would
    # drop the very failure that matters.
    assert len(dump.failures_since([event(0, "CREATE_FAILED")], NOW)) == 1


def test_successes_are_never_reported():
    events = [event(+5, "CREATE_COMPLETE"), event(+6, "UPDATE_IN_PROGRESS")]
    assert dump.failures_since(events, NOW) == []


@pytest.mark.parametrize("status", ["CREATE_FAILED", "UPDATE_FAILED", "DELETE_FAILED",
                                    "UPDATE_ROLLBACK_FAILED"])
def test_every_failure_status_is_caught(status):
    assert len(dump.failures_since([event(+1, status)], NOW)) == 1


def test_an_event_with_no_timestamp_is_kept_rather_than_dropped():
    # Losing a failure is worse than showing one extra.
    stray = {"ResourceStatus": "CREATE_FAILED", "LogicalResourceId": "X"}
    assert dump.failures_since([stray], NOW) == [stray]


# ── the --since value the workflow passes ──────────────────────────────────

@pytest.mark.parametrize("text", [
    "2026-09-20T18:00:00Z",          # what github.run_started_at looks like
    "2026-09-20T18:00:00+00:00",     # what the AWS CLI prints
    "2026-09-20T18:00:00.123456Z",
])
def test_the_timestamp_forms_that_actually_arrive_are_accepted(text):
    assert dump.parse_since(text) <= NOW + timedelta(seconds=1)


def test_a_naive_timestamp_does_not_crash_the_comparison():
    # CloudFormation timestamps are timezone-aware; comparing an aware to a
    # naive datetime raises TypeError, which would take out the one step whose
    # job is to explain a failure.
    since = dump.parse_since("2026-09-20T18:00:00")
    assert dump.failures_since([event(+5, "CREATE_FAILED")], since)


def test_an_empty_since_is_refused():
    with pytest.raises(ValueError):
        dump.parse_since("")


# ── the workflow wiring ────────────────────────────────────────────────────

def test_the_workflow_passes_the_run_start(tmp_path):
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml").read_text()
    assert "dump_stack_failures.py" in text
    assert "github.run_started_at" in text, "the dump must be scoped to this run"
