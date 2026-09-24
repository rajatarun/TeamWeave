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


# ── the dump must never be the thing that fails ────────────────────────────

def test_an_unusable_since_degrades_instead_of_failing(monkeypatch, capsys):
    """`${{ github.run_started_at }}` expanded to nothing and this returned 2.

    So the one run that most needed a diagnosis got a second red step and no
    diagnosis at all: reporting became the failure. A window it had to guess
    at is worth far more than an exit code.
    """
    from unittest import mock

    cfn = mock.Mock()
    cfn.describe_stacks.return_value = {"Stacks": [{"StackStatus": "UPDATE_COMPLETE"}]}
    cfn.get_paginator.return_value.paginate.return_value = [{"StackEvents": []}]
    monkeypatch.setattr(dump.boto3, "client", lambda *a, **kw: cfn)
    monkeypatch.setattr(dump.sys, "argv", ["dump", "some-stack", "--since", ""])

    assert dump.main() == 0
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "may include an earlier run" in out, "a guessed window has to say it guessed"


def test_a_usable_since_does_not_warn(monkeypatch, capsys):
    from unittest import mock

    cfn = mock.Mock()
    cfn.describe_stacks.return_value = {"Stacks": [{"StackStatus": "UPDATE_COMPLETE"}]}
    cfn.get_paginator.return_value.paginate.return_value = [{"StackEvents": []}]
    monkeypatch.setattr(dump.boto3, "client", lambda *a, **kw: cfn)
    monkeypatch.setattr(dump.sys, "argv", ["dump", "s", "--since", "2026-09-20T18:00:00Z"])

    assert dump.main() == 0
    assert "::warning::" not in capsys.readouterr().out


def test_the_workflow_takes_its_own_timestamp(): 
    from pathlib import Path
    import yaml
    wf = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml").read_text()
    )
    steps = wf["jobs"]["deploy"]["steps"]
    names = [s.get("name") for s in steps]
    # Recorded in the job rather than read from a context that expanded empty.
    assert "Record when this deploy started" in names
    recorder = next(s for s in steps if s.get("name") == "Record when this deploy started")
    assert "DEPLOY_STARTED_AT" in recorder["run"] and "GITHUB_ENV" in recorder["run"]
    dump_step = next(s for s in steps if s.get("name") == "Dump CloudFormation events on failure")
    assert "${DEPLOY_STARTED_AT}" in dump_step["run"]
    assert "github.run_started_at" not in dump_step["run"]
    assert names.index("Record when this deploy started") < names.index("SAM Deploy")


# ── early validation has no stack event ────────────────────────────────────

def test_a_validation_error_from_an_earlier_change_set_is_not_reported():
    old = NOW - timedelta(hours=3)
    summaries = [{
        "Status": "FAILED",
        "ChangeSetName": "samcli-deploy-old",
        "CreationTime": old,
        "StatusReason": "AWS::EarlyValidation::ResourceExistenceCheck",
    }]
    events = [{
        "EventId": "old",
        "EventType": "VALIDATION_ERROR",
        "Timestamp": old,
        "LogicalResourceId": "HealthKbProvisionLogGroup",
        "ValidationStatusReason": "already exists",
    }]
    assert dump.failed_change_sets_since(summaries, NOW) == []
    assert dump.validation_events_since(events, NOW) == []


def test_a_create_failed_stack_event_is_not_treated_as_early_validation():
    # DescribeEvents with FailedEvents also returns provisioning failures.
    # Those already print from the stack stream; printing them again would
    # look like a second failure.
    event = {
        "EventType": "STACK_EVENT",
        "ResourceStatus": "CREATE_FAILED",
        "LogicalResourceId": "HealthKnowledgeBase",
        "ResourceStatusReason": "parameter validation failed",
        "Timestamp": NOW,
    }
    assert dump.is_early_validation(event) is False
    assert dump.validation_events_since([event], NOW) == []


def _client(pages_by_operation, calls=None):
    from unittest import mock

    cfn = mock.Mock()
    cfn.describe_stacks.return_value = {
        "Stacks": [{"StackStatus": "UPDATE_ROLLBACK_COMPLETE"}]
    }

    def get_paginator(operation):
        paginator = mock.Mock()

        def paginate(**kwargs):
            if calls is not None:
                calls.append((operation, kwargs))
            return pages_by_operation.get(operation, [{}])

        paginator.paginate.side_effect = paginate
        return paginator

    cfn.get_paginator.side_effect = get_paginator
    return cfn


def test_a_failed_changeset_names_the_resource_when_no_stack_event_does(monkeypatch, capsys):
    """The SAM waiter only prints the hook name. The resource is on DescribeEvents."""
    validation = {
        "EventId": "evt-1",
        "EventType": "VALIDATION_ERROR",
        "Timestamp": NOW + timedelta(minutes=5),
        "LogicalResourceId": "HealthKbProvisionLogGroup",
        "ResourceType": "AWS::Logs::LogGroup",
        "PhysicalResourceId": "/aws/lambda/tarun-content-team-HealthKbProvisionFunction",
        "ValidationName": "AWS::EarlyValidation::ResourceExistenceCheck",
        "ValidationStatus": "FAILED",
        "ValidationStatusReason": (
            "Resource of type 'AWS::Logs::LogGroup' with identifier "
            "'/aws/lambda/tarun-content-team-HealthKbProvisionFunction' already exists."
        ),
        "ValidationPath": "/Resources/HealthKbProvisionLogGroup/Properties/LogGroupName",
    }
    # Returned for the stack read and again for the change set. Same EventId,
    # so the resource is printed once.
    calls = []
    cfn = _client({
        "describe_stack_events": [{"StackEvents": []}],
        "list_change_sets": [{"Summaries": [{
            "Status": "FAILED",
            "ChangeSetName": "samcli-deploy-230",
            "ChangeSetId": "arn:aws:cloudformation:us-east-1:1:changeSet/samcli-deploy-230/abc",
            "CreationTime": NOW + timedelta(minutes=4),
            "StatusReason": (
                "The following hook(s)/validation failed: "
                "[AWS::EarlyValidation::ResourceExistenceCheck]"
            ),
        }]}],
        "describe_events": [{"OperationEvents": [validation]}],
    }, calls)
    monkeypatch.setattr(dump.boto3, "client", lambda *a, **kw: cfn)
    monkeypatch.setattr(dump.sys, "argv", [
        "dump", "tarun-content-team", "--since", "2026-09-20T18:00:00Z", "--region", "us-east-1",
    ])

    assert dump.main() == 0
    out = capsys.readouterr().out
    assert "No resource failed in this run" not in out
    assert "HealthKbProvisionLogGroup" in out
    assert "AWS::Logs::LogGroup" in out
    assert "AWS::EarlyValidation::ResourceExistenceCheck" in out
    assert "already exists" in out
    assert "/Resources/HealthKbProvisionLogGroup/Properties/LogGroupName" in out
    assert out.count("already exists") == 1
    change_set_reads = [kwargs for operation, kwargs in calls if kwargs.get("ChangeSetName")]
    assert change_set_reads, "validation events for a failed changeset are not on the stack stream"
    assert change_set_reads[0]["Filters"] == {"FailedEvents": True}
    assert change_set_reads[0]["ChangeSetName"].endswith("/abc")


def test_describe_events_access_denied_is_printed_and_does_not_fail(monkeypatch, capsys):
    from unittest import mock
    from botocore.exceptions import ClientError

    denied = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "not allowed to DescribeEvents"}},
        "DescribeEvents",
    )
    cfn = mock.Mock()
    cfn.describe_stacks.return_value = {"Stacks": [{"StackStatus": "UPDATE_ROLLBACK_COMPLETE"}]}

    def get_paginator(operation):
        paginator = mock.Mock()
        if operation == "describe_events":
            paginator.paginate.side_effect = denied
        elif operation == "list_change_sets":
            paginator.paginate.return_value = [{"Summaries": [{
                "Status": "FAILED",
                "ChangeSetName": "samcli-deploy-230",
                "CreationTime": NOW + timedelta(minutes=1),
                "StatusReason": "The following hook(s)/validation failed: [AWS::EarlyValidation::ResourceExistenceCheck]",
            }]}]
        else:
            paginator.paginate.return_value = [{"StackEvents": []}]
        return paginator

    cfn.get_paginator.side_effect = get_paginator
    monkeypatch.setattr(dump.boto3, "client", lambda *a, **kw: cfn)
    monkeypatch.setattr(dump.sys, "argv", [
        "dump", "tarun-content-team", "--since", "2026-09-20T18:00:00Z", "--region", "us-east-1",
    ])

    assert dump.main() == 0
    out = capsys.readouterr().out
    assert "AccessDenied" in out
    assert "describe-events" in out
    assert "--stack-name tarun-content-team" in out
    assert "No resource failed in this run" not in out
    assert "ResourceExistenceCheck" in out
