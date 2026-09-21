"""The id a caller is handed must be the id it can poll back.

`POST /team/task` answers 202 with a `run_id`; `GET /team/task/{run_id}`
resolves it by rebuilding the Step Functions execution ARN from it. Those two
halves live in different modules and nothing held them to each other, so the
trigger minted a uuid, started an execution *without naming it*, and handed
back an id that named no execution at all. Every poll answered

    404 {"error": "run_id not found"}

while the pipeline itself ran perfectly. A2A's `message:send` / `tasks/{id}`
pair had the same split.

The fake here models the one Step Functions behaviour that makes this a bug:
an unnamed execution gets a uuid **of the service's own choosing**, not the
one in the input. A fake that echoes a fixed ARN -- which is what the existing
trigger tests use -- reports success either way, which is precisely why the
defect survived a green suite and a green deploy.

These tests drive the real handlers end to end rather than asserting on source
text, so they fail if the round trip breaks by any route, including one that
still passes `name=`.
"""
from __future__ import annotations

import json
import os
import uuid

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from botocore.exceptions import ClientError  # noqa: E402

STATE_MACHINE_ARN = "arn:aws:states:us-east-1:239571291755:stateMachine:tarun-content-team-state-machine"
EXECUTION_PREFIX = STATE_MACHINE_ARN.replace(":stateMachine:", ":execution:", 1)


class FakeSfn:
    """Step Functions, as far as naming and lookup are concerned."""

    def __init__(self):
        self.executions = {}   # executionArn -> description
        self.starts = []

    def start_execution(self, stateMachineArn, input, name=None, **_kw):
        self.starts.append({"stateMachineArn": stateMachineArn, "input": input, "name": name})
        # The service invents its own name when the caller supplies none. It
        # is never the run_id buried in the input -- that is the whole defect.
        effective = name or str(uuid.uuid4())
        if effective in {arn.rsplit(":", 1)[-1] for arn in self.executions}:
            raise ClientError(
                {"Error": {"Code": "ExecutionAlreadyExists", "Message": effective}},
                "StartExecution",
            )
        arn = f"{stateMachineArn.replace(':stateMachine:', ':execution:', 1)}:{effective}"
        self.executions[arn] = {"status": "RUNNING", "input": input}
        return {"executionArn": arn, "startDate": 0}

    def describe_execution(self, executionArn, **_kw):
        try:
            return dict(self.executions[executionArn], executionArn=executionArn)
        except KeyError:
            raise ClientError(
                {"Error": {"Code": "ExecutionDoesNotExist",
                           "Message": f"Execution Does Not Exist: '{executionArn}'"}},
                "DescribeExecution",
            )

    def succeed(self, arn, output):
        self.executions[arn] = {"status": "SUCCEEDED", "output": json.dumps(output)}


@pytest.fixture
def sfn(monkeypatch):
    from src.orchestrator import a2a_handler, status_handler, trigger_handler

    fake = FakeSfn()
    monkeypatch.setenv("STATE_MACHINE_ARN", STATE_MACHINE_ARN)
    monkeypatch.setattr(trigger_handler, "sfn", fake)
    monkeypatch.setattr(status_handler, "sfn", fake)
    monkeypatch.setattr(a2a_handler, "sfn", fake)
    return fake


def _start_run(team="tarun_visibility_team"):
    from src.orchestrator import trigger_handler

    event = {
        "httpMethod": "POST",
        "path": "/team/task",
        "body": json.dumps({"team": team, "version": "v1",
                            "request": {"topic": "anything"}}),
    }
    response = trigger_handler.handler(event, None)
    assert response["statusCode"] == 202, response
    return json.loads(response["body"])


def _poll(run_id):
    from src.orchestrator import status_handler

    return status_handler.handler({"httpMethod": "GET", "pathParameters": {"run_id": run_id}}, None)


# ── the round trip ───────────────────────────────────────────────────────────

def test_returned_run_id_resolves_to_a_real_execution(sfn):
    body = _start_run()
    response = _poll(body["run_id"])
    assert response["statusCode"] == 200, (
        "the run_id handed to the caller did not name an execution: "
        f"{response['body']}"
    )
    assert json.loads(response["body"])["status"] == "RUNNING"


def test_poll_returns_the_result_of_that_run(sfn):
    body = _start_run()
    arn, = sfn.executions
    sfn.succeed(arn, {"steps": {"editor": {"post": "the finished thing"}}})

    payload = json.loads(_poll(body["run_id"])["body"])
    assert payload["status"] == "SUCCEEDED"
    assert payload["result"]["steps"]["editor"]["post"] == "the finished thing"


def test_execution_is_named_with_the_run_id_the_caller_is_given(sfn):
    body = _start_run()
    assert sfn.starts[0]["name"] == body["run_id"], (
        "start_execution must name the execution, or Step Functions picks a "
        "uuid the caller was never told"
    )


def test_run_id_matches_the_execution_id_and_the_payload(sfn):
    """The three ids the 202 reports are one id, not three."""
    body = _start_run()
    assert body["state_fn_execution_id"] == body["run_id"]
    assert body["state_fn_execution_arn"] == f"{EXECUTION_PREFIX}:{body['run_id']}"
    assert json.loads(sfn.starts[0]["input"])["run_id"] == body["run_id"]


def test_an_unknown_run_id_is_still_a_404(sfn):
    """The 404 must keep working -- the fix is not to stop reporting misses."""
    response = _poll(str(uuid.uuid4()))
    assert response["statusCode"] == 404
    assert json.loads(response["body"])["error"] == "run_id not found"


def test_run_id_is_a_legal_step_functions_execution_name(sfn):
    from src.orchestrator import run_ids

    body = _start_run()
    assert run_ids.is_valid_execution_name(body["run_id"])
    assert len(body["run_id"]) <= run_ids.EXECUTION_NAME_MAX


# ── the same round trip over A2A ─────────────────────────────────────────────

def test_a2a_task_id_resolves_to_a_real_execution(sfn):
    from src.orchestrator import a2a_handler

    sent = a2a_handler.handler({
        "httpMethod": "POST",
        "path": "/a2a/v1/message:send",
        "body": json.dumps({"message": {"parts": [{"text": "write something"}]}}),
    }, None)
    assert sent["statusCode"] == 200, sent
    task_id = json.loads(sent["body"])["task"]["id"]

    fetched = a2a_handler.handler({
        "httpMethod": "GET",
        "path": f"/a2a/v1/tasks/{task_id}",
        "pathParameters": {"task_id": task_id},
    }, None)
    assert fetched["statusCode"] == 200, (
        f"A2A handed out a task id it cannot resolve: {fetched['body']}"
    )
