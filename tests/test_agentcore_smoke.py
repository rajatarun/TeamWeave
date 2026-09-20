"""The post-deploy check that the runtime actually serves a turn.

A green `sam deploy` says CloudFormation created the runtime. It says nothing
about whether the program inside it boots -- an entrypoint exposing no ASGI
app, a zip without the SDK, an execution role that cannot reach Bedrock all
fail at the first invocation and nowhere earlier. This script is what makes
the deploy find that out instead of the first pipeline run hours later.

What these pin is the part that is easy to get subtly wrong: the three
outcomes have to stay distinguishable. A broken runtime must fail the deploy;
a CI role that is not allowed to invoke must not, because that is the check
failing rather than the runtime; and neither may be reported as success.
"""
from __future__ import annotations

import io
import json
import os
from unittest import mock

import pytest
from botocore.exceptions import ClientError, UnknownServiceError

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from scripts import agentcore_smoke as smoke  # noqa: E402

ARN = "arn:aws:bedrock-agentcore:us-east-1:239571291755:runtime/teamweave_agent-X"


class FakeClient:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def invoke_agent_runtime(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0) if self._responses else self._responses
        if isinstance(item, Exception):
            raise item
        return item


def ok_response(body: dict, status: int = 200) -> dict:
    return {"statusCode": status, "response": io.BytesIO(json.dumps(body).encode())}


def client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "InvokeAgentRuntime")


def run(client, argv=("--runtime-arn", ARN, "--attempts", "2", "--delay", "0")):
    with mock.patch.object(smoke.boto3, "client", return_value=client), \
         mock.patch("sys.argv", ["agentcore_smoke.py", *argv]):
        return smoke.main()


def test_a_runtime_that_answers_passes():
    client = FakeClient(ok_response({"result": '{"ok": true}', "modelId": "us.amazon.nova-micro-v1:0"}))
    assert run(client) == 0


def test_a_broken_runtime_fails_the_step():
    # The literal contract with the workflow: a non-zero exit is what stops
    # the deploy, so it is asserted as a number rather than as the constant.
    client = FakeClient(ok_response({"result": ""}), ok_response({"result": ""}))
    assert run(client) != 0


def test_the_session_id_clears_the_service_minimum():
    # InvokeAgentRuntime's runtimeSessionId has a minimum length of 33. A
    # shorter one is a ValidationException at the API, so it would fail the
    # deploy while looking like a broken runtime.
    client = FakeClient(ok_response({"result": "ok"}))
    run(client)
    assert len(client.calls[0]["runtimeSessionId"]) >= 33


def test_the_prompt_carries_the_output_contract():
    # The same shape prompt_builder composes, so this exercises the path a
    # real turn takes rather than a shape only the smoke test ever sends.
    client = FakeClient(ok_response({"result": "ok"}))
    run(client)
    sent = json.loads(client.calls[0]["payload"].decode())
    assert "OUTPUT CONTRACT" in sent["prompt"]
    assert sent["sessionId"] == client.calls[0]["runtimeSessionId"]


def test_an_entrypoint_error_fails_the_deploy():
    # The entrypoint returns {"error": ..., "result": ""} with HTTP 200 for a
    # payload it cannot read. Checking only the status code would pass a
    # runtime that rejects every request.
    client = FakeClient(ok_response({"error": "payload contained no prompt", "result": ""}))
    assert run(client) == smoke.EXIT_BROKEN


def test_an_error_beside_a_non_empty_result_still_fails():
    # The case the previous test cannot see: with an empty result, the result
    # check alone catches it, so deleting the error check entirely broke
    # nothing. An error reported *alongside* text is what isolates it -- and
    # is the worse failure, because that text would otherwise be accepted as
    # the agent's answer.
    client = FakeClient(ok_response({"error": "model refused", "result": "I cannot help"}))
    assert run(client) == smoke.EXIT_BROKEN


def test_an_empty_result_fails_the_deploy():
    client = FakeClient(ok_response({"result": "   "}))
    assert run(client) == smoke.EXIT_BROKEN


def test_an_http_error_fails_the_deploy():
    client = FakeClient(ok_response({"result": "ok"}, status=502), ok_response({"result": "ok"}, status=502))
    assert run(client) == smoke.EXIT_BROKEN


def test_a_runtime_that_never_answers_fails_the_deploy():
    client = FakeClient(client_error("ThrottlingException"), client_error("ThrottlingException"))
    assert run(client) == smoke.EXIT_BROKEN


def test_a_cold_runtime_is_retried():
    # Invoking a resource the same deploy just created is the one case where
    # "not ready yet" is real and temporary.
    client = FakeClient(client_error("ResourceNotReadyException"), ok_response({"result": "ok"}))
    assert run(client) == smoke.EXIT_OK
    assert len(client.calls) == 2


@pytest.mark.parametrize("code", ["AccessDeniedException", "UnrecognizedClientException", "ExpiredTokenException"])
def test_a_ci_permission_gap_does_not_fail_the_deploy(code, capsys):
    # This is the check failing, not the runtime. Failing every deploy on it
    # would be wrong -- but it is warned about, never reported as a pass.
    # EXIT_CANNOT_VERIFY and EXIT_OK are both 0, so the exit code alone cannot
    # tell them apart: what must differ is what the run says it proved.
    client = FakeClient(client_error(code))
    # Literal 0, not EXIT_CANNOT_VERIFY: asserting against the constant just
    # follows it wherever it is set, so it cannot catch the constant being
    # changed to a failing code.
    assert run(client) == 0
    output = capsys.readouterr()
    assert "served a turn" not in output.out
    # On stdout, and with the ::warning:: prefix. GitHub parses workflow
    # commands from stdout only, so writing this to stderr produced no
    # annotation at all -- in a step that exits 0. A passing step with no
    # visible result is exactly what this script exists to prevent.
    assert "::warning::" in output.out
    assert "NOT VERIFIED" in output.out


def test_a_permission_gap_is_not_retried():
    client = FakeClient(client_error("AccessDeniedException"), ok_response({"result": "ok"}))
    run(client)
    assert len(client.calls) == 1, "retrying cannot grant a permission"


def test_a_permission_gap_says_so_loudly(capsys):
    run(FakeClient(client_error("AccessDeniedException")))
    assert "::warning::" in capsys.readouterr().out


def test_a_pass_is_announced_so_the_run_page_can_be_read_without_the_log(capsys):
    # The success path needs an annotation too, or "did it actually invoke?"
    # is only answerable by paging through the job log.
    client = FakeClient(ok_response({"result": "ok", "modelId": "us.amazon.nova-micro-v1:0"}))
    assert run(client) == 0
    out = capsys.readouterr().out
    assert "::notice::" in out
    assert "served a turn" in out


def test_a_broken_runtime_is_announced_as_an_error(capsys):
    client = FakeClient(ok_response({"error": "boom", "result": "text"}))
    assert run(client) != 0
    assert "::error::" in capsys.readouterr().out


def test_the_outcome_reaches_the_step_summary(tmp_path, monkeypatch):
    # So the run page states the result without anyone opening the log.
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    run(FakeClient(ok_response({"result": "ok"})))
    assert "served a turn" in summary.read_text()


def test_an_unwritable_summary_never_fails_the_deploy(monkeypatch):
    # Reporting must not become the thing that breaks the build.
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", "/proc/nonexistent/summary.md")
    assert run(FakeClient(ok_response({"result": "ok"}))) == 0


def test_a_botocore_without_the_service_cannot_verify():
    exc = UnknownServiceError(service_name="bedrock-agentcore", known_service_names="s3")
    with mock.patch.object(smoke.boto3, "client", side_effect=exc), \
         mock.patch("sys.argv", ["agentcore_smoke.py", "--runtime-arn", ARN]):
        assert smoke.main() == smoke.EXIT_CANNOT_VERIFY


def test_cannot_verify_and_broken_are_told_apart():
    assert smoke.cannot_verify_reason(client_error("AccessDeniedException"))
    assert not smoke.cannot_verify_reason(client_error("ThrottlingException"))
    assert not smoke.cannot_verify_reason(RuntimeError("no response body"))
