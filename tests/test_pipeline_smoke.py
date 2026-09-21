"""The deploy's end-to-end check: did a team actually run and produce something?

Every other gate in this repository answers a narrower question. The suite says
the code is consistent, cfn-lint says the template is well formed, `sam deploy`
says CloudFormation accepted the resources, and `agentcore_smoke.py` says one
runtime boots. None of them says a person can ask a team to do something and
get an answer, which is the only thing the platform is for.

The case worth the most care is the **empty success**: a run that reaches
SUCCEEDED with nothing in its final step. Every signal the platform emits says
that worked. It is the exact shape of the A2A card that served `"skills": []`
and the IPv6 discovery that matched nothing — so here it is a failure.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location(
        "pipeline_smoke", REPO / "scripts" / "pipeline_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_final_step_is_the_last_one(smoke):
    # The deliverable is the end of the pipeline — the formatter, the writer —
    # not the first step's working notes.
    result = {"steps": {"analyzer": {"a": 1}, "rewriter": {"b": 2}, "formatter": {"doc": "final"}}}
    step_id, output = smoke.final_step_output(result)
    assert step_id == "formatter"
    assert output == {"doc": "final"}


def test_no_steps_means_no_final_output(smoke):
    assert smoke.final_step_output({}) == ("", None)
    assert smoke.final_step_output({"steps": {}}) == ("", None)
    assert smoke.final_step_output({"steps": "not a dict"}) == ("", None)


@pytest.mark.parametrize("empty", [None, "", "   ", {}, []])
def test_an_empty_result_is_not_a_pass(smoke, empty):
    """The failure that hides best.

    A pipeline can reach SUCCEEDED having produced nothing: every status is
    green, the deploy passes, and the person who asked gets an empty box.
    """
    assert smoke.is_substantive(empty) is False


@pytest.mark.parametrize("real", [{"document": "text"}, ["a"], "written", 0, False])
def test_real_output_passes(smoke, real):
    # 0 and False are content: a step that legitimately returns them has
    # produced an answer, and treating falsiness as emptiness would fail a
    # working run.
    assert smoke.is_substantive(real) is True


def test_the_smoke_request_satisfies_the_teams_schema(smoke):
    """The canned request must fill the team's required fields.

    A smoke test that starts a run with a field the pipeline needs left empty
    exercises the machinery and proves nothing about the output.
    """
    import json

    team_path = REPO / "config" / "examples" / "teams" / smoke.DEFAULT_TEAM / "v1" / "team.json"
    config = json.loads(team_path.read_text())
    required = {
        f["name"] for f in config["request_schema"]["fields"] if f.get("required")
    }
    missing = sorted(required - set(smoke.DEFAULT_REQUEST))
    assert not missing, f"the smoke request omits required field(s): {missing}"
    for name in required:
        assert smoke.DEFAULT_REQUEST[name].strip(), f"{name} is blank in the smoke request"


def test_the_smoke_team_exists(smoke):
    assert (REPO / "config" / "examples" / "teams" / smoke.DEFAULT_TEAM).is_dir()


def test_workflow_commands_go_to_stdout(smoke, capsys):
    # GitHub Actions parses ::error:: and ::warning:: from stdout only. On
    # stderr they produce no annotation at all, which is how an earlier
    # cannot-verify path went silent in a step that exited 0.
    smoke.announce("warning", "something worth seeing")
    captured = capsys.readouterr()
    assert "::warning::something worth seeing" in captured.out
    assert captured.err == ""


def test_the_deploy_runs_it():
    workflow = (REPO / ".github" / "workflows" / "deploy.yml").read_text()
    assert "scripts/pipeline_smoke.py" in workflow, "the check exists but never runs"


def test_it_runs_after_the_configs_it_exercises_are_uploaded():
    """Order matters: run first and it tests the previous deploy's configs."""
    workflow = (REPO / ".github" / "workflows" / "deploy.yml").read_text()
    sync = workflow.index("Sync team configs to S3")
    register = workflow.index("Register agents and AgentCore release channels")
    smoke_at = workflow.index("Run one real team pipeline")
    assert sync < smoke_at, "the pipeline check runs before the team configs are uploaded"
    assert register < smoke_at, "the pipeline check runs before the agents are registered"


# ── The decision itself ──────────────────────────────────────────────────────
# The helpers above can all be right and the script still wave a failed run
# through, because what fails the deploy is main()'s exit code. These drive it
# against a fabricated Step Functions and assert what it decides.

import json as _json  # noqa: E402


class FakeSfn:
    """Enough of the Step Functions client for main() to run against."""

    def __init__(self, status, output, *, start_error=None):
        self._status = status
        self._output = output
        self._start_error = start_error
        self.started = []

    def start_execution(self, **kwargs):
        if self._start_error:
            raise self._start_error
        self.started.append(kwargs)
        return {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:run-1"}

    def describe_execution(self, **_kwargs):
        described = {"status": self._status}
        if self._output is not None:
            described["output"] = _json.dumps(self._output)
        if self._status not in {"RUNNING", "SUCCEEDED"}:
            described["cause"] = "the worker raised"
        return described


def run_main(smoke, monkeypatch, fake, extra_args=()):
    monkeypatch.setattr(smoke.boto3, "client", lambda *a, **k: fake)
    monkeypatch.setattr(
        "sys.argv",
        ["pipeline_smoke.py", "--state-machine-arn", "arn:sm", "--poll", "0", *extra_args],
    )
    return smoke.main()


GOOD = {"steps": {"analyzer": {"a": 1}, "formatter": {"document": "the rewritten text"}}}


def test_a_real_run_passes_the_deploy(smoke, monkeypatch, capsys):
    code = run_main(smoke, monkeypatch, FakeSfn("SUCCEEDED", GOOD))
    assert code == 0
    assert "::notice::" in capsys.readouterr().out


def test_a_failed_run_fails_the_deploy_and_says_why(smoke, monkeypatch, capsys):
    """The exit code is not enough; the reason has to be right.

    A failed run also has no final output, so a script that dropped the status
    check entirely would still exit 1 — reporting "produced nothing usable"
    for a run that actually errored, and sending whoever reads it to debug the
    wrong thing.
    """
    code = run_main(smoke, monkeypatch, FakeSfn("FAILED", {}))
    assert code == 1
    out = capsys.readouterr().out
    assert "::error::" in out
    assert "FAILED" in out, "the message must name the status the run ended in"
    assert "the worker raised" in out, "the cause has to survive into the message"


def test_a_timed_out_run_is_reported_as_timed_out(smoke, monkeypatch, capsys):
    code = run_main(smoke, monkeypatch, FakeSfn("TIMED_OUT", {}))
    assert code == 1
    assert "TIMED_OUT" in capsys.readouterr().out


def test_an_empty_success_fails_the_deploy(smoke, monkeypatch, capsys):
    """SUCCEEDED with nothing in the final step is the point of this check."""
    code = run_main(smoke, monkeypatch, FakeSfn("SUCCEEDED", {"steps": {"formatter": {}}}))
    assert code == 1
    out = capsys.readouterr().out
    assert "::error::" in out
    assert "produced nothing" in out


def test_a_success_with_no_steps_at_all_fails_the_deploy(smoke, monkeypatch):
    assert run_main(smoke, monkeypatch, FakeSfn("SUCCEEDED", {"steps": {}})) == 1


def test_a_run_that_never_finishes_fails_the_deploy(smoke, monkeypatch, capsys):
    code = run_main(smoke, monkeypatch, FakeSfn("RUNNING", None), extra_args=("--timeout", "0"))
    assert code == 1
    assert "::error::" in capsys.readouterr().out


def test_a_permissions_gap_warns_without_failing_the_deploy(smoke, monkeypatch, capsys):
    """The check failing is not the pipeline failing.

    Failing every deploy over a CI role that cannot start executions would be
    wrong; reporting it as a pass would be worse. It says NOT VERIFIED, loudly.
    """
    from botocore.exceptions import ClientError

    denied = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "StartExecution")
    code = run_main(smoke, monkeypatch, FakeSfn("SUCCEEDED", GOOD, start_error=denied))
    assert code == 0
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "NOT VERIFIED" in out


def test_a_real_start_failure_still_fails_the_deploy(smoke, monkeypatch, capsys):
    # Not every ClientError is a permissions gap; a missing state machine is
    # a broken deploy and must not be waved through as "not verified".
    from botocore.exceptions import ClientError

    missing = ClientError(
        {"Error": {"Code": "StateMachineDoesNotExist", "Message": "gone"}}, "StartExecution")
    code = run_main(smoke, monkeypatch, FakeSfn("SUCCEEDED", GOOD, start_error=missing))
    assert code == 1
    assert "::error::" in capsys.readouterr().out


def test_the_run_it_starts_is_the_team_it_says(smoke, monkeypatch):
    fake = FakeSfn("SUCCEEDED", GOOD)
    run_main(smoke, monkeypatch, fake)
    payload = _json.loads(fake.started[0]["input"])
    assert payload["team"] == smoke.DEFAULT_TEAM
    assert payload["version"] == smoke.DEFAULT_VERSION
    assert payload["request"] == smoke.DEFAULT_REQUEST
