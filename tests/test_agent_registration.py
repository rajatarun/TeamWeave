"""Registering every TeamWeave agent in the AgentCore registry.

AgentCore has no "agent" resource. Its registry is a runtime plus named
endpoints on it, and InvokeAgentRuntime's `qualifier` *is* an endpoint name --
the service model says so: "an endpoint name that points to a specific
version". So an agent's registry identity is an endpoint, and its `qualifier`
in team.json is what addresses it.

One runtime with one endpoint per agent, rather than a runtime per agent. A
runtime is a whole code artifact; twelve would mean twelve builds, uploads and
startup validations per deploy, which is the exact shape of the Classic
provisioning step that outgrew the CLI timeout and orphaned agents.

These pin the parts that fail quietly: a name that cannot address what it
claims to, two agents collapsing onto one identity, and an endpoint left
pointing at last deploy's artifact.
"""
from __future__ import annotations

import json
import os
import re
from unittest import mock

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from scripts import register_agents as reg  # noqa: E402

# The service's own constraint, from the botocore model for
# bedrock-agentcore-control: EndpointName is [a-zA-Z][a-zA-Z0-9_]{0,47}.
SERVICE_PATTERN = re.compile(r"\A[a-zA-Z][a-zA-Z0-9_]{0,47}\Z")


def team(*agent_ids, **kw):
    return {"agents": [{"id": a, "bedrock": dict(kw)} for a in agent_ids]}


# ── endpoint names ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("agent_id", [
    "strategist",
    "brand-strategist",          # hyphens: illegal in an endpoint name
    "brand.strategist v2",       # dots and spaces
    "2nd_writer",                # must start with a letter
    "-leading-hyphen",
    "a" * 80,                    # over the 48-character cap
    "UPPER_and_lower",
])
def test_every_name_is_one_the_service_accepts(agent_id):
    # The alphabet that made AgentRuntimeName fail to create when it was built
    # from a hyphenated stack name. Agent ids are hyphenated far more often
    # than stack names are, so this is the likelier version of that failure.
    assert SERVICE_PATTERN.match(reg.endpoint_name(agent_id))


def test_a_name_is_stable_across_deploys():
    # The name is the address. A different one next deploy would orphan the
    # endpoint and silently repoint the agent at the runtime default.
    assert reg.endpoint_name("brand-strategist") == reg.endpoint_name("brand-strategist")


def test_a_name_is_derived_from_the_id_not_invented():
    assert "brand" in reg.endpoint_name("brand-strategist")


def test_an_id_with_nothing_usable_is_refused():
    with pytest.raises(ValueError):
        reg.endpoint_name("---")


# ── planning ───────────────────────────────────────────────────────────────

def test_every_agent_across_every_team_is_planned():
    names, problems = reg.plan_endpoints({
        "teams/a/v1/team.json": team("writer", "editor"),
        "teams/b/v1/team.json": team("coach"),
    })
    assert not problems
    assert set(names) == {"writer", "editor", "coach"}


def test_two_agents_colliding_on_one_name_is_an_error():
    # "a-b" and "a_b" both sanitise to "a_b". Sharing an endpoint means
    # sharing a registry identity: telemetry merges and a version pin meant
    # for one moves the other. That looks like nothing at all until someone
    # reads a dashboard, so it must never be a silent alias.
    names, problems = reg.plan_endpoints({"teams/a/v1/team.json": team("a-b", "a_b")})
    assert problems and "share one registry identity" in problems[0]


def test_the_same_agent_in_two_teams_is_not_a_collision():
    names, problems = reg.plan_endpoints({
        "teams/a/v1/team.json": team("writer"),
        "teams/b/v1/team.json": team("writer"),
    })
    assert not problems and names == {"writer": "writer"}


def test_an_agent_with_no_id_is_reported_not_skipped():
    names, problems = reg.plan_endpoints({"teams/a/v1/team.json": {"agents": [{"bedrock": {}}]}})
    assert problems


def test_an_agent_named_only_by_name_still_registers():
    plan, problems = reg.plan_endpoints({"teams/a/v1/team.json": {"agents": [{"name": "writer"}]}})
    assert not problems and plan == {"writer": "writer"}


# ── the AWS calls ──────────────────────────────────────────────────────────

class FakeControl:
    def __init__(self):
        self.created, self.updated = [], []

    def create_agent_runtime_endpoint(self, **kw):
        self.created.append(kw); return {}

    def update_agent_runtime_endpoint(self, **kw):
        self.updated.append(kw); return {}


def test_a_missing_endpoint_is_created_pinned_to_the_current_version():
    client = FakeControl()
    action = reg.ensure_endpoint(client, "rt-1", "writer", "7", "TeamWeave agent writer", {})
    assert action == "created"
    assert client.created[0]["agentRuntimeId"] == "rt-1"
    assert client.created[0]["name"] == "writer"
    assert client.created[0]["agentRuntimeVersion"] == "7"


def test_an_endpoint_already_on_this_version_is_left_alone():
    client = FakeControl()
    existing = {"writer": {"name": "writer", "targetVersion": "7"}}
    assert reg.ensure_endpoint(client, "rt-1", "writer", "7", "d", existing) == "current"
    assert not client.created and not client.updated


def test_an_endpoint_on_an_older_version_is_repointed():
    # Otherwise it keeps serving the previous deploy's artifact and the
    # registry drifts from the code with nothing reporting it.
    client = FakeControl()
    existing = {"writer": {"name": "writer", "targetVersion": "6"}}
    action = reg.ensure_endpoint(client, "rt-1", "writer", "7", "d", existing)
    assert "repointed" in action
    assert client.updated[0]["endpointName"] == "writer"
    assert client.updated[0]["agentRuntimeVersion"] == "7"


def test_the_live_version_is_read_when_there_is_no_target():
    # A list item carries liveVersion and targetVersion and no
    # `agentRuntimeVersion`; comparing against a field that does not exist
    # reads as "" and re-points every endpoint on every deploy.
    client = FakeControl()
    existing = {"writer": {"name": "writer", "liveVersion": "7"}}
    assert reg.ensure_endpoint(client, "rt-1", "writer", "7", "d", existing) == "current"
    assert not client.updated


def test_endpoints_are_listed_across_pages():
    pages = [
        {"runtimeEndpoints": [{"name": "a"}], "nextToken": "t"},
        {"runtimeEndpoints": [{"name": "b"}]},
    ]
    client = mock.Mock()
    client.list_agent_runtime_endpoints.side_effect = pages
    # A single page would make page two's endpoints look absent and be
    # recreated, which the service rejects as already existing.
    assert set(reg.existing_endpoints(client, "rt-1")) == {"a", "b"}


# ── writing identity back into team.json ───────────────────────────────────


def test_writing_back_preserves_the_rest_of_the_bedrock_block():
    doc = team("writer", model_id="us.amazon.nova-micro-v1:0")
    reg.write_back(doc, {"writer": "writer"}, "arn:rt")
    assert doc["agents"][0]["bedrock"]["model_id"] == "us.amazon.nova-micro-v1:0"


def test_an_unchanged_agent_is_not_rewritten():
    # So a no-op deploy does not rewrite every team.json in S3.
    doc = team("writer", runtimeArn="arn:rt")
    assert reg.write_back(doc, {"writer": "writer"}, "arn:rt") == 0


def test_what_is_written_back_is_what_the_orchestrator_can_use():
    # The other half of the contract. An empty qualifier is not a gap: it
    # means the runtime's DEFAULT endpoint, which is where every agent runs.
    from src.orchestrator.agent_runtime import AgentCoreRuntime, AgentRef

    doc = team("writer")
    reg.write_back(doc, {"writer": "writer"}, "arn:rt")
    bedrock = doc["agents"][0]["bedrock"]
    ref = AgentRef(runtime_arn=bedrock["runtimeArn"], qualifier=bedrock.get("qualifier", ""))
    assert AgentCoreRuntime().missing_fields(ref) == ""
    assert AgentCoreRuntime.resolve_arn(ref) == "arn:rt"
    assert ref.qualifier == ""


# ── the deploy actually runs it ────────────────────────────────────────────

import yaml  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load((REPO / ".github" / "workflows" / "deploy.yml").read_text())


@pytest.fixture(scope="module")
def registry_step(workflow):
    steps = workflow["jobs"]["deploy"]["steps"]
    return next(s for s in steps if s.get("name") == "Register agents and AgentCore release channels")


def test_the_registry_step_runs_on_every_agentcore_deploy(registry_step):
    assert registry_step["if"] == "env.ENABLE_AGENTCORE == 'true'"
    assert "register_agents.py" in registry_step["run"]


def test_registration_happens_after_the_config_sync(workflow):
    # It writes each agent's qualifier into the same S3 object the sync
    # merges. Running first would hand the sync a runtime identity it then had
    # to preserve, rather than one it simply carries across.
    names = [s.get("name") for s in workflow["jobs"]["deploy"]["steps"]]
    assert names.index("Sync team configs to S3") < names.index(
        "Register agents and AgentCore release channels"
    )


def test_each_stack_output_is_queried_on_its_own(registry_step):
    # One filter matching several OutputKeys returns them in the stack's
    # output order, not the order they are read into, so the runtime id and
    # the config bucket can silently swap places.
    run = registry_step["run"]
    for output in ("AgentCoreRuntimeId", "AgentCoreRuntimeVersion",
                   "AgentCoreRuntimeArn", "ConfigBucket"):
        assert f"stack_output {output}" in run, f"{output} is not fetched on its own"
    # The multi-key form is what reorders silently; it must not come back.
    assert "||OutputKey==" not in run
    assert run.count("OutputKey==") == 1, "the helper should hold the only filter"


def test_a_missing_stack_output_stops_the_step(registry_step):
    # `--output text` prints "None" for a missing output, which would
    # otherwise be passed to the API as a literal runtime id.
    assert '"None"' in registry_step["run"]


def test_the_template_publishes_what_the_step_reads():
    template = yaml.safe_load(
        re.sub(r"!\w+", "", (REPO / "infra" / "template.yaml").read_text())
    )
    outputs = template["Outputs"]
    for name in ("AgentCoreRuntimeId", "AgentCoreRuntimeVersion", "AgentCoreRuntimeArn"):
        assert name in outputs, f"{name} is read by the registry step but never published"


def test_the_ci_role_may_manage_endpoints_but_not_delete_them():
    # An agent dropped from a config may still be addressed by a run in
    # flight, so the registry never deletes; the grant says so too.
    template = yaml.safe_load(
        re.sub(r"!\w+", "", (REPO / "infra" / "template.yaml").read_text())
    )
    policy = template["Resources"]["SamAssumeRoleAgentCoreRegistryPermissions"]
    actions = policy["Properties"]["PolicyDocument"]["Statement"][0]["Action"]
    assert "bedrock-agentcore:CreateAgentRuntimeEndpoint" in actions
    assert "bedrock-agentcore:UpdateAgentRuntimeEndpoint" in actions
    assert "bedrock-agentcore:ListAgentRuntimeEndpoints" in actions
    assert not any("Delete" in a for a in actions)


# ── how a failure is reported ──────────────────────────────────────────────
#
# Mutation testing found these untested: putting the reason back on stderr,
# and letting an AWS error escape as a bare traceback, both passed. That is
# the same defect the smoke test had -- a step whose job is to explain itself,
# unable to.

import io  # noqa: E402
from botocore.exceptions import BotoCoreError, ClientError  # noqa: E402


def run_main(monkeypatch, s3, control, argv_extra=()):
    argv = ["register_agents.py", "--runtime-id", "rt-1",
            "--runtime-arn", "arn:rt", "--runtime-version", "7",
            "--bucket", "b", "--prefix", "teams", *argv_extra]
    clients = {"s3": s3, "bedrock-agentcore-control": control}
    monkeypatch.setattr(reg.boto3, "client", lambda name, **kw: clients[name])
    monkeypatch.setattr(reg.sys, "argv", argv)
    return reg.main()


def fake_s3(teams):
    client = mock.Mock()
    client.list_objects_v2.return_value = {
        "Contents": [{"Key": k} for k in teams], "IsTruncated": False
    }
    client.get_object.side_effect = lambda Bucket, Key: {
        "Body": io.BytesIO(json.dumps(teams[Key]).encode())
    }
    return client


def test_a_successful_run_announces_what_it_registered(monkeypatch, capsys):
    control = FakeControl()
    control.list_agent_runtime_endpoints = lambda **kw: {"runtimeEndpoints": []}
    rc = run_main(monkeypatch, fake_s3({"teams/a/v1/team.json": team("writer")}), control)
    out = capsys.readouterr().out
    assert rc == 0
    assert "::notice::" in out and "1 agent(s) on runtime" in out
    # The notice must say where identity actually lives, or the next person
    # re-adds an endpoint per agent.
    assert "gen_ai.agent.id" in out


def test_an_aws_failure_is_announced_rather_than_raised(monkeypatch, capsys):
    # A traceback in a log nobody can page to is not a diagnosis.
    control = mock.Mock()
    control.list_agent_runtime_endpoints.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "nope"}},
        "ListAgentRuntimeEndpoints",
    )
    rc = run_main(monkeypatch, fake_s3({"teams/a/v1/team.json": team("writer")}), control)
    assert rc == 1
    out = capsys.readouterr().out
    assert "::error::" in out and "AccessDeniedException" in out


def test_a_botocore_failure_is_announced_too(monkeypatch, capsys):
    control = mock.Mock()
    control.list_agent_runtime_endpoints.side_effect = BotoCoreError()
    assert run_main(monkeypatch, fake_s3({"teams/a/v1/team.json": team("writer")}), control) == 1
    assert "::error::" in capsys.readouterr().out


def test_an_empty_prefix_is_announced_as_an_error(monkeypatch, capsys):
    # A wrong prefix looks exactly like a platform with no teams.
    s3 = mock.Mock()
    s3.list_objects_v2.return_value = {"Contents": [], "IsTruncated": False}
    assert run_main(monkeypatch, s3, FakeControl()) == 1
    out = capsys.readouterr().out
    assert "::error::" in out and "No team.json" in out


def test_a_name_collision_is_announced_as_an_error(monkeypatch, capsys):
    teams = {"teams/a/v1/team.json": team("a-b", "a_b")}
    assert run_main(monkeypatch, fake_s3(teams), FakeControl()) == 1
    out = capsys.readouterr().out
    assert "::error::" in out and "share one registry identity" in out


def test_the_reason_reaches_the_step_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "s.md"))
    s3 = mock.Mock()
    s3.list_objects_v2.return_value = {"Contents": [], "IsTruncated": False}
    run_main(monkeypatch, s3, FakeControl())
    assert "No team.json" in (tmp_path / "s.md").read_text()


def test_an_unwritable_summary_never_masks_the_real_result(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", "/proc/nonexistent/s.md")
    control = FakeControl()
    control.list_agent_runtime_endpoints = lambda **kw: {"runtimeEndpoints": []}
    rc = run_main(monkeypatch, fake_s3({"teams/a/v1/team.json": team("writer")}), control)
    assert rc == 0
    assert "::notice::" in capsys.readouterr().out


# ── release channels, not one endpoint per agent ───────────────────────────
#
# The first version created an endpoint per agent and hit
# ServiceQuotaExceededException at twelve, because the default is ten. That
# quota is a release budget: AWS documents endpoints as how production stays on
# a stable version while staging tests a newer one. Spending it on identity did
# not scale, and consumed the very endpoints that shadow invocation needs --
# the capability the design was argued for.


def test_endpoints_created_are_channels_not_agents(monkeypatch, capsys):
    control = FakeControl()
    control.list_agent_runtime_endpoints = lambda **kw: {"runtimeEndpoints": []}
    teams = {"teams/a/v1/team.json": team("aa", "bb", "cc", "dd")}
    assert run_main(monkeypatch, fake_s3(teams), control, ("--channels", "shadow")) == 0
    # One channel, four agents: the count must not track the agents.
    assert [c["name"] for c in control.created] == ["shadow"]


def test_the_endpoint_count_does_not_grow_with_agents(monkeypatch):
    control = FakeControl()
    control.list_agent_runtime_endpoints = lambda **kw: {"runtimeEndpoints": []}
    many = {"teams/a/v1/team.json": team(*[f"agent_{i}" for i in range(40)])}
    run_main(monkeypatch, fake_s3(many), control, ("--channels", "shadow"))
    assert len(control.created) == 1, "40 agents must still cost one endpoint"


def test_no_agent_is_given_a_per_agent_qualifier(monkeypatch):
    control = FakeControl()
    control.list_agent_runtime_endpoints = lambda **kw: {"runtimeEndpoints": []}
    doc = team("aa", "bb")
    s3 = fake_s3({"teams/a/v1/team.json": doc})
    puts = []
    s3.put_object.side_effect = lambda **kw: puts.append(json.loads(kw["Body"]))
    run_main(monkeypatch, s3, control, ("--channels", "shadow"))
    for agent in puts[-1]["agents"]:
        assert agent["bedrock"]["runtimeArn"] == "arn:rt"
        assert "qualifier" not in agent["bedrock"]


def test_a_qualifier_left_by_the_old_scheme_is_cleared(monkeypatch):
    # It names an endpoint that is no longer this agent's; leaving it would
    # route the agent at whatever that endpoint now serves.
    control = FakeControl()
    control.list_agent_runtime_endpoints = lambda **kw: {"runtimeEndpoints": []}
    doc = team("aa", runtimeArn="arn:rt", qualifier="aa")
    s3 = fake_s3({"teams/a/v1/team.json": doc})
    puts = []
    s3.put_object.side_effect = lambda **kw: puts.append(json.loads(kw["Body"]))
    run_main(monkeypatch, s3, control, ("--channels", "shadow"))
    assert "qualifier" not in puts[-1]["agents"][0]["bedrock"]


def test_a_full_quota_is_now_a_real_warning(monkeypatch, capsys):
    # Releases failing is a genuine signal, unlike the expected outcome it was
    # when every agent consumed a slot.
    control = QuotaFullControl()
    teams = {"teams/a/v1/team.json": team("aa")}
    assert run_main(monkeypatch, fake_s3(teams), control, ("--channels", "shadow")) == 0
    out = capsys.readouterr().out
    assert "::warning::" in out and "release channel" in out


def test_an_illegal_channel_name_is_refused(monkeypatch, capsys):
    control = FakeControl()
    control.list_agent_runtime_endpoints = lambda **kw: {"runtimeEndpoints": []}
    teams = {"teams/a/v1/team.json": team("aa")}
    assert run_main(monkeypatch, fake_s3(teams), control, ("--channels", "not-legal")) == 1
    assert "::error::" in capsys.readouterr().out


class QuotaFullControl(FakeControl):
    def list_agent_runtime_endpoints(self, **kw):
        return {"runtimeEndpoints": []}

    def create_agent_runtime_endpoint(self, **kw):
        raise quota_error()


def quota_error():
    return ClientError(
        {"Error": {"Code": "ServiceQuotaExceededException",
                   "Message": "maxEndpointsPerAgent limit exceeded"}},
        "CreateAgentRuntimeEndpoint",
    )


def test_a_quota_error_is_told_apart_from_a_real_error():
    assert reg.is_quota_error(quota_error())
    assert not reg.is_quota_error(ClientError(
        {"Error": {"Code": "AccessDeniedException"}}, "CreateAgentRuntimeEndpoint"))


# ── the deploy actually runs it ────────────────────────────────────────────

import yaml  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load((REPO / ".github" / "workflows" / "deploy.yml").read_text())


@pytest.fixture(scope="module")
def registry_step(workflow):
    steps = workflow["jobs"]["deploy"]["steps"]
    return next(s for s in steps if s.get("name") == "Register agents and AgentCore release channels")


def test_the_registry_step_runs_on_every_agentcore_deploy(registry_step):
    assert registry_step["if"] == "env.ENABLE_AGENTCORE == 'true'"
    assert "register_agents.py" in registry_step["run"]


def test_registration_happens_after_the_config_sync(workflow):
    # It writes each agent's qualifier into the same S3 object the sync
    # merges. Running first would hand the sync a runtime identity it then had
    # to preserve, rather than one it simply carries across.
    names = [s.get("name") for s in workflow["jobs"]["deploy"]["steps"]]
    assert names.index("Sync team configs to S3") < names.index(
        "Register agents and AgentCore release channels"
    )


def test_each_stack_output_is_queried_on_its_own(registry_step):
    # One filter matching several OutputKeys returns them in the stack's
    # output order, not the order they are read into, so the runtime id and
    # the config bucket can silently swap places.
    run = registry_step["run"]
    for output in ("AgentCoreRuntimeId", "AgentCoreRuntimeVersion",
                   "AgentCoreRuntimeArn", "ConfigBucket"):
        assert f"stack_output {output}" in run, f"{output} is not fetched on its own"
    # The multi-key form is what reorders silently; it must not come back.
    assert "||OutputKey==" not in run
    assert run.count("OutputKey==") == 1, "the helper should hold the only filter"


def test_a_missing_stack_output_stops_the_step(registry_step):
    # `--output text` prints "None" for a missing output, which would
    # otherwise be passed to the API as a literal runtime id.
    assert '"None"' in registry_step["run"]


def test_the_template_publishes_what_the_step_reads():
    template = yaml.safe_load(
        re.sub(r"!\w+", "", (REPO / "infra" / "template.yaml").read_text())
    )
    outputs = template["Outputs"]
    for name in ("AgentCoreRuntimeId", "AgentCoreRuntimeVersion", "AgentCoreRuntimeArn"):
        assert name in outputs, f"{name} is read by the registry step but never published"


def test_the_ci_role_may_manage_endpoints_but_not_delete_them():
    # An agent dropped from a config may still be addressed by a run in
    # flight, so the registry never deletes; the grant says so too.
    template = yaml.safe_load(
        re.sub(r"!\w+", "", (REPO / "infra" / "template.yaml").read_text())
    )
    policy = template["Resources"]["SamAssumeRoleAgentCoreRegistryPermissions"]
    actions = policy["Properties"]["PolicyDocument"]["Statement"][0]["Action"]
    assert "bedrock-agentcore:CreateAgentRuntimeEndpoint" in actions
    assert "bedrock-agentcore:UpdateAgentRuntimeEndpoint" in actions
    assert "bedrock-agentcore:ListAgentRuntimeEndpoints" in actions
    assert not any("Delete" in a for a in actions)


# ── how a failure is reported ──────────────────────────────────────────────
#
# Mutation testing found these untested: putting the reason back on stderr,
# and letting an AWS error escape as a bare traceback, both passed. That is
# the same defect the smoke test had -- a step whose job is to explain itself,
# unable to.

import io  # noqa: E402
from botocore.exceptions import BotoCoreError, ClientError  # noqa: E402


def run_main(monkeypatch, s3, control, argv_extra=()):
    argv = ["register_agents.py", "--runtime-id", "rt-1",
            "--runtime-arn", "arn:rt", "--runtime-version", "7",
            "--bucket", "b", "--prefix", "teams", *argv_extra]
    clients = {"s3": s3, "bedrock-agentcore-control": control}
    monkeypatch.setattr(reg.boto3, "client", lambda name, **kw: clients[name])
    monkeypatch.setattr(reg.sys, "argv", argv)
    return reg.main()


def fake_s3(teams):
    client = mock.Mock()
    client.list_objects_v2.return_value = {
        "Contents": [{"Key": k} for k in teams], "IsTruncated": False
    }
    client.get_object.side_effect = lambda Bucket, Key: {
        "Body": io.BytesIO(json.dumps(teams[Key]).encode())
    }
    return client


def test_a_successful_run_announces_what_it_registered(monkeypatch, capsys):
    control = FakeControl()
    control.list_agent_runtime_endpoints = lambda **kw: {"runtimeEndpoints": []}
    rc = run_main(monkeypatch, fake_s3({"teams/a/v1/team.json": team("writer")}), control)
    out = capsys.readouterr().out
    assert rc == 0
    assert "::notice::" in out and "1 agent(s) on runtime" in out
    # The notice has to say where identity actually lives, or the next person
    # reading it re-adds an endpoint per agent.
    assert "gen_ai.agent.id" in out


def test_an_aws_failure_is_announced_rather_than_raised(monkeypatch, capsys):
    # A traceback in a log nobody can page to is not a diagnosis.
    control = mock.Mock()
    control.list_agent_runtime_endpoints.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "nope"}},
        "ListAgentRuntimeEndpoints",
    )
    rc = run_main(monkeypatch, fake_s3({"teams/a/v1/team.json": team("writer")}), control)
    assert rc == 1
    out = capsys.readouterr().out
    assert "::error::" in out and "AccessDeniedException" in out


def test_a_botocore_failure_is_announced_too(monkeypatch, capsys):
    control = mock.Mock()
    control.list_agent_runtime_endpoints.side_effect = BotoCoreError()
    assert run_main(monkeypatch, fake_s3({"teams/a/v1/team.json": team("writer")}), control) == 1
    assert "::error::" in capsys.readouterr().out


def test_an_empty_prefix_is_announced_as_an_error(monkeypatch, capsys):
    # A wrong prefix looks exactly like a platform with no teams.
    s3 = mock.Mock()
    s3.list_objects_v2.return_value = {"Contents": [], "IsTruncated": False}
    assert run_main(monkeypatch, s3, FakeControl()) == 1
    out = capsys.readouterr().out
    assert "::error::" in out and "No team.json" in out


def test_a_name_collision_is_announced_as_an_error(monkeypatch, capsys):
    teams = {"teams/a/v1/team.json": team("a-b", "a_b")}
    assert run_main(monkeypatch, fake_s3(teams), FakeControl()) == 1
    out = capsys.readouterr().out
    assert "::error::" in out and "share one registry identity" in out


def test_the_reason_reaches_the_step_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "s.md"))
    s3 = mock.Mock()
    s3.list_objects_v2.return_value = {"Contents": [], "IsTruncated": False}
    run_main(monkeypatch, s3, FakeControl())
    assert "No team.json" in (tmp_path / "s.md").read_text()


def test_an_unwritable_summary_never_masks_the_real_result(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", "/proc/nonexistent/s.md")
    control = FakeControl()
    control.list_agent_runtime_endpoints = lambda **kw: {"runtimeEndpoints": []}
    rc = run_main(monkeypatch, fake_s3({"teams/a/v1/team.json": team("writer")}), control)
    assert rc == 0
    assert "::notice::" in capsys.readouterr().out


# ── the per-runtime endpoint quota ─────────────────────────────────────────
#
# AgentCore caps endpoints per runtime, and twelve agents exceeded it on the
# first real run: ServiceQuotaExceededException, "maxEndpointsPerAgent limit
# exceeded". That is an account quota, not something a deploy can route
# around, so what matters is that it degrades honestly rather than either
# failing every deploy or passing quietly.


def quota_error():
    return ClientError(
        {"Error": {"Code": "ServiceQuotaExceededException",
                   "Message": "maxEndpointsPerAgent limit exceeded"}},
        "CreateAgentRuntimeEndpoint",
    )


class QuotaLimitedControl:
    """Accepts `limit` endpoints, then refuses like the real service."""

    def __init__(self, limit):
        self.limit = limit
        self.created = []

    def list_agent_runtime_endpoints(self, **kw):
        return {"runtimeEndpoints": []}

    def create_agent_runtime_endpoint(self, **kw):
        if len(self.created) >= self.limit:
            raise quota_error()
        self.created.append(kw["name"])
        return {}

    def update_agent_runtime_endpoint(self, **kw):
        return {}



def test_a_quota_error_is_told_apart_from_a_real_error():
    assert reg.is_quota_error(quota_error())
    assert not reg.is_quota_error(ClientError(
        {"Error": {"Code": "AccessDeniedException"}}, "CreateAgentRuntimeEndpoint"))



def test_a_non_quota_error_still_fails_the_deploy(monkeypatch, capsys):
    # Degrading on a quota must not turn every AWS error into a warning.
    control = QuotaLimitedControl(limit=99)
    control.create_agent_runtime_endpoint = mock.Mock(side_effect=ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "nope"}},
        "CreateAgentRuntimeEndpoint"))
    teams = {"teams/a/v1/team.json": team("aa")}
    assert run_main(monkeypatch, fake_s3(teams), control) == 1
    assert "::error::" in capsys.readouterr().out


# ── Which runtime an agent is recorded against ───────────────────────────────
# The worker resolves an agent's own runtimeArn *before* its team's, so
# stamping the shared ARN onto every agent is not a neutral record: it shadows
# the per-team routing entirely. The pipeline's failure named
# teamweave_agent-BMHHKuDNo1 while three per-team runtimes sat unused and
# fully deployed.

import importlib.util as _importlib_util  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


def _register_agents():
    spec = _importlib_util.spec_from_file_location(
        "register_agents", _Path(__file__).resolve().parents[1] / "scripts" / "register_agents.py")
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEAM_MAP = {
    "doc_rewrite_team": "arn:aws:bedrock-agentcore:us-east-1:1:runtime/teamweave_doc_rewrite_team-a",
    "tarun_visibility_team": "arn:aws:bedrock-agentcore:us-east-1:1:runtime/teamweave_tarun_visibility_team-b",
}
SHARED = "arn:aws:bedrock-agentcore:us-east-1:1:runtime/teamweave_agent-shared"


def test_an_agent_is_recorded_against_its_own_teams_runtime():
    mod = _register_agents()
    team = {"team": {"name": "doc_rewrite_team"}}
    assert mod.runtime_for_team(team, TEAM_MAP, SHARED) == TEAM_MAP["doc_rewrite_team"]


def test_two_teams_are_not_recorded_against_the_same_runtime():
    mod = _register_agents()
    a = mod.runtime_for_team({"team": {"name": "doc_rewrite_team"}}, TEAM_MAP, SHARED)
    b = mod.runtime_for_team({"team": {"name": "tarun_visibility_team"}}, TEAM_MAP, SHARED)
    assert a != b, "per-team runtimes recorded as one runtime are not per-team"
    assert SHARED not in (a, b), "the shared runtime shadows the team's"


def test_a_team_the_map_does_not_name_falls_back():
    # A team added as JSON before its runtime exists still gets a working
    # agent record rather than an empty one.
    mod = _register_agents()
    assert mod.runtime_for_team({"team": {"name": "brand_new"}}, TEAM_MAP, SHARED) == SHARED


def test_a_config_with_no_team_name_falls_back():
    mod = _register_agents()
    assert mod.runtime_for_team({}, TEAM_MAP, SHARED) == SHARED
    assert mod.runtime_for_team({"team": {}}, TEAM_MAP, SHARED) == SHARED


def test_an_empty_map_falls_back_rather_than_recording_nothing():
    mod = _register_agents()
    assert mod.runtime_for_team({"team": {"name": "doc_rewrite_team"}}, {}, SHARED) == SHARED


def test_write_back_stamps_the_team_runtime_on_each_agent():
    """End of the path: what actually lands in team.json."""
    mod = _register_agents()
    team = {
        "team": {"name": "doc_rewrite_team"},
        "agents": [{"id": "a1", "bedrock": {"runtimeArn": SHARED, "qualifier": "old"}}],
    }
    arn = mod.runtime_for_team(team, TEAM_MAP, SHARED)
    changed = mod.write_back(team, {"a1": "a1"}, arn)
    assert changed == 1
    assert team["agents"][0]["bedrock"]["runtimeArn"] == TEAM_MAP["doc_rewrite_team"]
    assert "qualifier" not in team["agents"][0]["bedrock"]


def test_the_deploy_passes_the_map_to_the_registrar():
    workflow = (_Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml").read_text()
    assert "--team-runtime-arns" in workflow, (
        "the registrar is never told the map, so every agent is stamped with the shared runtime"
    )
    assert "TEAM_RUNTIME_ARNS=$(stack_output AgentCoreTeamRuntimeArns)" in workflow
