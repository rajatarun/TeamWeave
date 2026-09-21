"""A role that runs agent turns must be allowed to call the substrate it uses.

`AGENT_RUNTIME` defaults to `agentcore`, which is a *different AWS service*
from Classic with a different action: `bedrock-agentcore:InvokeAgentRuntime`,
not `bedrock:InvokeAgent`. Only the Classic permission was granted, so every
agent turn in the VPC was refused with AccessDeniedException — from functions
that had been reaching the right endpoint all along.

The deploy's own smoke test could not catch it: CI invokes the runtime as the
**deployer** role, which may do anything. A check that runs as the wrong
identity proves the runtime answers somebody, not that it answers the caller
who needs it.

So the check is derived from the template: whichever substrate the stack
selects, the roles that run turns must carry its action.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO / "infra" / "template.yaml").read_text()

# What each substrate needs. agent_runtime.py picks between them.
SUBSTRATE_ACTIONS = {
    "agentcore": "bedrock-agentcore:InvokeAgentRuntime",
    "classic": "bedrock:InvokeAgent",
}

# Roles whose functions invoke agents: the pipeline worker and the synchronous
# /agent/converse turn.
TURN_RUNNING_ROLES = ("WorkerRole", "ConversationRole")


class CfnLoader(yaml.SafeLoader):
    """CloudFormation short forms, keeping the argument.

    A loader that discards it renders every `!Sub "arn:..."` as an empty
    marker, so a test asserting on resource ARNs sees nothing and fails on a
    correct template — or worse, passes vacuously on a broken one.
    """


def _keep(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {"__fn__": suffix, "__arg__": value}


CfnLoader.add_multi_constructor("!", _keep)


@pytest.fixture(scope="module")
def resources():
    return yaml.load(TEMPLATE, Loader=CfnLoader)["Resources"]


def actions_of(role: dict) -> set[str]:
    granted: set[str] = set()
    for policy in (role.get("Properties") or {}).get("Policies") or []:
        for statement in ((policy.get("PolicyDocument") or {}).get("Statement") or []):
            if statement.get("Effect") != "Allow":
                continue
            action = statement.get("Action")
            if isinstance(action, str):
                granted.add(action)
            elif isinstance(action, list):
                granted.update(a for a in action if isinstance(a, str))
    return granted


def roles_that_run_turns(resources) -> dict[str, dict]:
    found = {n: r for n, r in resources.items() if n in TURN_RUNNING_ROLES}
    return found


def test_the_scan_finds_the_roles_it_checks(resources):
    found = roles_that_run_turns(resources)
    assert "WorkerRole" in found, f"found only: {sorted(found)}"
    assert actions_of(found["WorkerRole"]), "WorkerRole appears to grant nothing"


def test_the_default_substrate_is_agentcore():
    # If this ever flips, the requirement below follows it rather than
    # pinning a permission the stack no longer uses.
    assert 'AGENT_RUNTIME: !If [AgentCoreEnabled, "agentcore", "classic"]' in TEMPLATE


@pytest.mark.parametrize("role_name", TURN_RUNNING_ROLES)
def test_every_turn_running_role_can_call_agentcore(resources, role_name):
    role = resources.get(role_name)
    if role is None:
        pytest.skip(f"{role_name} is not in this template")
    granted = actions_of(role)
    assert SUBSTRATE_ACTIONS["agentcore"] in granted, (
        f"{role_name} cannot call the default substrate, so every agent turn it "
        f"makes is refused with AccessDeniedException. Granted: {sorted(granted)}"
    )


@pytest.mark.parametrize("role_name", TURN_RUNNING_ROLES)
def test_the_classic_rollback_still_works(resources, role_name):
    """AGENT_RUNTIME=classic is the one-variable rollback; it must stay usable."""
    role = resources.get(role_name)
    if role is None:
        pytest.skip(f"{role_name} is not in this template")
    assert SUBSTRATE_ACTIONS["classic"] in actions_of(role)


def test_the_agentcore_grant_names_agentcore_resources(resources):
    """A Bedrock Classic ARN does not match an AgentCore runtime.

    Granting the right action against `arn:aws:bedrock:...:agent/*` would look
    correct in a diff and still deny every call.
    """
    statements = []
    for policy in (resources["WorkerRole"]["Properties"].get("Policies") or []):
        statements.extend((policy.get("PolicyDocument") or {}).get("Statement") or [])
    matching = [
        s for s in statements
        if SUBSTRATE_ACTIONS["agentcore"] in (
            s.get("Action") if isinstance(s.get("Action"), list) else [s.get("Action")]
        )
    ]
    assert matching, "no statement grants the AgentCore action"
    # Render the *resources* only. The action string itself contains
    # "bedrock-agentcore", so dumping the whole statement finds the substring
    # via the action and passes even when every resource is a Classic ARN --
    # which is the mistake this test exists to catch.
    for statement in matching:
        resources_field = statement.get("Resource")
        entries = resources_field if isinstance(resources_field, list) else [resources_field]
        rendered = yaml.dump(entries)
        assert "bedrock-agentcore:" in rendered, (
            f"the AgentCore action is granted against non-AgentCore resources: {entries}"
        )


def test_the_deploy_smoke_runs_as_a_different_identity_than_the_worker():
    """Why the smoke test could not have caught this, recorded on purpose.

    scripts/agentcore_smoke.py runs on the GitHub runner as the deployer role.
    It proves the runtime boots and answers; it says nothing about whether the
    worker's role may call it. Both checks are needed, and neither substitutes
    for the other.
    """
    workflow = (REPO / ".github" / "workflows" / "deploy.yml").read_text()
    assert "agentcore_smoke.py" in workflow
    assert "pipeline_smoke.py" in workflow, (
        "without a real pipeline run, nothing exercises the worker's own permissions"
    )
