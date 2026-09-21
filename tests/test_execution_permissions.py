"""Reading a run is authorized against the execution, not the state machine.

`GET /team/task/{run_id}` answered:

    not authorized to perform: states:DescribeExecution on resource:
    arn:...:execution:tarun-content-team-state-machine:d9a86ce5-...

The action was granted. The resource was not:

    !Sub "${StateMachine}:*"  ->  arn:...:stateMachine:<name>:*

An execution's ARN uses the `execution:` resource-type segment, not
`stateMachine:`, so the statement matched nothing and every poll of every run
was denied — while the policy reads, at a glance, exactly like one that works.

This is the same family as `bedrock:InvokeAgent` versus
`bedrock-agentcore:InvokeAgentRuntime`: the right-looking grant against the
wrong resource. The A2A role, written later, got it right; the status role
predated it and nothing compared them.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO / "infra" / "template.yaml").read_text()

# Actions authorized against an execution ARN rather than the state machine.
EXECUTION_SCOPED = {
    "states:DescribeExecution",
    "states:GetExecutionHistory",
    "states:StopExecution",
}

# The IAM service prefix is `states`. `sfn` is botocore's client name -- the
# same confusion as AWS_ENDPOINT_URL_SFN -- and grants nothing.
NOT_AN_IAM_PREFIX = "sfn:"


class CfnLoader(yaml.SafeLoader):
    """CloudFormation short forms, keeping the argument.

    A loader that drops it renders every `!Sub "arn:..."` empty, and a test
    asserting on ARNs then passes vacuously on a broken template.
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
def statements():
    """Every Allow statement in every role, with its role name."""
    resources = yaml.load(TEMPLATE, Loader=CfnLoader)["Resources"]
    out = []
    for name, resource in resources.items():
        if resource.get("Type") != "AWS::IAM::Role":
            continue
        for policy in (resource.get("Properties") or {}).get("Policies") or []:
            for statement in ((policy.get("PolicyDocument") or {}).get("Statement") or []):
                if statement.get("Effect") == "Allow":
                    out.append((name, statement))
    return out


def actions_of(statement) -> list[str]:
    action = statement.get("Action")
    if isinstance(action, str):
        return [action]
    return [a for a in (action or []) if isinstance(a, str)]


def resource_text(statement) -> str:
    """Every resource in a statement, rendered including !Sub arguments."""
    resources = statement.get("Resource")
    entries = resources if isinstance(resources, list) else [resources]
    return yaml.dump(entries)


def test_the_scan_finds_the_statements_it_checks(statements):
    assert len(statements) >= 20, f"only found {len(statements)} statements"
    granting = [s for _, s in statements
                if any(a in EXECUTION_SCOPED for a in actions_of(s))]
    assert granting, "no role grants DescribeExecution — has it been renamed?"


def test_execution_scoped_actions_name_execution_arns(statements):
    """The bug: the right action against the state machine's own ARN."""
    wrong = []
    for role, statement in statements:
        scoped = [a for a in actions_of(statement) if a in EXECUTION_SCOPED]
        if not scoped:
            continue
        text = resource_text(statement)
        if ":execution:" not in text:
            wrong.append(f"{role}: {scoped} against {text.strip()}")
    assert not wrong, (
        "these grant an execution-scoped action against something that is not an "
        f"execution ARN, so every call is denied: {wrong}"
    )


def test_no_execution_grant_uses_the_state_machine_arn(statements):
    # `!Sub "${StateMachine}:*"` is the specific shape that was wrong: it looks
    # scoped and correct, and expands to the stateMachine resource type.
    for role, statement in statements:
        if not any(a in EXECUTION_SCOPED for a in actions_of(statement)):
            continue
        text = resource_text(statement)
        assert "${StateMachine}" not in text, (
            f"{role} scopes an execution action with the state machine's own ARN"
        )


def test_no_policy_uses_the_botocore_client_name_as_an_iam_prefix(statements):
    """`sfn:` grants nothing; the IAM prefix is `states:`.

    It is not harmful, but it makes a policy read as though it covers a case
    it does not — and reviewing the next one is harder for it.
    """
    offenders = [
        f"{role}: {a}" for role, statement in statements
        for a in actions_of(statement) if a.startswith(NOT_AN_IAM_PREFIX)
    ]
    assert not offenders, f"these actions do not exist in IAM: {offenders}"


def test_starting_a_run_is_still_scoped_to_the_state_machine(statements):
    """StartExecution *is* authorized against the state machine.

    The two are genuinely different, so fixing one must not be applied to the
    other: scoping StartExecution to an execution ARN would break starting
    runs just as thoroughly as the original bug broke polling them.

    Checked per statement, not by searching the template: a regex is satisfied
    by any one correct grant while another is wrong, which is exactly what it
    did when this was first written.
    """
    checked = 0
    for role, statement in statements:
        if "states:StartExecution" not in actions_of(statement):
            continue
        checked += 1
        text = resource_text(statement)
        assert ":execution:" not in text, (
            f"{role} scopes StartExecution to an execution ARN, which cannot match: "
            f"the execution does not exist until the call succeeds"
        )
    assert checked, "no role grants StartExecution — has it been renamed?"
