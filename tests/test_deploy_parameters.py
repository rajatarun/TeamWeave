"""Template parameters the workflow decides must actually reach the stack.

`sam deploy` sends `UsePreviousValue=true` for every parameter absent from
`--parameter-overrides`. So a parameter's `Default:` in template.yaml governs
only the *first* deploy of a stack; after that, changing it has no effect at
all and nothing says so. The stack keeps whatever it was created with.

That is how EnableAgentCore stayed "false" across three green deploys after
its default was changed to "true": the AgentCoreEnabled condition never fired,
no AgentCore resource was created, and every function kept
AGENT_RUNTIME=classic with an empty AGENTCORE_RUNTIME_ARN -- while the
workflow, reading its own `env.ENABLE_AGENTCORE`, skipped Classic provisioning
on the strength of a switch the stack had never seen. Two sources of truth
disagreeing silently, which is worse than either setting being wrong.

The general rule these pin: if a workflow step branches on an env var, and a
template parameter governs the same feature, the workflow must *send* that
parameter rather than hope the template default applies.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO / ".github" / "workflows" / "deploy.yml"

# Workflow env var -> the template parameter that governs the same feature.
# Every env var used in a step's `if:` must appear here, so adding a new gate
# forces a decision about whether the stack needs to be told.
ENV_TO_PARAMETER = {
    "ENABLE_AGENTCORE": "EnableAgentCore",
}


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


@pytest.fixture(scope="module")
def deploy_run(workflow) -> str:
    steps = workflow["jobs"]["deploy"]["steps"]
    return next(s for s in steps if s.get("name") == "SAM Deploy")["run"]


def overridden_parameters(deploy_run: str) -> set[str]:
    return set(re.findall(r'"([A-Za-z0-9]+)=\$\{', deploy_run))


def gating_env_vars(workflow) -> set[str]:
    found = set()
    for step in workflow["jobs"]["deploy"]["steps"]:
        condition = step.get("if")
        if isinstance(condition, str):
            found.update(re.findall(r"env\.([A-Z0-9_]+)", condition))
    return found


@pytest.mark.parametrize("env_name,parameter", sorted(ENV_TO_PARAMETER.items()))
def test_a_feature_switch_is_sent_to_the_stack(deploy_run, env_name, parameter):
    assert f'"{parameter}=${{{env_name}}}"' in deploy_run, (
        f"{parameter} is not in --parameter-overrides, so sam deploy will send "
        f"UsePreviousValue=true and the stack will ignore {env_name} entirely"
    )


def test_every_gating_env_var_is_accounted_for(workflow):
    # A new `if: env.X` gate that governs a template parameter, with nobody
    # having sent that parameter, is exactly the defect above. Adding the gate
    # must therefore fail this test until someone maps it.
    unmapped = gating_env_vars(workflow) - set(ENV_TO_PARAMETER)
    assert not unmapped, (
        f"these env vars gate a step but are not mapped to a template parameter: "
        f"{sorted(unmapped)}. If one governs no parameter, map it to '' explicitly."
    )


@pytest.mark.parametrize("env_name", sorted(ENV_TO_PARAMETER))
def test_the_switch_has_a_workflow_level_value(workflow, env_name):
    # Referenced but never set expands to the empty string, which sam rejects
    # as "Key=" -- a deploy failure rather than a silent wrong value, but still
    # worth catching here.
    assert workflow["env"].get(env_name), f"{env_name} is used but never set at workflow level"


@pytest.mark.parametrize("env_name,parameter", sorted(ENV_TO_PARAMETER.items()))
def test_the_workflow_value_is_one_the_parameter_accepts(workflow, parameter, env_name):
    template = yaml.safe_load(
        re.sub(r"!\w+", "", (REPO / "infra" / "template.yaml").read_text())
    )
    allowed = template["Parameters"][parameter].get("AllowedValues")
    if allowed is None:
        pytest.skip(f"{parameter} constrains no values")
    assert str(workflow["env"][env_name]) in [str(v) for v in allowed]


def test_the_failure_dump_shows_this_runs_failure(workflow):
    """A failed deploy has to say what failed *in this run*, without paging.

    Two rounds of this. First the dump printed 50 raw events per stack with
    their full ResourceProperties -- thousands of lines, the one event with a
    ResourceStatusReason pushed out of the readable tail. Filtering to
    failures fixed that and introduced the second: it then printed the
    previous three deploys' rollbacks into a log whose own deploy had
    succeeded, which reads exactly like the current run failing.

    The filtering now lives in scripts/dump_stack_failures.py, where
    tests/test_stack_failure_dump.py exercises it on real event shapes rather
    than by matching strings in a shell block. What is left to pin here is
    that the workflow calls it and scopes it to this run.
    """
    steps = workflow["jobs"]["deploy"]["steps"]
    step = next(s for s in steps if s.get("name") == "Dump CloudFormation events on failure")
    run = step["run"]
    assert "dump_stack_failures.py" in run, "the dump must go through the script"
    # Scoped to this run. The timestamp is recorded in the job rather than
    # read from ${{ github.run_started_at }}, which expanded to an empty
    # string and made the dump itself the second failing step.
    assert "${DEPLOY_STARTED_AT}" in run, "an unscoped dump reports old runs' failures"
    assert step.get("if") == "failure()"
    # The raw form is what buried the answer; it must not come back.
    assert "describe-stack-events" not in run
