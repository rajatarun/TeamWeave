"""The VPC's only IPv4 route out, and what depends on it.

Every Bedrock service — including the AgentCore runtime that every agent turn
runs on — publishes no dual-stack endpoint, so from these subnets it is
reachable over IPv4 and nothing else. IPv4 egress is one t4g.nano NAT
instance. That instance is therefore the whole agent platform's egress, and it
was a **spot** instance with `InstanceInterruptionBehavior: stop`.

When spot stopped it, the route kept pointing at a stopped instance and every
Bedrock call failed with

    Connect timeout on endpoint URL:
    https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/.../invocations

Silently, because nothing watched the instance and a connect timeout reads
like a slow service rather than a missing route. That is what the first real
pipeline run ever attempted died of.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHARED = (REPO / "infra" / "shared.yaml").read_text()
WORKFLOW = (REPO / ".github" / "workflows" / "deploy.yml").read_text()
TEMPLATE = (REPO / "infra" / "template.yaml").read_text()


def nat_launch_template() -> str:
    """The launch template block, comments included."""
    block = SHARED.split("NatInstanceLaunchTemplate:", 1)
    assert len(block) == 2, "the NAT launch template has been renamed or removed"
    return block[1].split("\n  NatInstance:", 1)[0]


def nat_launch_config() -> str:
    """The same block with comments stripped.

    The comment explaining the spot failure naturally contains the words
    "MarketType: spot" and "InstanceInterruptionBehavior" -- so asserting
    against the raw text checks the prose, not the configuration, and fails
    on a correct template. Matching the documentation instead of the thing
    documented is the same mistake as a test that greps source.
    """
    return "\n".join(
        line for line in nat_launch_template().splitlines()
        if not line.strip().startswith("#")
    )


def test_the_nat_instance_still_exists():
    # Removing it entirely would take IPv4 egress with it.
    assert "NatInstance:" in SHARED
    assert "LambdaIpv4EgressRouteToNat:" in SHARED


def test_the_only_ipv4_route_is_not_on_spot():
    """A single point of failure must not also be interruptible.

    Spot is right for work that can be retried elsewhere. This instance is the
    only way out for every Bedrock call in the VPC, and its interruption
    behaviour was `stop` — which leaves the route pointing at a stopped
    instance rather than failing loudly.
    """
    config = nat_launch_config()
    assert "MarketType: spot" not in config, (
        "the NAT instance is on spot again; an interruption silently removes all "
        "IPv4 egress and every Bedrock call connect-times-out"
    )
    assert "InstanceInterruptionBehavior" not in config


def test_the_deploy_checks_the_nat_is_running():
    # The failure was invisible because nothing looked. A stopped instance and
    # a healthy one are indistinguishable from inside a Lambda: both give a
    # connect timeout.
    assert "describe-instances" in WORKFLOW, "nothing checks the NAT's state"
    assert "NAT_STATE" in WORKFLOW
    guard = WORKFLOW.split('if [ "${NAT_STATE}" != "running" ]; then', 1)
    assert len(guard) == 2, "the NAT's state is read but never checked"
    body = guard[1].split("\n          fi", 1)[0]
    assert "exit 1" in body, "a stopped NAT does not fail the deploy"
    # GitHub parses workflow commands from stdout only.
    assert "::error::" in body
    assert ">&2" not in body


def test_the_check_runs_before_anything_invokes_an_agent():
    """Order matters: a dead NAT should be named, not inferred from a timeout."""
    nat_at = WORKFLOW.index("Check the VPC's only IPv4 path out")
    smoke_at = WORKFLOW.index("Smoke-test the AgentCore runtime")
    pipeline_at = WORKFLOW.index("Run one real team pipeline")
    assert nat_at < smoke_at < pipeline_at


@pytest.mark.parametrize("service", ["bedrock-agentcore", "bedrock-runtime", "secretsmanager"])
def test_the_services_that_need_ipv4_are_still_pinned_to_it(service):
    """These have no dual-stack endpoint, which is *why* NAT is load-bearing.

    If one ever gains IPv6 its pin can go and the NAT carries less; until then
    removing a pin does not free the NAT, it breaks the service.
    """
    var = "AWS_ENDPOINT_URL_" + service.upper().replace("-", "_")
    if service == "secretsmanager":
        var = "AWS_ENDPOINT_URL_SECRETS_MANAGER"
    assert var in TEMPLATE


def test_the_ipv4_dependency_is_written_down_where_it_bites():
    # The next person to see a t4g.nano and think "spot would be cheaper"
    # should find the reason in the template, not in an incident.
    comments = "\n".join(
        line for line in nat_launch_template().splitlines() if line.strip().startswith("#")
    )
    assert re.search(r"only\W*IPv4|IPv4 path|IPv4 route", comments), (
        "nothing in the launch template says this instance is load-bearing"
    )
