"""The deploy's IPv6 discovery query, run against a real EC2 response shape.

Everything in this stack is configured for IPv6 egress: an Egress-Only
Internet Gateway, `AWS_USE_DUALSTACK_ENDPOINT=true`, `Ipv6AllowedForDualStack`
on every function, `AssignIpv6AddressOnCreation` on both Lambda subnets. None
of it did anything, because the one step that connects them — discovering the
/56 AWS assigned to the VPC and feeding it back as `VpcIpv6Block` — used

    Vpcs[0].Ipv6CidrBlockAssociationSet[?State==`associated`]

and a VpcIpv6CidrBlockAssociation has no top-level `State`. The state lives at
`Ipv6CidrBlockState.State`. The filter therefore matched nothing, the
expression yielded None, `VpcIpv6Block` was never passed, the subnets never
got a /64, and the shared stack reported `DualStackEnabled: false` — while
every log line about IPv6 said the feature was on. All egress went out the NAT
instance.

Nothing failed. A JMESPath that matches nothing is not an error, the `|| echo
""` fallback never fired because the CLI call succeeded, and the two-pass
re-deploy used the same broken function for its second look, so it concluded
the CIDR still had not been assigned.

So this test does not grep the workflow for a string. It extracts the query
the workflow actually runs and evaluates it against a response constructed
from botocore's own EC2 service model — the same data the CLI validates
against, which is available locally and does not go stale.
"""
from __future__ import annotations

import re
from pathlib import Path

import botocore.session
import jmespath
import pytest

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = (REPO / ".github" / "workflows" / "deploy.yml").read_text()

ASSIGNED_CIDR = "2600:1f18:abcd:ef00::/56"


@pytest.fixture(scope="module")
def association_members() -> set[str]:
    """What a VpcIpv6CidrBlockAssociation really contains, per botocore."""
    model = botocore.session.get_session().get_service_model("ec2")
    vpc = model.operation_model("DescribeVpcs").output_shape.members["Vpcs"].member
    return set(vpc.members["Ipv6CidrBlockAssociationSet"].member.members)


def discovery_query() -> str:
    """The --query the deploy passes to `aws ec2 describe-vpcs`."""
    match = re.search(r"--query '(Vpcs\[0\]\.Ipv6CidrBlockAssociationSet[^']*)'", WORKFLOW)
    assert match, "the workflow no longer runs an IPv6 discovery query this test can find"
    return match.group(1)


def describe_vpcs_response(state: str = "associated") -> dict:
    return {"Vpcs": [{
        "VpcId": "vpc-f3c92a8a",
        "CidrBlock": "172.31.0.0/16",
        "Ipv6CidrBlockAssociationSet": [{
            "AssociationId": "vpc-cidr-assoc-0abc123",
            "Ipv6CidrBlock": ASSIGNED_CIDR,
            "Ipv6CidrBlockState": {"State": state},
            "NetworkBorderGroup": "us-east-1",
            "Ipv6Pool": "Amazon",
        }],
    }]}


def test_the_state_is_nested_not_top_level(association_members):
    # The fact the original query got wrong, taken from botocore rather than
    # from memory. If AWS ever adds a top-level State this test says so.
    assert "Ipv6CidrBlockState" in association_members
    assert "State" not in association_members


def test_the_deploy_query_finds_the_assigned_cidr():
    # The whole point: this returning None is indistinguishable from "AWS has
    # not assigned a block yet", which is why it went unnoticed for so long.
    assert jmespath.search(discovery_query(), describe_vpcs_response()) == ASSIGNED_CIDR


def test_the_deploy_query_ignores_a_block_still_associating():
    # Passing a CIDR that is not associated yet would make CloudFormation
    # carve /64s out of a block the VPC does not hold.
    assert jmespath.search(discovery_query(), describe_vpcs_response("associating")) is None


def test_the_deploy_query_is_empty_when_no_block_is_assigned():
    # The genuine first-deploy case, which the two-pass logic exists to handle.
    empty = {"Vpcs": [{"VpcId": "vpc-f3c92a8a", "Ipv6CidrBlockAssociationSet": []}]}
    assert jmespath.search(discovery_query(), empty) is None


def test_the_deploy_feeds_the_discovered_cidr_back_as_a_parameter():
    # Discovery is pointless if the result is not passed to the stack, and the
    # subnets get their /64 only from this parameter.
    assert "VpcIpv6Block=${EFFECTIVE_IPV6}" in WORKFLOW
    assert "VpcIpv6Block=${NEWLY_ASSIGNED}" in WORKFLOW


# ── The subnet's own /64 ─────────────────────────────────────────────────────
# A subnet accepts exactly one IPv6 CIDR. Once the VPC-level discovery above
# started working, the stack proposed the /64 it computes (index 8 and 9 of
# the /56) for subnets that already held one, and EC2 refused:
#
#   Subnet ID 'subnet-01714c...' has reached the limit of associated IPV6 CIDRs
#
# That is not an update EC2 can make — a different value is a *second*
# association. So the deploy discovers what each subnet already has and passes
# it, and the template prefers it over anything it would compute.


def subnet_query() -> str:
    match = re.search(r"--query '(Subnets\[0\]\.Ipv6CidrBlockAssociationSet[^']*)'", WORKFLOW)
    assert match, "the workflow no longer discovers the subnets' existing IPv6 CIDR"
    return match.group(1)


def describe_subnets_response(state: str = "associated", associations: int = 1) -> dict:
    assoc = [{
        "AssociationId": "subnet-cidr-assoc-0abc123",
        "Ipv6CidrBlock": "2600:1f18:abcd:ef08::/64",
        "Ipv6CidrBlockState": {"State": state},
    }][:associations]
    return {"Subnets": [{"SubnetId": "subnet-01714c0c1b4590e4e",
                         "Ipv6CidrBlockAssociationSet": assoc}]}


def test_the_subnet_association_state_is_nested_too():
    model = botocore.session.get_session().get_service_model("ec2")
    subnet = model.operation_model("DescribeSubnets").output_shape.members["Subnets"].member
    members = set(subnet.members["Ipv6CidrBlockAssociationSet"].member.members)
    assert "Ipv6CidrBlockState" in members
    assert "State" not in members


def test_the_deploy_finds_the_subnets_existing_cidr():
    assert jmespath.search(subnet_query(), describe_subnets_response()) == "2600:1f18:abcd:ef08::/64"


def test_a_subnet_with_no_association_yields_nothing():
    # The genuine first-deploy case: the template then computes the /64.
    assert jmespath.search(subnet_query(), describe_subnets_response(associations=0)) is None


def test_the_deploy_passes_both_subnet_cidrs_as_parameters():
    assert "LambdaSubnetIpv6CidrAz1=${SUBNET_IPV6_AZ1}" in WORKFLOW
    assert "LambdaSubnetIpv6CidrAz2=${SUBNET_IPV6_AZ2}" in WORKFLOW


def test_the_template_prefers_the_discovered_cidr_over_the_computed_one():
    """Order matters: computing first would re-introduce the failure.

    The discovered value has to win. A template that falls back to !Select
    whenever VpcIpv6Block is set would keep proposing a /64 the subnet does
    not have, which is the second association EC2 refuses.
    """
    shared = (REPO / "infra" / "shared.yaml").read_text()
    for condition, index in (("HasDiscoveredIpv6Az1", "8"), ("HasDiscoveredIpv6Az2", "9")):
        block = shared.split(f"- {condition}", 1)[1][:400]
        assert "LambdaSubnetIpv6Cidr" in block.split("HasVpcIpv6Block")[0], (
            f"{condition} must resolve to the discovered CIDR before any computed one"
        )
        assert f"!Select [{index}," in block


def test_a_discovered_cidr_alone_makes_the_subnet_dual_stack():
    # AssignIpv6AddressOnCreation gated only on VpcIpv6Block would leave a
    # subnet that has a /64 handing out ENIs with no IPv6 address.
    shared = (REPO / "infra" / "shared.yaml").read_text()
    for condition in ("HasDiscoveredIpv6Az1", "HasDiscoveredIpv6Az2"):
        assert shared.count(f"- {condition}") >= 2, (
            f"{condition} should gate both Ipv6CidrBlock and AssignIpv6AddressOnCreation"
        )


def test_the_second_pass_reuses_the_same_discovery():
    """The two-pass deploy has to look again with a query that works.

    On a first deploy the block does not exist until the stack requests it, so
    the re-deploy is the only chance to activate dual-stack in the same run.
    With the broken query the second look returned None as well, so the branch
    ran and concluded nothing had been assigned.
    """
    assert WORKFLOW.count("_discover_ipv6") >= 3   # definition + both calls
    assert "NEWLY_ASSIGNED=$(_discover_ipv6)" in WORKFLOW
