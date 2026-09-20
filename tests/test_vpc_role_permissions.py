"""A function in the VPC needs a role that can attach its ENI.

Lambda creates the elastic network interface using the function's execution
role, *before* any of its code runs. A role without `ec2:CreateNetworkInterface`
does not produce a function that fails at run time — it produces a function
that cannot be created:

    A2AFunction  CREATE_FAILED
    "The provided execution role does not have permissions to call
     CreateNetworkInterface on EC2"

and CloudFormation rolls the entire stack back. That happened to A2AFunction:
the template was valid, cfn-lint was clean, 781 tests passed, `sam build`
succeeded, and the deploy died six minutes in. Nothing in the repository knew
that declaring VpcConfig on a function obliges its role.

This is the third member of the same family — a new function needs a Makefile
target (test_makefile_targets.py), a log group, *and* ENI permissions, and
each one was learned from a red deploy. So this test derives the pairing from
the template rather than listing functions: a function added tomorrow is
covered the moment it declares VpcConfig.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "infra" / "template.yaml"

ENI_ACTION = "ec2:CreateNetworkInterface"
# Lambda needs all three over the ENI's life: one to attach, one to look it
# up, one to release it when the function is deleted or scaled in.
REQUIRED_ACTIONS = {
    "ec2:CreateNetworkInterface",
    "ec2:DescribeNetworkInterfaces",
    "ec2:DeleteNetworkInterface",
}


class CfnLoader(yaml.SafeLoader):
    """CloudFormation short forms, keeping the argument.

    The loader in test_agentcore_infra.py discards it, which is fine there.
    Here the argument is the whole point: `Role: !GetAtt A2ARole.Arn` has to
    yield "A2ARole" or there is no way to find the role a function uses.
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
    return yaml.load(TEMPLATE.read_text(), Loader=CfnLoader)["Resources"]


def role_of(function: dict) -> str | None:
    """The logical id of the role a function runs as, if it names one."""
    role = (function.get("Properties") or {}).get("Role")
    if isinstance(role, dict) and role.get("__fn__") == "GetAtt":
        arg = role.get("__arg__")
        if isinstance(arg, str):
            return arg.split(".")[0]
        if isinstance(arg, list) and arg:
            return str(arg[0])
    return None


def actions_granted(role: dict) -> set[str]:
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


def vpc_functions(resources: dict) -> dict[str, str]:
    """Functions that declare VpcConfig, mapped to their role's logical id."""
    out: dict[str, str] = {}
    for name, resource in resources.items():
        if resource.get("Type") != "AWS::Serverless::Function":
            continue
        if "VpcConfig" not in (resource.get("Properties") or {}):
            continue
        role = role_of(resource)
        if role:
            out[name] = role
    return out


def test_the_scan_finds_the_functions_it_is_meant_to_check(resources):
    # A loader change or a property rename could make every assertion below
    # vacuous, and a vacuous test is indistinguishable from a passing one.
    found = vpc_functions(resources)
    assert "A2AFunction" in found, "the function this test was written for"
    assert "WorkerFunction" in found
    assert len(found) >= 8, f"suspiciously few VPC functions found: {sorted(found)}"


def test_every_vpc_function_has_a_role_that_can_attach_an_eni(resources):
    missing = []
    for function, role_name in sorted(vpc_functions(resources).items()):
        role = resources.get(role_name)
        assert role is not None, f"{function} names a role {role_name} the template does not define"
        if ENI_ACTION not in actions_granted(role):
            missing.append(f"{function} -> {role_name}")
    assert not missing, (
        "these functions declare VpcConfig but their role cannot create the ENI, "
        f"so CloudFormation will fail to create them and roll the stack back: {missing}"
    )


def test_the_eni_permissions_cover_the_whole_lifecycle(resources):
    # Create alone gets the function deployed and then leaks interfaces:
    # delete is what releases them when the function is removed or scaled in.
    incomplete = []
    for function, role_name in sorted(vpc_functions(resources).items()):
        granted = actions_granted(resources[role_name])
        if ENI_ACTION in granted and not REQUIRED_ACTIONS <= granted:
            incomplete.append(f"{function} -> {role_name} missing {sorted(REQUIRED_ACTIONS - granted)}")
    assert not incomplete, incomplete


def test_ipv6_addressing_is_granted_where_the_subnets_are_dual_stack(resources):
    """Ipv6AllowedForDualStack needs ec2:AssignIpv6Addresses on the role.

    Egress here is an Egress-Only Internet Gateway, so a function whose ENI
    gets no IPv6 address has no route out at all for anything not pinned to
    IPv4 or served by a gateway endpoint.
    """
    for function, role_name in sorted(vpc_functions(resources).items()):
        vpc = resources[function]["Properties"]["VpcConfig"]
        if "Ipv6AllowedForDualStack" not in yaml.dump(vpc):
            continue
        granted = actions_granted(resources[role_name])
        assert "ec2:AssignIpv6Addresses" in granted, (
            f"{function} asks for an IPv6 address on its ENI but {role_name} "
            f"cannot assign one"
        )
