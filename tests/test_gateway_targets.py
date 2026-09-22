"""The weave systems that are MCP servers, as AgentCore Gateway targets.

Only a system speaking MCP over **HTTP** can be one: a `GatewayTarget` takes
an `Endpoint`, so a stdio server has nothing to point at. DeployWeave is stdio
(`mcp.run()` with no transport) and is absent for that reason rather than by
oversight.

The defect these exist to stop is the one the gateway shipped with: it was
created with a single target, that target was gated on
`ScreenWeaveMcpEndpoint`, and **nothing in the deploy passed that parameter**.
So the condition was false on every deploy, the gateway had nothing behind it,
and no signal said so -- the parameter's own description already called that
"infrastructure for nothing".
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO / "infra" / "template.yaml").read_text()
WORKFLOW = (REPO / ".github" / "workflows" / "deploy.yml").read_text()


class CfnLoader(yaml.SafeLoader):
    """CloudFormation short forms, keeping the argument."""


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
def doc():
    return yaml.load(TEMPLATE, Loader=CfnLoader)


@pytest.fixture(scope="module")
def targets(doc):
    return {name: res for name, res in doc["Resources"].items()
            if res.get("Type") == "AWS::BedrockAgentCore::GatewayTarget"}


def mcp_parameters(doc):
    return sorted(p for p in doc["Parameters"] if p.endswith("McpEndpoint"))


# ── the gateway has something behind it ─────────────────────────────────────

def test_there_are_targets_at_all():
    """A gateway with no targets is infrastructure for nothing."""
    assert TEMPLATE.count("AWS::BedrockAgentCore::GatewayTarget") >= 1


def test_every_mcp_parameter_has_a_target(doc, targets):
    endpoints = set()
    for res in targets.values():
        ep = res["Properties"]["TargetConfiguration"]["Mcp"]["McpServer"]["Endpoint"]
        assert ep.get("__fn__") == "Ref", ep
        endpoints.add(ep["__arg__"])
    missing = sorted(set(mcp_parameters(doc)) - endpoints)
    assert not missing, f"declared but pointed at by no target: {missing}"


def test_every_target_points_at_its_own_parameter(doc, targets):
    """A target reading another's parameter deploys fine and sends every call
    for one tool to a different service. The runtime map had exactly this bug
    with two teams' substitutions crossed."""
    for name, res in targets.items():
        ep = res["Properties"]["TargetConfiguration"]["Mcp"]["McpServer"]["Endpoint"]["__arg__"]
        stem = name.replace("GatewayTarget", "")
        assert ep == f"{stem}McpEndpoint", f"{name} points at {ep}"


def test_every_target_is_gated_on_its_own_parameter(doc, targets):
    """The shipped bug, one level up: one shared condition gated every target
    on ScreenWeave's parameter, so a sibling that *was* configured still got
    no target whenever ScreenWeave was not."""
    conditions = doc["Conditions"]
    for name, res in targets.items():
        stem = name.replace("GatewayTarget", "")
        cond = conditions[res["Condition"]]
        assert f"{stem}McpEndpoint" in str(cond), (
            f"{name}'s condition {res['Condition']} does not test {stem}McpEndpoint"
        )


def test_no_two_targets_share_a_condition(doc, targets):
    used = [res["Condition"] for res in targets.values()]
    assert len(used) == len(set(used)), f"targets share a condition: {used}"


def test_every_target_names_a_distinct_tool(targets):
    names = [res["Properties"]["Name"] for res in targets.values()]
    assert len(names) == len(set(names)), names


# ── the deploy actually passes them ─────────────────────────────────────────

def test_the_deploy_passes_every_mcp_parameter(doc):
    """The original defect. The parameter existed, the target was gated on it,
    and nothing set it -- so the condition was false on every deploy and the
    gateway stayed empty while every signal read as success."""
    for param in mcp_parameters(doc):
        assert param in WORKFLOW, (
            f"{param} is declared and gates a target, but the deploy never "
            f"passes it -- the target is created on no deploy"
        )


def test_the_deploy_passes_no_parameter_the_template_does_not_declare(doc):
    """The other direction, which the check above cannot see.

    `sam deploy` refuses a `--parameter-overrides` key the template does not
    declare -- but only on a run where that sibling's stack actually resolves,
    so a spec added for an undeclared parameter deploys green until the day
    the sibling exists. Offline, both sides are readable now.
    """
    declared = set(mcp_parameters(doc))
    for param in re.findall(r"(\w+McpEndpoint)", WORKFLOW):
        assert param in declared, (
            f"the deploy passes {param}, which infra/template.yaml does not "
            f"declare -- sam deploy will reject it once the sibling resolves"
        )


def test_each_endpoint_is_read_from_a_sibling_stack_not_hardcoded(doc):
    """A pasted URL rots silently when a sibling is redeployed."""
    for param in mcp_parameters(doc):
        assert f"{param}\"" in WORKFLOW or f"{param}=" in WORKFLOW
    # No literal API hostnames among the resolution specs.
    specs = re.findall(r'"([a-z0-9-]+:[A-Za-z]+:[^:]*:\w+McpEndpoint)"', WORKFLOW)
    assert len(specs) >= len(mcp_parameters(doc)), specs
    for spec in specs:
        assert "execute-api" not in spec and "lambda-url" not in spec, spec


def test_a_missing_sibling_is_reported_rather_than_silent():
    assert "MCP_MISSING" in WORKFLOW
    assert "::warning::No MCP endpoint found for" in WORKFLOW


def test_an_empty_gateway_is_reported_rather_than_silent():
    assert "infrastructure for nothing" in WORKFLOW


def test_a_missing_sibling_does_not_fail_the_deploy():
    """One tool fewer is not a broken platform -- unlike the ContextWeave URL,
    which a declared RAG mode makes mandatory."""
    start = WORKFLOW.index("MCP_WIRED=()")
    # Search for the end marker *from* the start: `sam deploy \\` also appears in
    # the shared-stack step far earlier in the file, and anchoring on the first
    # occurrence produced a reversed slice -- an empty string, in which any
    # "not in" assertion passes. A mutation that inserted `exit 1` into the
    # loop survived this test until the search was anchored.
    end = WORKFLOW.index("sam deploy \\", start)
    block = WORKFLOW[start:end]
    assert "MCP_MISSING" in block, "the slice no longer covers the MCP resolution loop"
    assert "exit 1" not in block, "a missing MCP sibling must not fail the deploy"


# ── only HTTP MCP servers can be targets ────────────────────────────────────

def test_stdio_only_servers_are_not_targets(doc):
    """DeployWeave speaks MCP over stdio, so there is no endpoint to point a
    target at. Absent on purpose; a target for it could never resolve."""
    assert "DeployWeaveMcpEndpoint" not in doc["Parameters"]
    # And the deploy must not try to resolve one either: a spec here would
    # pass a parameter the template does not declare (see above) rather than
    # quietly doing nothing.
    assert "DeployWeaveMcpEndpoint" not in WORKFLOW
