"""The AgentCore resources in the SAM template, against the published schemas.

A CloudFormation property name that does not exist is not caught by YAML
parsing or by `sam validate` alone -- it surfaces as a failed stack update,
which on an existing stack means a rollback of everything else in the deploy.
Checking the declarations against cfn-lint's resource schemas catches that
here, at no cost.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "infra" / "template.yaml"


class CfnLoader(yaml.SafeLoader):
    """CloudFormation short forms (!Ref, !Sub, ...) are not plain YAML."""


CfnLoader.add_multi_constructor("!", lambda loader, suffix, node: {"__fn__": suffix})


@pytest.fixture(scope="module")
def template():
    return yaml.load(TEMPLATE.read_text(), Loader=CfnLoader)


@pytest.fixture(scope="module")
def schemas():
    cfnlint_data = pytest.importorskip("cfnlint.data")
    base = os.path.dirname(cfnlint_data.__file__)
    out = {}
    for path in glob.glob(os.path.join(base, "schemas", "resources", "*.json")):
        try:
            doc = json.load(open(path))
        except Exception:
            continue
        name = doc.get("typeName", "")
        if name.startswith("AWS::BedrockAgentCore::"):
            out[name] = doc
    if not out:
        pytest.skip("cfn-lint ships no AgentCore schemas in this version")
    return out


def agentcore_resources(template):
    return {
        name: res
        for name, res in template["Resources"].items()
        if str(res.get("Type", "")).startswith("AWS::BedrockAgentCore::")
    }


# The substrate that is neither per team nor per tool: memory, the tool
# gateway, and the stack-wide runtime that serves a team with none of its own.
SHARED_AGENTCORE_RESOURCES = {
    "AgentCoreMemory", "AgentCoreRuntime", "AgentCoreGateway",
}


def test_the_template_declares_the_agentcore_substrate(template):
    """An inventory, so a new AgentCore resource is a deliberate addition.

    Per-team runtimes and per-sibling gateway targets are derived rather than
    listed — `tests/test_team_runtimes.py` holds the runtimes to the teams in
    config/examples/teams and `tests/test_gateway_targets.py` holds the targets
    to the MCP endpoint parameters, which are the couplings that matter.
    Listing them here too would mean two places to edit and one of them
    forgotten.
    """
    declared = set(agentcore_resources(template))
    per_team = {n for n in declared if n.startswith("AgentCoreRuntime") and n != "AgentCoreRuntime"}
    per_tool = {n for n in declared if n.endswith("GatewayTarget")}
    assert declared - per_team - per_tool == SHARED_AGENTCORE_RESOURCES
    assert per_team, "no per-team runtimes: every team would share one blast radius"
    assert per_tool, "no gateway targets: the gateway is infrastructure for nothing"


def test_every_agentcore_resource_matches_its_schema(template, schemas):
    for name, res in agentcore_resources(template).items():
        schema = schemas.get(res["Type"])
        assert schema, f"{name}: no published schema for {res['Type']}"
        props = res.get("Properties", {})

        missing = [r for r in schema.get("required", []) if r not in props]
        assert not missing, f"{name}: missing required {missing}"

        allowed = set(schema.get("properties", {}))
        unknown = [k for k in props if k not in allowed]
        assert not unknown, f"{name}: {unknown} are not properties of {res['Type']}"

        read_only = {c.split("/")[-1] for c in schema.get("readOnlyProperties", [])}
        assigned = [k for k in props if k in read_only]
        assert not assigned, f"{name}: {assigned} are read-only"


def test_the_runtime_code_configuration_is_complete(template, schemas):
    props = template["Resources"]["AgentCoreRuntime"]["Properties"]
    code = props["AgentRuntimeArtifact"]["CodeConfiguration"]
    required = schemas["AWS::BedrockAgentCore::Runtime"]["definitions"]["CodeConfiguration"]["required"]
    for key in required:
        assert key in code, f"CodeConfiguration missing {key}"


def test_enum_values_are_ones_the_schema_accepts(template, schemas):
    defs = schemas["AWS::BedrockAgentCore::Runtime"]["definitions"]
    props = template["Resources"]["AgentCoreRuntime"]["Properties"]
    runtime = props["AgentRuntimeArtifact"]["CodeConfiguration"]["Runtime"]
    assert runtime in defs["AgentManagedRuntimeType"]["enum"]
    assert props["NetworkConfiguration"]["NetworkMode"] in defs["NetworkMode"]["enum"]


def test_the_entrypoint_matches_a_module_that_exists(template):
    entry = template["Resources"]["AgentCoreRuntime"]["Properties"]["AgentRuntimeArtifact"]["CodeConfiguration"]["EntryPoint"]
    assert isinstance(entry, list) and entry, "EntryPoint must be a non-empty list"
    # A runtime pointing at a module that is not packaged fails at first
    # invocation, long after the deploy reports success.
    # The zip is flat, so the entrypoint names a file at its root. It must be
    # one the packaging step actually copies there.
    assert entry[0] == "app.py", "entrypoint must match the packaged layout"
    assert (REPO / "src" / "agentcore" / entry[0]).is_file()

    workflow = (REPO / ".github" / "workflows" / "deploy.yml").read_text()
    assert "src/agentcore/app.py src/agentcore/agent.py .agentcore-build/" in workflow, \
        "packaging step no longer copies the entrypoint to the zip root"


def test_everything_is_behind_the_feature_condition(template):
    # Still conditional, but now the condition is the rollback: setting
    # EnableAgentCore=false takes the whole substrate back to Classic in one
    # parameter rather than a revert.
    for name in ("AgentCoreMemory", "AgentCoreRuntime", "AgentCoreRuntimeRole",
                 "AgentCoreGateway", "AgentCoreGatewayRole"):
        assert template["Resources"][name].get("Condition") == "AgentCoreEnabled", name
    # Each target is narrower still: a target whose sibling published no
    # endpoint is infrastructure pointing nowhere, so every one carries a
    # condition of its own -- see tests/test_gateway_targets.py for which.
    targets = [n for n in template["Resources"] if n.endswith("GatewayTarget")]
    assert targets
    for name in targets:
        cond = template["Resources"][name].get("Condition")
        assert cond in template["Conditions"], f"{name} has no declared condition"
        assert cond != "AgentCoreEnabled", \
            f"{name} is gated only on the substrate, so it deploys with no endpoint"


def test_agentcore_is_the_default_substrate(template):
    # Classic is in maintenance with a frozen model catalogue; AgentCore is
    # the platform's substrate now. Classic stays selectable as a rollback.
    assert template["Parameters"]["EnableAgentCore"]["Default"] == "true"
    assert set(template["Parameters"]["EnableAgentCore"]["AllowedValues"]) == {"true", "false"}


def test_the_worker_is_told_which_substrate_and_where(template):
    # The runtime rejects every call without an ARN, so selecting agentcore
    # without passing the ARN would fail every step. They travel together.
    env = template["Globals"]["Function"]["Environment"]["Variables"]
    assert "AGENT_RUNTIME" in env
    assert "AGENTCORE_RUNTIME_ARN" in env


def test_the_conditional_outputs_are_guarded(template):
    for name in ("AgentCoreRuntimeArn", "AgentCoreMemoryId", "AgentCoreRuntimeRoleArn"):
        assert template["Outputs"][name].get("Condition") == "AgentCoreEnabled", name


def test_the_execution_role_is_assumable_only_by_agentcore(template):
    stmt = template["Resources"]["AgentCoreRuntimeRole"]["Properties"]["AssumeRolePolicyDocument"]["Statement"][0]
    assert stmt["Principal"]["Service"] == "bedrock-agentcore.amazonaws.com"
    # Confused-deputy guards: without these any account could have AgentCore
    # assume this role on their behalf.
    assert "aws:SourceAccount" in json.dumps(stmt["Condition"])
    assert "aws:SourceArn" in json.dumps(stmt["Condition"])


def test_the_workflow_packages_the_agent_before_deploying(template):
    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "deploy.yml").read_text())
    names = [s.get("name") for s in workflow["jobs"]["deploy"]["steps"]]
    assert names.index("Package the AgentCore agent") < names.index("SAM Deploy")


def test_the_workflow_serialises_deploys():
    # Three pushes in quick succession ran three deploys at once against one
    # stack and one provisioning Lambda.
    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "deploy.yml").read_text())
    assert workflow["concurrency"]["group"]
    assert workflow["concurrency"]["cancel-in-progress"] is False


def test_packaging_failure_stops_the_deploy():
    # AgentCore is the substrate now and the Runtime cannot create without
    # this artifact, so a packaging failure must stop the deploy rather than
    # let CloudFormation try and roll back.
    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "deploy.yml").read_text())
    step = next(s for s in workflow["jobs"]["deploy"]["steps"]
                if s.get("name") == "Package the AgentCore agent")
    assert step.get("continue-on-error") is not True
    assert "import app" in step["run"], "packaging must prove the entrypoint boots"


def test_the_memory_name_default_matches_the_service_pattern(template, schemas):
    # Hyphens are not allowed, which is why this cannot be built from a stack
    # name. Asserting the default against the published pattern keeps that
    # from being rediscovered in CI.
    import re

    schema = schemas["AWS::BedrockAgentCore::Memory"]
    prop = schema["properties"]["Name"]
    # The constraint lives behind a $ref, not inline on the property.
    if "$ref" in prop:
        prop = schema["definitions"][prop["$ref"].rsplit("/", 1)[-1]]
    pattern = prop.get("pattern")
    assert pattern, "schema no longer publishes a Name pattern"
    default = template["Parameters"]["AgentCoreMemoryName"]["Default"]
    assert re.match(pattern, default), f"{default!r} does not match {pattern}"


def test_the_expiry_default_and_bound_respect_the_service_minimum(template, schemas):
    schema_min = schemas["AWS::BedrockAgentCore::Memory"]["properties"]["EventExpiryDuration"].get("minimum")
    assert schema_min is not None
    param = template["Parameters"]["AgentCoreMemoryExpiryDays"]
    assert param["MinValue"] >= schema_min, "parameter allows a value the service rejects"
    assert param["Default"] >= schema_min


def test_the_gateway_name_default_matches_its_own_pattern(template, schemas):
    # Memory forbids hyphens, Gateway allows them. The two services differ, so
    # each default is checked against its own published pattern rather than
    # one assumption applied to both.
    import re

    schema = schemas["AWS::BedrockAgentCore::Gateway"]
    prop = schema["properties"]["Name"]
    if "$ref" in prop:
        prop = schema["definitions"][prop["$ref"].rsplit("/", 1)[-1]]
    pattern = prop.get("pattern")
    assert pattern
    assert re.match(pattern, template["Parameters"]["AgentCoreGatewayName"]["Default"])


def test_the_gateway_needs_no_external_identity_provider(template):
    # CUSTOM_JWT would mean introducing an IdP to reach a service the platform
    # already owns.
    assert template["Resources"]["AgentCoreGateway"]["Properties"]["AuthorizerType"] == "AWS_IAM"


def test_the_target_points_at_an_https_mcp_endpoint(template, schemas):
    import re

    endpoint_schema = schemas["AWS::BedrockAgentCore::GatewayTarget"]["definitions"]["McpServerTargetConfiguration"]
    assert "Endpoint" in endpoint_schema.get("required", []), "schema no longer requires Endpoint"
    targets = [n for n in template["Resources"] if n.endswith("GatewayTarget")]
    assert targets
    for name in targets:
        cfg = template["Resources"][name]["Properties"]["TargetConfiguration"]
        assert "McpServer" in cfg["Mcp"], name
        assert "Endpoint" in cfg["Mcp"]["McpServer"], name


def test_no_gateway_target_without_an_endpoint(template):
    # Every MCP endpoint parameter defaults to empty, so an ordinary enable
    # does not create a target pointing nowhere: a sibling the deploy could
    # not resolve simply contributes no tool.
    params = [n for n in template["Parameters"] if n.endswith("McpEndpoint")]
    assert params, "no MCP endpoint parameters: the gateway has nothing to target"
    for name in params:
        assert template["Parameters"][name]["Default"] == "", name


def test_the_gateway_role_keeps_the_confused_deputy_guards(template):
    stmt = template["Resources"]["AgentCoreGatewayRole"]["Properties"]["AssumeRolePolicyDocument"]["Statement"][0]
    assert stmt["Principal"]["Service"] == "bedrock-agentcore.amazonaws.com"
    assert "aws:SourceAccount" in json.dumps(stmt["Condition"])
    assert "aws:SourceArn" in json.dumps(stmt["Condition"])


def test_classic_provisioning_is_skipped_on_the_agentcore_path():
    """The provisioning step is Bedrock Agents Classic, and only that.

    It invokes ProvisionTeamFunction, which calls CreateAgent and writes
    agentId/aliasId back into team.json. An agent on AgentCore has no
    equivalent: its runtime is an AWS::BedrockAgentCore::Runtime created by
    this template, and its runtimeArn is a stack output rather than something
    a Lambda provisions. Leaving the step running after the substrate flips
    would rebuild Classic agents that nothing invokes.
    """
    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "deploy.yml").read_text())
    step = next(s for s in workflow["jobs"]["deploy"]["steps"]
                if s.get("name") == "Provision team agents")
    assert step.get("if"), "provisioning must be conditional on the substrate"
    assert "ENABLE_AGENTCORE" in step["if"]
    # The flag it branches on has to exist, or the condition is always true.
    assert workflow["env"]["ENABLE_AGENTCORE"] is not None


# ── Every AgentCore name, against its own published pattern ────────────────
#
# Two resources have now failed to create on exactly this: AWS::BedrockAgentCore
# ::Memory's Name and ::Runtime's AgentRuntimeName both forbid hyphens, and both
# were built from ${AWS::StackName} -- which is `tarun-content-team`. Neither
# name could ever have been created, and neither `sam validate` nor cfn-lint
# says so, because the template is valid: the value is only wrong once the
# stack name is substituted into it.
#
# The first was fixed with a parameter and a test written for that one
# property. That left the second, which is why this walks every AgentCore
# resource and checks every property the schema constrains, resolving !Ref and
# !Sub against the parameter defaults and the workflow's real STACK_NAME. A
# third resource added later is covered without anyone remembering to add a
# test.

class ResolvingLoader(yaml.SafeLoader):
    """Keeps the *value* of a CloudFormation short form, not just its name."""


def _keep(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {"__fn__": suffix, "__value__": value}


ResolvingLoader.add_multi_constructor("!", _keep)


def _workflow_stack_name() -> str:
    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "deploy.yml").read_text())
    return workflow["env"]["STACK_NAME"]


def _resolve(value, parameters, stack_name):
    """The string CloudFormation will actually send, or None if unknowable."""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict) or "__fn__" not in value:
        return None
    fn, inner = value["__fn__"], value.get("__value__")
    if fn == "Ref" and isinstance(inner, str):
        param = parameters.get(inner)
        return str(param["Default"]) if param and "Default" in param else None
    if fn == "Sub" and isinstance(inner, str):
        out = inner.replace("${AWS::StackName}", stack_name)
        for name, spec in parameters.items():
            if "Default" in spec:
                out = out.replace("${" + name + "}", str(spec["Default"]))
        # Anything still unresolved (${AWS::Region}, a GetAtt) is not a name
        # this test can judge.
        return None if "${" in out else out
    return None


def _constraint(schema, prop_name):
    prop = schema.get("properties", {}).get(prop_name)
    if not isinstance(prop, dict):
        return None
    if "$ref" in prop:
        prop = schema.get("definitions", {}).get(prop["$ref"].rsplit("/", 1)[-1], {})
    return prop if prop.get("pattern") else None


@pytest.fixture(scope="module")
def resolving_template():
    return yaml.load(TEMPLATE.read_text(), Loader=ResolvingLoader)


def test_every_agentcore_name_can_actually_be_created(resolving_template, schemas):
    import re

    stack_name = _workflow_stack_name()
    parameters = resolving_template.get("Parameters", {})
    checked = []
    problems = []

    for logical_id, resource in agentcore_resources(resolving_template).items():
        schema = schemas.get(resource["Type"])
        if not schema:
            continue
        for prop_name, raw in (resource.get("Properties") or {}).items():
            constraint = _constraint(schema, prop_name)
            if not constraint:
                continue
            resolved = _resolve(raw, parameters, stack_name)
            if resolved is None:
                continue
            checked.append(f"{logical_id}.{prop_name}")
            pattern = constraint["pattern"]
            if not re.fullmatch(pattern, resolved):
                problems.append(
                    f"{logical_id}.{prop_name} resolves to {resolved!r}, which does not "
                    f"match the service pattern {pattern!r}"
                )
            maximum = constraint.get("maxLength")
            if maximum and len(resolved) > maximum:
                problems.append(
                    f"{logical_id}.{prop_name} resolves to {len(resolved)} characters, "
                    f"over the {maximum} the service accepts"
                )

    assert not problems, "\n".join(problems)
    # A test that silently checked nothing would pass for the wrong reason --
    # and the two names that failed in production are exactly what it must see.
    assert "AgentCoreRuntime.AgentRuntimeName" in checked, checked
    assert "AgentCoreMemory.Name" in checked, checked


def test_no_agentcore_name_is_built_from_the_stack_name(resolving_template, schemas):
    # The root cause rather than the symptom. These names forbid hyphens,
    # stack names routinely contain them, and the result is a template that
    # validates cleanly and cannot deploy. Naming them by parameter keeps the
    # allowed alphabet visible where the value is written.
    for logical_id, resource in agentcore_resources(resolving_template).items():
        schema = schemas.get(resource["Type"])
        if not schema:
            continue
        for prop_name, raw in (resource.get("Properties") or {}).items():
            if not _constraint(schema, prop_name):
                continue
            if isinstance(raw, dict) and raw.get("__fn__") == "Sub":
                assert "${AWS::StackName}" not in str(raw.get("__value__")), (
                    f"{logical_id}.{prop_name} is built from the stack name, whose "
                    f"hyphens this property does not allow"
                )
