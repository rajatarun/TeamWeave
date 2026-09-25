"""Every environment variable a handler needs is one the function is given.

A2AFunction was deployed with three missing. The stack was valid, cfn-lint was
clean, 844 tests passed, the deploy went green — and the card it served said

    "skills": []

while twelve agents were registered, because CONFIG_BUCKET was empty so it
read no team configs at all. `message:send` would have returned 500 on every
call for the same reason: no STATE_MACHINE_ARN. The entire A2A surface was
inert and every signal said it shipped.

This is the fourth thing a new function here turns out to need — after a
Makefile target (test_makefile_targets.py), a log group, and ENI permissions
(test_vpc_role_permissions.py) — and the fourth learned from production
rather than from the repository. So, like those, this derives the obligation
instead of listing functions: read each handler's module, collect what it
reads from the environment, and check the template supplies it.

A name is allowed to be unset only by appearing in OPTIONAL below with a
reason. That is the point: an unwired variable becomes a deliberate decision
someone wrote down, instead of a silent empty string.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "infra" / "template.yaml"

# Names a handler may read without the template supplying them, and why.
# Anything not here must be wired.
OPTIONAL = {
    # Lambda sets these itself, in every execution environment.
    "AWS_REGION": "set by the Lambda runtime",
    "AWS_DEFAULT_REGION": "set by the Lambda runtime",
    # Genuine feature switches and overrides, each with a working default.
    "A2A_BASE_URL": "overrides the URL otherwise derived from the request",
    "A2A_DISCOVERY": "opt-out switch; discovery is on by default",
    "TEAM_CONFIG_PREFIX": "accepted as an alias; the stack sets CONFIG_PREFIX",
    "SCHEMAS_DIR": "defaults to the schemas directory beside the module",
    "GEMINI_LAMBDA_ARN": "optional research integration; empty disables it. Set on the\n                          functions that use it, absent on ProvisionTeamFunction, where\n                          it degrades to off rather than failing",
    "FOUNDATION_MODEL": "has a default model id",
    "GEMINI_MODEL": "override of the model map's research_web category; unset uses the map",
    "STRUCTURED_TRANSFORM_MODEL_ID": "empty uses schema_repair; a value is a logged override",
    "ENRICH_MODEL": "override of schema_repair; unset uses the map",
    "IMAGE_MODEL_ID": "optional Bedrock image override; empty uses the map",
    "GEMINI_IMAGE_MODEL": "override of image_generation; unset uses the map",
    "VECTOR_EMBEDDING_MODEL_ID": "explicit pgvector embeddings run only when set",
    "MODEL_MAP_PATH": "tests point the loader at a fixture; production uses config/model_map.yaml",
    "AGENT_MODEL_ID": "set on AgentCore runtimes; the program's last resort when a turn omits modelId",
    "CONTEXTWEAVE_API_KEY": "optional; injected via Secrets Manager, not a parameter",
}


class CfnLoader(yaml.SafeLoader):
    """CloudFormation short forms. Values are irrelevant here — only the keys."""


CfnLoader.add_multi_constructor("!", lambda loader, suffix, node: {"__fn__": suffix})


@pytest.fixture(scope="module")
def template():
    return yaml.load(TEMPLATE.read_text(), Loader=CfnLoader)


def globals_vars(template) -> set[str]:
    return set(
        ((template.get("Globals") or {}).get("Function") or {})
        .get("Environment", {})
        .get("Variables") or {}
    )


def functions(template) -> dict[str, dict]:
    return {
        name: resource
        for name, resource in template["Resources"].items()
        if resource.get("Type") == "AWS::Serverless::Function"
    }


def env_names_read(handler: str) -> set[str]:
    """Environment variables the handler's module reads, from its source."""
    module_path = REPO / (handler.rsplit(".", 1)[0] + ".py")
    if not module_path.exists():
        return set()
    source = module_path.read_text()
    return (
        set(re.findall(r'os\.environ\.get\(\s*["\']([A-Z][A-Z0-9_]*)["\']', source))
        | set(re.findall(r'os\.environ\[\s*["\']([A-Z][A-Z0-9_]*)["\']', source))
    )


def test_the_scan_finds_functions_and_variables(template):
    # A loader or regex change could make every assertion below vacuous.
    found = functions(template)
    assert "A2AFunction" in found, "the function this test was written for"
    assert len(found) >= 8, f"suspiciously few functions: {sorted(found)}"
    assert len(globals_vars(template)) >= 5
    assert "CONFIG_BUCKET" in env_names_read("src/orchestrator/a2a_handler.handler")


@pytest.mark.parametrize("required", ["CONFIG_BUCKET", "STATE_MACHINE_ARN", "CONFIG_PREFIX"])
def test_the_a2a_function_has_what_it_needs(template, required):
    """The three that were missing, named individually.

    Without CONFIG_BUCKET the card advertises no skills; without
    STATE_MACHINE_ARN message:send returns 500. Both look like a working
    deployment from outside.
    """
    provided = set((functions(template)["A2AFunction"]["Properties"].get("Environment") or {})
                   .get("Variables") or {})
    assert required in provided | globals_vars(template)


def test_no_handler_reads_a_variable_its_function_is_not_given(template):
    gaps: list[str] = []
    for name, resource in sorted(functions(template).items()):
        handler = (resource.get("Properties") or {}).get("Handler", "")
        own = set((resource["Properties"].get("Environment") or {}).get("Variables") or {})
        supplied = own | globals_vars(template)
        for var in sorted(env_names_read(handler) - supplied):
            if var not in OPTIONAL:
                gaps.append(f"{name} reads {var} but nothing sets it")
    assert not gaps, (
        "these resolve to empty at run time, which fails silently rather than "
        f"loudly: {gaps} — wire them, or add the name to OPTIONAL with a reason"
    )


def test_every_optional_name_carries_a_reason():
    # An allowlist with blank entries is an allowlist nobody reviewed.
    for name, reason in OPTIONAL.items():
        assert reason.strip(), f"{name} is allowlisted with no reason"


def test_the_allowlist_cannot_be_used_to_silence_this(template):
    """The easiest way to make this test pass is to allowlist the problem.

    So the names whose absence is a silent outage are barred from OPTIONAL
    outright, and any name the template *does* supply is barred too — an
    allowlist entry for a wired variable is either stale or someone widening
    the exemption to quiet a failure.
    """
    never_optional = {"CONFIG_BUCKET", "STATE_MACHINE_ARN", "CONFIG_PREFIX",
                      "DDB_TABLE", "ARTIFACT_BUCKET"}
    overlap = never_optional & set(OPTIONAL)
    assert not overlap, (
        f"{sorted(overlap)} must be wired, not exempted — an empty value here "
        f"is an outage that reports success"
    )

    assert OPTIONAL, "an empty allowlist means the rule above checks nothing"
