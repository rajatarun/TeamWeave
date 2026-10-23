"""A Lambda function's environment is capped at 4096 bytes, and CloudFormation finds out last.

The cap is on the sum of every variable's name and value. Nothing earlier
checks it: the template is valid, cfn-lint is clean, the tests pass, and the
update fails minutes into `sam deploy` with

    Lambda was unable to configure your environment variables because the
    environment variables you have provided exceeded the 4KB limit.
    Measured size: 4472 bytes

and rolls the stack back. That is what adding four per-sibling secret ARNs
(about 570 bytes) to the worker did. The values that decide the size are
resolved at deploy time (ARNs, URLs), so this estimates them:

- a Ref/GetAtt to a resource is assumed to be a 110-byte ARN;
- a parameter takes its default, and an empty default (filled in by the
  deploy) is assumed to be 44 bytes.

The assumptions were fitted to that one measurement: this estimate puts the
failing configuration at 4438 bytes against the 4472 Lambda reported, so it
runs about 1% low. The budget leaves more than that below the cap. When this
fails, shorten or drop a variable (one prefix instead of a list of ARNs)
before raising anything.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
LIMIT = 4096
BUDGET = 4000
ARN_BYTES = 110
EMPTY_PARAMETER_BYTES = 44


class _Loader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {"Ref" if suffix == "Ref" else f"Fn::{suffix}": value}


_Loader.add_multi_constructor("!", _tag)
TEMPLATE = yaml.load((REPO / "infra" / "template.yaml").read_text(), Loader=_Loader)
PARAMETERS = TEMPLATE["Parameters"]
RESOURCES = TEMPLATE["Resources"]
GLOBAL_ENV = TEMPLATE.get("Globals", {}).get("Function", {}).get("Environment", {}).get("Variables", {})
NO_VALUE = {"Ref": "AWS::NoValue"}


def value_bytes(value) -> int:
    if isinstance(value, (str, int, float, bool)):
        return len(str(value))
    if "Ref" in value:
        name = value["Ref"]
        if name in PARAMETERS:
            default = str(PARAMETERS[name].get("Default", ""))
            return len(default) if default else EMPTY_PARAMETER_BYTES
        return 20 if name.startswith("AWS::") else ARN_BYTES
    if "Fn::GetAtt" in value:
        return ARN_BYTES
    if "Fn::Sub" in value:
        template = value["Fn::Sub"]
        template = template if isinstance(template, str) else template[0]
        return len(re.sub(r"\$\{[^}]+\}", "x" * 25, template))
    if "Fn::If" in value:
        _condition, yes, no = value["Fn::If"]
        return max(value_bytes(yes), value_bytes(no))
    return 200  # Join and anything else: assume it is long


def environment_bytes(logical_id: str) -> int:
    own = RESOURCES[logical_id]["Properties"].get("Environment", {}).get("Variables", {})
    merged = {**GLOBAL_ENV, **own}
    return sum(len(name) + value_bytes(value) for name, value in merged.items() if value != NO_VALUE)


FUNCTIONS = sorted(
    name for name, resource in RESOURCES.items()
    if resource["Type"] == "AWS::Serverless::Function"
)


@pytest.mark.parametrize("logical_id", FUNCTIONS)
def test_a_functions_environment_fits_under_the_cap(logical_id):
    size = environment_bytes(logical_id)
    assert size <= BUDGET, (
        f"{logical_id}'s environment is about {size} bytes; Lambda refuses {LIMIT} and the "
        f"deploy only finds out after it has started. Shorten or drop a variable.")


def test_the_estimate_flags_the_change_that_broke_the_deploy():
    """The four per-sibling ARNs, as they were, put the worker over the budget."""
    worker = RESOURCES["WorkerFunction"]["Properties"]["Environment"]["Variables"]
    removed = {key: value for key, value in worker.items() if key.startswith("CIPHERWEAVE_")}
    assert set(removed) == {"CIPHERWEAVE_TOKEN_URL", "CIPHERWEAVE_SECRET_PREFIX"}
    four_arns = {f"CIPHERWEAVE_{name}_SECRET_ARN": {"Ref": f"CipherWeaveClientSecret{name.capitalize()}"}
                 for name in ("SCREENWEAVE", "TOOLWEAVE", "CIPHERWEAVE", "DEVICEWEAVE")}
    old = (environment_bytes("WorkerFunction")
           - len("CIPHERWEAVE_SECRET_PREFIX") - value_bytes(removed["CIPHERWEAVE_SECRET_PREFIX"])
           + sum(len(k) + value_bytes(v) for k, v in four_arns.items()))
    assert old > LIMIT, old
