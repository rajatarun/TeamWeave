"""Every makefile-built function needs a target, or SAM Build fails.

`BuildMethod: makefile` makes SAM shell out to `make build-<LogicalId>`. Add a
function with that metadata and no matching target and the template is valid,
cfn-lint is clean, every unit test passes -- and `sam build` fails on a
missing rule, after CI has already spent a couple of minutes getting there.

That is exactly how A2AFunction landed: nine functions had a target, the tenth
did not, and nothing in the repository knew the two lists had to agree.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "infra" / "template.yaml"
MAKEFILE = REPO / "Makefile"


class CfnLoader(yaml.SafeLoader):
    pass


CfnLoader.add_multi_constructor("!", lambda loader, suffix, node: {"__fn__": suffix})


@pytest.fixture(scope="module")
def template():
    return yaml.load(TEMPLATE.read_text(), Loader=CfnLoader)


@pytest.fixture(scope="module")
def make_targets() -> set[str]:
    return set(re.findall(r"^build-(\w+):", MAKEFILE.read_text(), re.M))


def makefile_built(template) -> set[str]:
    return {
        name
        for name, res in template["Resources"].items()
        if res.get("Type") == "AWS::Serverless::Function"
        and str((res.get("Metadata") or {}).get("BuildMethod", "")).lower() == "makefile"
    }


def test_every_makefile_built_function_has_a_target(template, make_targets):
    missing = makefile_built(template) - make_targets
    assert not missing, (
        f"these functions use BuildMethod: makefile but the Makefile has no "
        f"build-<name> target, so `sam build` fails: {sorted(missing)}"
    )


def test_the_check_is_actually_looking_at_something(template, make_targets):
    # A regex that matched nothing would make the test above vacuous.
    functions = makefile_built(template)
    assert len(functions) >= 5, f"only found {functions}"
    assert "A2AFunction" in functions
    assert len(make_targets) >= 5


def test_no_target_exists_for_a_function_that_does_not(template, make_targets):
    # A stale target is dead weight rather than a failure, but it means the
    # Makefile and the template have already drifted once.
    known = set(template["Resources"])
    # OrchestratorFunction is a historical alias kept for local builds.
    stale = {t for t in make_targets if t not in known} - {"OrchestratorFunction"}
    assert not stale, f"Makefile builds functions the template does not define: {sorted(stale)}"
