"""infra/template.yaml must pass cfn-lint before a deploy.

The deploy job runs ``sam validate --lint`` after pytest, and only on a
push to main. ``ScalingConfig.MaximumConcurrency: 1`` is below that
property's minimum of 2, so the linter failed the tarun-content-team
deploy with E3034 after the change had already merged. This test runs the
same linter during pytest. Warnings stay warnings; an error fails the run.
"""
from __future__ import annotations

from pathlib import Path

import cfnlint.api

TEMPLATE = Path(__file__).resolve().parents[1] / "infra" / "template.yaml"


def test_the_template_passes_cfn_lint():
    matches = cfnlint.api.lint(TEMPLATE.read_text(), regions=["us-east-1"])
    errors = [match for match in matches if match.rule.severity == "error"]
    rendered = "\n".join(
        f"{match.rule.id} line {match.linenumber}: {match.message}"
        for match in errors
    )
    assert not errors, f"cfn-lint reported errors:\n{rendered}"
