"""Where the deploy writes in the config bucket, and where the Lambda reads.

`GET /teams` returned `{"teams": [], "count": 0}` with a 200. No error, no
exception, an empty team picker in the UI and empty Roles and Departments
tabs — the API answered honestly that it found nothing, because it was
looking in the wrong place.

The provisioner derives three keys from one variable:

    teams_prefix     = "{OUTPUT_PREFIX}/teams"
    roles_key        = "{OUTPUT_PREFIX}/roles.json"
    departments_key  = "{OUTPUT_PREFIX}/departments.json"

`OUTPUT_PREFIX` was "teams", so it scanned `teams/teams/` for team configs
while the deploy wrote them to `teams/<name>/<version>/team.json`, and looked
for `teams/roles.json` while the deploy put roles at the bucket root.

Neither side is wrong on its own, which is why nothing caught it: the workflow
uploads exactly where it says, the Lambda reads exactly where it is told, and
only the pair is broken. So this test computes **both** sides from their real
sources and compares them.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO / "infra" / "template.yaml").read_text()
WORKFLOW = (REPO / ".github" / "workflows" / "deploy.yml").read_text()
PROVISIONER = (REPO / "config" / "examples" / "lambda_handler.py").read_text()


def output_prefix() -> str:
    """OUTPUT_PREFIX as the template gives it to ProvisionTeamFunction."""
    after = TEMPLATE.split("  ProvisionTeamFunction:", 1)[1]
    # Stop at the next top-level resource, not at the first two-space indent --
    # which is the very next line and yields an empty block.
    block = re.split(r"\n  \w+:\n    Type: ", after)[0]
    assert len(block) > 200, "the ProvisionTeamFunction block did not parse"
    match = re.search(r'^\s+OUTPUT_PREFIX:\s*(.*)$', block, re.M)
    assert match, "ProvisionTeamFunction no longer sets OUTPUT_PREFIX"
    value = match.group(1).strip()
    # Strip a comment tail and quotes.
    value = value.split("#", 1)[0].strip()
    return value.strip('"').strip("'")


def lambda_keys(prefix: str) -> dict[str, str]:
    """The keys the provisioner builds, using its own expressions."""
    return {
        "teams_prefix": f"{prefix}/teams" if prefix else "teams",
        "roles_key": f"{prefix}/roles.json" if prefix else "roles.json",
        "departments_key": f"{prefix}/departments.json" if prefix else "departments.json",
    }


def workflow_team_prefix() -> str:
    """The prefix sync_team_configs.py is invoked with."""
    match = re.search(r"^\s*TEAM_CONFIG_PREFIX:\s*(\S+)\s*$", WORKFLOW, re.M)
    assert match, "the workflow no longer sets TEAM_CONFIG_PREFIX"
    return match.group(1).strip().strip('"').strip("'")


def workflow_upload_key(filename: str) -> str:
    """The key an `aws s3 cp` in the workflow writes to."""
    match = re.search(rf'"s3://\$\{{CONFIG_BUCKET\}}/([^"]*{re.escape(filename)})"', WORKFLOW)
    assert match, f"the workflow no longer uploads {filename}"
    return match.group(1)


def test_the_derivations_find_real_values():
    # Every assertion below is vacuous if the parsing silently returns "".
    assert workflow_team_prefix() == "teams"
    assert workflow_upload_key("roles.json")
    assert "OUTPUT_PREFIX" in TEMPLATE


def test_the_provisioner_still_derives_its_keys_this_way():
    """If the Lambda changes how it builds keys, this test must change too."""
    assert 'f"{prefix}/teams" if prefix else "teams"' in PROVISIONER
    assert 'f"{prefix}/roles.json" if prefix else "roles.json"' in PROVISIONER


def test_the_lambda_scans_where_the_deploy_writes_team_configs():
    """The one that was broken. Team configs land under `teams/`."""
    keys = lambda_keys(output_prefix())
    assert keys["teams_prefix"] == workflow_team_prefix(), (
        f"the deploy writes team configs under {workflow_team_prefix()!r} but the "
        f"provisioner scans {keys['teams_prefix']!r}, so GET /teams returns an "
        f"empty list with a 200 and the UI shows no teams"
    )


def test_the_lambda_reads_roles_where_the_deploy_puts_them():
    keys = lambda_keys(output_prefix())
    assert keys["roles_key"] == workflow_upload_key("roles.json")


def test_the_lambda_reads_departments_where_the_deploy_puts_them():
    keys = lambda_keys(output_prefix())
    assert keys["departments_key"] == workflow_upload_key("departments.json")


def test_the_scan_shape_matches_what_the_sync_writes():
    """`{teams_prefix}/{team}/{version}/team.json`, three path segments deep.

    The scanner keeps only keys that split into exactly three parts with the
    last starting with "team", so a prefix that is off by one segment yields
    nothing rather than an error.
    """
    assert 'len(parts) == 3 and parts[2].startswith("team")' in PROVISIONER
    sync = (REPO / "scripts" / "sync_team_configs.py").read_text()
    assert 'f"{prefix}/{rel}" if prefix else rel' in sync


def test_the_upload_message_names_the_key_it_writes():
    """The log said `teams/` while copying to the root.

    A message that contradicts the command sends whoever reads it looking in
    the wrong place — which is the same failure as the prefix itself.
    """
    line = next(l for l in WORKFLOW.splitlines() if "Uploading roles.json" in l)
    assert "/teams/" not in line, f"the message still claims a prefix the command does not use: {line}"
