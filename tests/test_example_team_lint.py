"""Examples/team_lint stays aligned with shipped example teams."""
from __future__ import annotations

from pathlib import Path

import pytest

from examples.team_lint import lint_team_file

REPO = Path(__file__).resolve().parents[1]
TEAMS = sorted((REPO / "config" / "examples" / "teams").glob("*/v1/team.json"))


@pytest.mark.parametrize("path", TEAMS, ids=[p.parent.parent.name for p in TEAMS])
def test_example_teams_pass_offline_lint(path):
    problems = lint_team_file(path)
    assert problems == [], problems
