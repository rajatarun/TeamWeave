#!/usr/bin/env python3
"""Lint a team.json offline before uploading it to the config bucket.

Usage::

    python examples/validate_team.py \\
        config/examples/teams/daily_operator/v1/team.json

    python examples/validate_team.py --all-examples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from examples.team_lint import lint_team_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a TeamWeave team.json")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Path(s) to team.json files",
    )
    parser.add_argument(
        "--all-examples",
        action="store_true",
        help="Lint every team under config/examples/teams/*/v1/team.json",
    )
    args = parser.parse_args()

    if args.all_examples:
        paths = sorted((REPO / "config" / "examples" / "teams").glob("*/v1/team.json"))
    else:
        paths = [Path(p) for p in args.paths]

    if not paths:
        parser.error("pass at least one team.json path or use --all-examples")

    failed = False
    for path in paths:
        problems = lint_team_file(path)
        if problems:
            failed = True
            print(f"FAIL {path}")
            for line in problems:
                print(f"  - {line}")
        else:
            print(f"OK   {path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
