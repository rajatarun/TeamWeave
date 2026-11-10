#!/usr/bin/env python3
"""Build a sample A2A Agent Card from the example team configs (offline).

TeamWeave serves a live card from S3-loaded configs in production. This script
shows the same shape using the copies in ``config/examples/teams``.

Usage::

    python examples/generate_a2a_card.py
    python examples/generate_a2a_card.py --base-url https://api.example.com
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.orchestrator import a2a
from examples.team_lint import load_team


def load_example_teams() -> dict[str, dict]:
    teams: dict[str, dict] = {}
    root = REPO / "config" / "examples" / "teams"
    for path in sorted(root.glob("*/v1/team.json")):
        team_name = path.parent.parent.name
        config = load_team(path)
        teams[team_name] = config
    return teams


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an A2A Agent Card JSON document")
    parser.add_argument(
        "--base-url",
        default="https://YOUR_API_GATEWAY_URL",
        help="Base URL for supportedInterfaces (no trailing slash)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit minified JSON",
    )
    args = parser.parse_args()

    card = a2a.build_agent_card(
        base_url=args.base_url,
        version="examples",
        teams=load_example_teams(),
        documentation_url="https://github.com/rajatarun/TeamWeave",
    )
    indent = None if args.compact else 2
    print(json.dumps(card, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
