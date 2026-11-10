"""Offline checks for a TeamWeave team.json (no AWS required).

Used by ``examples/validate_team.py`` and unit tests. Mirrors the invariants
in ``tests/test_workflow_integrity.py`` so contributors can lint configs before
deploying to S3.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

AMBIENT_INPUTS = {
    "request",
    "rag_context",
    "research_context",
    "owner_profile_context",
    "gemini_brief",
    "owner",
    "rag_meta",
}


def load_team(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def lint_team(config: dict[str, Any]) -> list[str]:
    """Return human-readable problems; empty list means the config looks consistent."""
    problems: list[str] = []
    workflow = config.get("workflow") or []
    agents_list = config.get("agents") or []
    steps = [s["step"] for s in workflow]
    known = set(steps)
    agents = {a["id"] for a in agents_list}

    for step in workflow:
        sid = step.get("step", "?")
        if sid not in agents:
            problems.append(f"step '{sid}' has no matching agent id")
        for ref in step.get("inputs") or []:
            base = ref.split(".")[0]
            if base not in AMBIENT_INPUTS and base not in known:
                problems.append(f"step '{sid}' input '{ref}' names unknown step '{base}'")

    for aid in agents:
        if aid not in steps:
            problems.append(f"agent '{aid}' is never run by any workflow step")

    schemas = set((config.get("schemas") or {}).keys())
    for agent in agents_list:
        ref = agent.get("schema_ref")
        if ref and ref not in schemas:
            problems.append(f"agent '{agent.get('id')}' schema_ref '{ref}' is undefined")

    used_schemas = {a["schema_ref"] for a in agents_list if a.get("schema_ref")}
    for orphan in sorted(schemas - used_schemas):
        problems.append(f"schema '{orphan}' is defined but unused")

    target = ((config.get("globals") or {}).get("revision") or {}).get("default_jump_to_step")
    if target and target not in known:
        problems.append(f"revision.default_jump_to_step '{target}' is not in the workflow")

    return problems


def lint_team_file(path: Path) -> list[str]:
    try:
        config = load_team(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"could not read team.json: {exc}"]
    return lint_team(config)
