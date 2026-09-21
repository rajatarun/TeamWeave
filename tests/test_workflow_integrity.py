"""A workflow may not reference a step that does not exist.

Trimming a team means deleting agents and their steps, and the risk is a
survivor still naming a dropped one in its `inputs`. The worker would resolve
that to nothing and hand the agent a prompt missing the context it was written
to expect — a worse answer, not an error, which is the failure mode this whole
codebase keeps producing.

Derived from the configs, so it covers teams that do not exist yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TEAMS_DIR = REPO / "config" / "examples" / "teams"

# Inputs the worker supplies itself, which name no step.
AMBIENT_INPUTS = {"request", "rag_context", "research_context", "owner_profile_context",
                  "gemini_brief", "owner", "rag_meta"}


def team_files() -> list[Path]:
    return sorted(TEAMS_DIR.glob("*/v1/team.json"))


def team_ids() -> list[str]:
    return [p.parent.parent.name for p in team_files()]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_there_are_teams_to_check():
    assert team_files(), "no team configs found — has the layout moved?"


@pytest.mark.parametrize("path", team_files(), ids=team_ids())
def test_every_step_input_names_a_real_step(path):
    config = load(path)
    steps = [s["step"] for s in config.get("workflow", [])]
    known = set(steps)
    dangling = []
    for step in config.get("workflow", []):
        for ref in step.get("inputs", []):
            base = ref.split(".")[0]
            if base in AMBIENT_INPUTS or base in known:
                continue
            dangling.append(f"{step['step']} <- {ref}")
    assert not dangling, (
        f"these inputs name steps the workflow does not contain, so the agent is "
        f"handed nothing where context should be: {dangling}"
    )


@pytest.mark.parametrize("path", team_files(), ids=team_ids())
def test_every_step_has_an_agent(path):
    config = load(path)
    agents = {a["id"] for a in config.get("agents", [])}
    orphans = [s["step"] for s in config.get("workflow", []) if s["step"] not in agents]
    assert not orphans, f"these steps have no agent defined: {orphans}"


@pytest.mark.parametrize("path", team_files(), ids=team_ids())
def test_every_agent_is_used_by_a_step(path):
    """An agent no step runs is dead weight that still gets registered."""
    config = load(path)
    steps = {s["step"] for s in config.get("workflow", [])}
    unused = [a["id"] for a in config.get("agents", []) if a["id"] not in steps]
    assert not unused, f"these agents are defined but never run: {unused}"


@pytest.mark.parametrize("path", team_files(), ids=team_ids())
def test_every_schema_ref_resolves(path):
    config = load(path)
    schemas = set(config.get("schemas", {}))
    missing = [
        f"{a['id']} -> {a['schema_ref']}"
        for a in config.get("agents", [])
        if a.get("schema_ref") and a["schema_ref"] not in schemas
    ]
    assert not missing, f"these agents validate against schemas the team does not define: {missing}"


@pytest.mark.parametrize("path", team_files(), ids=team_ids())
def test_no_orphan_schemas(path):
    # A schema nothing references is one a future step can be wired to by
    # mistake, expecting a producer that was deleted.
    config = load(path)
    used = {a["schema_ref"] for a in config.get("agents", []) if a.get("schema_ref")}
    orphans = sorted(set(config.get("schemas", {})) - used)
    assert not orphans, f"these schemas are defined but unused: {orphans}"


@pytest.mark.parametrize("path", team_files(), ids=team_ids())
def test_a_revision_jump_target_still_exists(path):
    """`default_jump_to_step` must survive a trim, or a revision goes nowhere."""
    config = load(path)
    target = ((config.get("globals") or {}).get("revision") or {}).get("default_jump_to_step")
    if not target:
        return
    steps = {s["step"] for s in config.get("workflow", [])}
    assert target in steps, f"revision jumps to {target}, which the workflow no longer contains"


def test_the_doc_rewrite_team_is_gone():
    # Removed deliberately; its runtime, its smoke-test role and its S3 copy
    # all went with it.
    assert not (TEAMS_DIR / "doc_rewrite_team").exists()
    template = (REPO / "infra" / "template.yaml").read_text()
    assert "DocRewriteTeam" not in template
    assert "doc_rewrite_team" not in template


def test_the_visibility_team_is_four_agents():
    config = load(TEAMS_DIR / "tarun_visibility_team" / "v1" / "team.json")
    assert len(config["agents"]) == 4, [a["id"] for a in config["agents"]]
    assert len(config["workflow"]) == 4


def test_the_visibility_pipeline_ends_with_the_post():
    """The last step's output is what a person came for.

    Trimming removed distribution and approval, which ran *after* the editor;
    the deliverable is the edited post, and the UI shows the final step.
    """
    config = load(TEAMS_DIR / "tarun_visibility_team" / "v1" / "team.json")
    last = config["workflow"][-1]["step"]
    agent = next(a for a in config["agents"] if a["id"] == last)
    assert agent["schema_ref"] == "final_copy_v1", (
        f"the pipeline ends at {agent['name']} producing {agent['schema_ref']}"
    )
