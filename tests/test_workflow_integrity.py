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


def test_the_visibility_team_is_five_members():
    config = load(TEAMS_DIR / "tarun_visibility_team" / "v1" / "team.json")
    assert len(config["agents"]) == 5, [a["id"] for a in config["agents"]]
    assert len(config["workflow"]) == 5


def test_everything_after_the_editor_consumes_the_approved_copy():
    """The rule trimming established, kept as the team grows.

    The pipeline used to end in distribution and approval steps that ran
    *after* the editor and produced a plan and a sign-off -- work about the
    deliverable rather than the deliverable. They were removed for that.

    A step may follow the editor, but only if it consumes what the editor
    approved: the illustrator does, which is why it is allowed to be last.
    A step that follows the editor and ignores its output is the old mistake
    coming back under a new name.
    """
    config = load(TEAMS_DIR / "tarun_visibility_team" / "v1" / "team.json")
    workflow = config["workflow"]
    agents = {a["id"]: a for a in config["agents"]}

    editor = next(s["step"] for s in workflow if agents[s["step"]]["schema_ref"] == "final_copy_v1")
    index = [s["step"] for s in workflow].index(editor)

    for step in workflow[index + 1:]:
        inputs = " ".join(step.get("inputs") or [])
        assert editor in inputs, (
            f"{agents[step['step']]['name']} runs after the editor without reading "
            f"its approved copy — the deliverable is the post, not work about it"
        )


def test_the_illustrator_is_not_an_agent_turn():
    """Image models do not implement Converse, which is all the AgentCore
    runtime program speaks. The modality is what routes it elsewhere; without
    it the worker would send an image model id through the agent runtime and
    the step would fail at the first call."""
    config = load(TEAMS_DIR / "tarun_visibility_team" / "v1" / "team.json")
    image_agents = [a for a in config["agents"]
                    if (a.get("bedrock") or {}).get("modality") == "image"]
    assert len(image_agents) == 1, [a["id"] for a in image_agents]


def test_the_image_members_provider_and_model_agree():
    """A Gemini model on the Bedrock provider (or the reverse) deploys fine
    and fails at the call, naming only the id -- which is how two deploys were
    spent on Bedrock ids. The pairing is checkable here."""
    from src.orchestrator.model_map import resolve_model

    config = load(TEAMS_DIR / "tarun_visibility_team" / "v1" / "team.json")
    agent = next(a for a in config["agents"]
                 if (a.get("bedrock") or {}).get("modality") == "image")
    bedrock = agent["bedrock"]
    choice = resolve_model(agent["model_category"], override=bedrock.get("model_id") or "")
    provider = bedrock.get("image_provider") or choice.provider

    assert provider in {"bedrock", "gemini"}, provider
    assert provider == choice.provider, (provider, choice.provider, choice.model_id)


def test_every_text_member_still_declares_a_text_model():
    """A modality typo would silently route a writer through the image path."""
    config = load(TEAMS_DIR / "tarun_visibility_team" / "v1" / "team.json")
    for agent in config["agents"]:
        modality = (agent.get("bedrock") or {}).get("modality", "text")
        assert modality in {"text", "image"}, (agent["id"], modality)
