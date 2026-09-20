"""TeamWeave as an A2A agent.

A2A 1.0.0 (Linux Foundation, January 2026) is how agents on different stacks
discover and call each other, which is the weave platform's whole premise --
until now a sibling service reached TeamWeave through a hardcoded URL and
private knowledge of its payload shape.

The card is built against the published v1.0 specification, and v1.0 changed
its shape: `supportedInterfaces` replaced the top-level `url` and
`preferredTransport` of 0.3.x, each entry carrying its own `protocolBinding`
and `protocolVersion`. Writing a 0.3-shaped card would parse and mean nothing
to a 1.0 client, so the shape itself is pinned here.

The other thing pinned is honesty: the card must not advertise a transport or
a capability TeamWeave does not serve. A lie in a machine-readable document is
one a machine acts on.
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from src.orchestrator import a2a  # noqa: E402

TEAMS = {
    "teams/visibility/v1/team.json": {
        "team": {"name": "visibility"},
        "agents": [
            {"id": "strategist", "name": "Brand Strategist", "role_id": "PBM-001",
             "goal_template": "Decide the week's content angle.", "schema_ref": "creative_brief_v1"},
            {"id": "writer", "goal_template": "Draft the post."},
        ],
    },
    "teams/improve/v1/team.json": {
        "team": {"name": "improvement"},
        "agents": [{"id": "coach", "goal_template": "Plan the week."}],
    },
}


@pytest.fixture(scope="module")
def card():
    return a2a.build_agent_card(base_url="https://api.example.com/prod",
                                version="1.4.0", teams=TEAMS)


# ── the v1.0 card shape ────────────────────────────────────────────────────

def test_the_card_carries_the_five_required_fields(card):
    for field in ("name", "description", "version", "skills", "supportedInterfaces"):
        assert card.get(field), f"{field} is required on an AgentCard"


def test_interfaces_use_the_v1_shape_not_the_0_3_one(card):
    # v1.0 replaced top-level url/preferredTransport with supportedInterfaces.
    # A 0.3-shaped card parses fine and tells a 1.0 client nothing.
    assert "url" not in card
    assert "preferredTransport" not in card
    interface = card["supportedInterfaces"][0]
    assert interface["protocolBinding"] == "HTTP+JSON"
    assert interface["protocolVersion"] == "1.0"
    assert interface["url"] == "https://api.example.com/prod/a2a"


def test_only_the_transport_that_exists_is_declared(card):
    # "Each interface MUST accurately declare its transport protocol and URL."
    # Declaring JSONRPC or GRPC would strand any client that preferred one.
    assert len(card["supportedInterfaces"]) == 1


def test_capabilities_are_not_overclaimed(card):
    # message:stream is not implemented, so streaming is false. A client that
    # believed otherwise would wait on a stream that never opens.
    assert card["capabilities"]["streaming"] is False
    assert card["capabilities"]["pushNotifications"] is False


def test_content_modes_are_declared(card):
    assert "application/json" in card["defaultInputModes"]
    assert "application/json" in card["defaultOutputModes"]


# ── skills are the agents ──────────────────────────────────────────────────

def test_every_agent_becomes_a_skill(card):
    assert {s["id"] for s in card["skills"]} == {
        "visibility.strategist", "visibility.writer", "improvement.coach"}


def test_a_skill_carries_the_four_required_fields(card):
    skill = next(s for s in card["skills"] if s["id"] == "visibility.strategist")
    for field in ("id", "name", "description", "tags"):
        assert skill.get(field), f"{field} is required on an AgentSkill"


def test_skill_ids_are_team_scoped():
    # The same agent id in two teams is two skills; a card's skill ids must be
    # unique or a caller cannot address one of them.
    teams = {
        "a/team.json": {"team": {"name": "alpha"}, "agents": [{"id": "writer"}]},
        "b/team.json": {"team": {"name": "beta"}, "agents": [{"id": "writer"}]},
    }
    assert len(a2a.skills_from_teams(teams)) == 2


def test_a_skill_uses_the_agents_display_name(card):
    skill = next(s for s in card["skills"] if s["id"] == "visibility.strategist")
    assert skill["name"] == "Brand Strategist"


def test_the_goal_template_becomes_the_description(card):
    skill = next(s for s in card["skills"] if s["id"] == "visibility.writer")
    assert "Draft the post." in skill["description"]


def test_a_schema_constrained_agent_says_so(card):
    # A caller needs to know the output is validated before it sends, not
    # after it fails.
    skill = next(s for s in card["skills"] if s["id"] == "visibility.strategist")
    assert "creative_brief_v1" in skill["description"]


def test_an_agent_with_no_identifier_is_skipped():
    teams = {"a/team.json": {"team": {"name": "x"}, "agents": [{"goal_template": "no id"}]}}
    assert a2a.skills_from_teams(teams) == []


def test_the_card_does_not_depend_on_the_order_teams_were_read_in():
    """The spec asks clients to cache the card; one that reorders itself every
    deploy defeats that.

    Comparing two calls on the same dict only proved the function is
    deterministic -- reversing the list passed that happily. S3 listing order
    is the thing that actually varies, so the inputs have to differ.
    """
    forward = dict(TEAMS)
    reversed_order = {k: TEAMS[k] for k in reversed(list(TEAMS))}
    assert json.dumps(a2a.skills_from_teams(forward)) == \
           json.dumps(a2a.skills_from_teams(reversed_order))


def test_skills_are_in_sorted_id_order():
    ids = [s["id"] for s in a2a.skills_from_teams(TEAMS)]
    assert ids == sorted(ids)


# ── runs as tasks ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("run_status,expected", [
    ("RUNNING", "TASK_STATE_WORKING"),
    ("SUCCEEDED", "TASK_STATE_COMPLETED"),
    ("FAILED", "TASK_STATE_FAILED"),
    ("TIMED_OUT", "TASK_STATE_FAILED"),
    ("ABORTED", "TASK_STATE_CANCELED"),
    ("SUBMITTED", "TASK_STATE_SUBMITTED"),
])
def test_run_status_maps_to_the_conventional_task_state(run_status, expected):
    assert a2a.task_state_for(run_status) == expected


def test_an_unknown_status_is_unspecified_not_guessed():
    # Step Functions has more statuses than the handler surfaces. Reporting an
    # unmapped one as completed would be worse than reporting it as unknown.
    assert a2a.task_state_for("PENDING_REDRIVE") == "TASK_STATE_UNSPECIFIED"
    assert a2a.task_state_for("") == "TASK_STATE_UNSPECIFIED"


def test_terminal_states_are_the_ones_the_spec_names():
    assert a2a.is_terminal("TASK_STATE_COMPLETED")
    assert a2a.is_terminal("TASK_STATE_FAILED")
    assert a2a.is_terminal("TASK_STATE_CANCELED")
    assert not a2a.is_terminal("TASK_STATE_WORKING")


def test_a_completed_run_returns_its_output_as_an_artifact():
    task = a2a.task_for_run("run-1", "SUCCEEDED", result={"post": "hello"})
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"][0]["text"] == '{"post": "hello"}'


def test_a_failed_run_carries_the_reason():
    task = a2a.task_for_run("run-1", "FAILED", error="StepFailed: boom")
    assert "boom" in task["status"]["message"]["parts"][0]["text"]
    assert task["status"]["message"]["role"] == "ROLE_AGENT"


def test_a_working_task_has_no_artifacts_even_when_output_exists():
    # Artifacts on an unfinished task read as a result that is final.
    # Step Functions exposes partial output on a running execution, so the
    # guard has to be the *state*, not merely the absence of a result --
    # passing no result at all tested nothing.
    task = a2a.task_for_run("run-1", "RUNNING", result={"partial": "draft"})
    assert "artifacts" not in task
    assert a2a.task_for_run("run-2", "FAILED", result={"x": 1}).get("artifacts") is None


def test_a_task_always_has_a_context_id():
    assert a2a.task_for_run("run-1", "RUNNING")["contextId"] == "run-1"


# ── message parsing ────────────────────────────────────────────────────────

def test_text_parts_are_joined():
    assert a2a.text_of_message(
        {"parts": [{"text": "one"}, {"text": "two"}]}) == "one\ntwo"


def test_non_text_parts_are_ignored_not_fatal():
    # Parts may carry files or structured data; a caller sending one alongside
    # text must not lose the text.
    assert a2a.text_of_message(
        {"parts": [{"file": {"uri": "s3://x"}}, {"text": "hi"}]}) == "hi"


def test_a_message_with_nothing_usable_is_empty_not_an_error():
    assert a2a.text_of_message({}) == ""
    assert a2a.text_of_message({"parts": [{"file": {"uri": "s3://x"}}]}) == ""


# ── the spec and the card must agree ───────────────────────────────────────
#
# A card is a contract a machine reads. If openapi/ says one thing and the
# handler serves another, a sibling service written against the spec breaks on
# a field it was promised.

import yaml  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spec():
    return yaml.safe_load((REPO / "openapi" / "teamweave.yaml").read_text())


def test_the_spec_documents_the_well_known_card(spec):
    assert "/.well-known/agent-card.json" in spec["paths"]


def test_the_documented_card_fields_are_the_ones_served(spec, card):
    schema = spec["components"]["schemas"]["A2AAgentCard"]
    for field in schema["required"]:
        assert field in card, f"spec requires {field} but the card omits it"
    for field in card:
        assert field in schema["properties"], f"the card serves {field}; the spec never mentions it"


def test_the_documented_skill_fields_are_the_ones_served(spec, card):
    skill_schema = spec["components"]["schemas"]["A2AAgentCard"]["properties"]["skills"]["items"]
    for field in skill_schema["required"]:
        assert field in card["skills"][0]
    for field in card["skills"][0]:
        assert field in skill_schema["properties"], f"skills carry {field}; the spec never mentions it"


def test_the_spec_only_admits_the_binding_that_is_served(spec, card):
    declared = spec["components"]["schemas"]["A2AAgentCard"]["properties"][
        "supportedInterfaces"]["items"]["properties"]["protocolBinding"]["enum"]
    assert declared == [card["supportedInterfaces"][0]["protocolBinding"]]


def test_every_task_state_the_mapping_can_produce_is_documented(spec):
    documented = set(
        spec["components"]["schemas"]["A2ATask"]["properties"]["task"]
        ["properties"]["status"]["properties"]["state"]["enum"]
    )
    produced = set(a2a._RUN_STATUS_TO_TASK_STATE.values()) | {a2a.TASK_UNKNOWN}
    assert produced <= documented, f"undocumented states: {produced - documented}"
