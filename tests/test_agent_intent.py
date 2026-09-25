"""The personal teams do the work they were asked for.

daily_operator used to be told to keep the person's words and not rewrite a
task. Asked to plan or explore, it repeated the request into the triage
buckets. The same shape — a goal that restates instead of acting — was in the
other teams added with it.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.orchestrator.models import AgentConfig, BedrockRef, TeamConfig, TeamGlobals
from src.orchestrator.prompt_builder import build_prompt

TEAMS = Path("config/examples/teams")
PERSONAL = (
    "daily_operator",
    "linkedin_quick_post",
    "job_hunter",
    "health_prep",
    "financial_advisors",
)


def _load(name: str) -> dict:
    return json.loads((TEAMS / name / "v1" / "team.json").read_text())


def test_the_prompt_says_the_request_is_not_the_answer():
    team = TeamConfig(
        team={},
        globals=TeamGlobals(
            north_star="ship", default_channel="personal", hard_constraints=[],
            features={}, rag={"mode": "none"}, artifact_store={}, revision={},
        ),
        agents=[], workflow=[], schemas={},
    )
    agent = AgentConfig(
        id="a", name="Planner", bedrock=BedrockRef(agentId="", aliasId=""),
        goal_template="Plan the day.", schema_ref="day_plan_v1",
    )
    prompt = build_prompt(team, agent, {"request": {"dump": "plan Lisbon"}}, {}, "", "", "")
    assert "DO THE WORK" in prompt
    assert "not the answer" in prompt
    assert "mirroring the request is a failed turn" in prompt


def test_daily_operator_is_not_told_to_echo():
    doc = _load("daily_operator")
    goals = " ".join(a["goal_template"] for a in doc["agents"]).lower()
    constraints = " ".join(doc["globals"]["hard_constraints"]).lower()
    assert "keep their words" not in goals
    assert "plan" in constraints and "explore" in constraints
    plan = next(a for a in doc["agents"] if a["id"] == "DO_plan")
    assert "work is the substance" in plan["goal_template"]
    schema = json.loads(Path("config/examples/schemas/day_plan_v1.json").read_text())
    assert "work" in schema["required"]


def test_the_other_personal_teams_are_told_to_produce_the_work():
    writer = next(a for a in _load("linkedin_quick_post")["agents"] if a["id"] == "LQP_writer")
    assert "paraphrase of the input is not a post" in writer["goal_template"]
    fit = next(a for a in _load("job_hunter")["agents"] if a["id"] == "JH_fit")
    assert "restates the posting is not a fit read" in fit["goal_template"]
    insights = next(a for a in _load("health_prep")["agents"] if a["id"] == "HP_insights")
    assert "not an insight" in insights["goal_template"]
    assert "suggestions" in insights["goal_template"]
    assert "do not echo the request" in insights["goal_template"]
    log = next(a for a in _load("health_prep")["agents"] if a["id"] == "HP_log")
    assert "error" in log["goal_template"]
    assert "knowledge base" in log["goal_template"]
    insights = next(a for a in _load("financial_advisors")["agents"] if a["id"] == "FA_insights")
    assert "do not echo the request" in insights["goal_template"]
    assert "not an insight" in insights["goal_template"]
    assert "retrieved_on" in insights["goal_template"]


def test_personal_teams_name_a_model_category():
    """The model map chooses the id. A missing category would run the default."""
    from src.orchestrator.model_map import categories

    known = set(categories())
    for name in PERSONAL:
        for agent in _load(name)["agents"]:
            category = agent.get("model_category")
            assert category in known, agent["id"]
            assert "model_id" not in agent.get("bedrock", {}), agent["id"]
