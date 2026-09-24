"""The health team returns insights and suggestions grounded in the records.

The last step used to require `questions_to_ask`. A schema-valid run was a
list of questions, including when the knowledge base had already returned
the person's own labs. The deliverable is now `health_insights_v1`, and a
question list fails validation or is repaired once.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError

from src.orchestrator.models import AgentConfig, BedrockRef, TeamConfig, TeamGlobals
from src.orchestrator.prompt_builder import build_prompt
from src.orchestrator.schema_validate import settle_health_insights, validate_output

REPO = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((REPO / "config/examples/schemas/health_insights_v1.json").read_text())
TEAM = json.loads(
    (REPO / "config/examples/teams/health_prep/v1/team.json").read_text())

VALID = {
    "summary": "The March note lists a higher home blood pressure than the January note.",
    "seek_care_now": False,
    "safety_note": "This is not a diagnosis or a substitute for care.",
    "insights": [{
        "title": "Home blood pressure is higher than the January note",
        "finding": "The March upload lists 142/90, up from 128/82 in January.",
        "evidence": [{
            "source": "s3://contextweave-health/bp.txt",
            "date": "2026-03-02",
            "value": "142/90",
        }],
        "significance": "moderate",
        "confidence": "medium",
    }],
    "suggestions": [{
        "action": "Record blood pressure morning and evening for seven days and bring the log.",
        "rationale": "Two readings months apart are not yet a trend.",
        "linked_insight": "Home blood pressure is higher than the January note",
        "priority": "high",
        "timeframe": "this week",
    }],
    "data_gaps": ["No sleep or activity excerpts were returned."],
    "follow_ups": [],
}


def test_the_schema_requires_insights_and_suggestions():
    assert SCHEMA["title"] == "health_insights_v1"
    for field in ("summary", "insights", "suggestions", "data_gaps",
                  "seek_care_now", "safety_note"):
        assert field in SCHEMA["required"], field
    assert SCHEMA["properties"]["insights"]["minItems"] == 1
    assert SCHEMA["properties"]["suggestions"]["minItems"] == 1
    assert SCHEMA["properties"]["follow_ups"]["maxItems"] == 2
    assert "questions_to_ask" not in SCHEMA["properties"]
    evidence = SCHEMA["properties"]["insights"]["items"]["properties"]["evidence"]
    assert evidence["minItems"] == 1
    assert set(evidence["items"]["required"]) == {"source", "date", "value"}
    validate_output(VALID, SCHEMA)


def test_a_question_list_fails_validation():
    """The old deliverable. It has no insights and no suggestions, so it is
    not a successful health run."""
    old = {
        "questions_to_ask": [
            "Should I be worried about this?",
            "What tests should I ask for?",
            "Is this serious?",
        ],
        "what_to_track_before": ["How often it happens?"],
        "bring_with_you": ["Previous results"],
        "this_is_not_advice": "This is not a diagnosis.",
    }
    with pytest.raises(ValidationError):
        validate_output(old, SCHEMA)


def test_an_answer_whose_substance_is_questions_fails_validation():
    shaped = json.loads(json.dumps(VALID))
    shaped["summary"] = "What should I ask about the March reading?"
    shaped["insights"][0]["title"] = "Is the March reading high?"
    shaped["insights"][0]["finding"] = "Does 142/90 need a visit?"
    shaped["suggestions"][0]["action"] = "Should I check this every day?"
    shaped["suggestions"][0]["rationale"] = "Would that be useful?"
    with pytest.raises(ValidationError, match="mostly questions"):
        validate_output(shaped, SCHEMA)


def test_a_question_in_data_gaps_fails_validation():
    shaped = json.loads(json.dumps(VALID))
    shaped["data_gaps"] = ["What is my usual sleep?"]
    with pytest.raises(ValidationError, match="mostly questions"):
        validate_output(shaped, SCHEMA)


def test_one_optional_follow_up_does_not_fail_a_grounded_answer():
    shaped = json.loads(json.dumps(VALID))
    shaped["follow_ups"] = ["Bring the home cuff to the next visit if you want the readings checked."]
    validate_output(shaped, SCHEMA)
    shaped["follow_ups"] = ["a", "b", "c"]
    with pytest.raises(ValidationError):
        validate_output(shaped, SCHEMA)


def test_a_question_list_is_reprompted_once():
    seen = []

    def transform(payload, schema, **_kwargs):
        seen.append(payload)
        assert schema["title"] == "health_insights_v1"
        return json.loads(json.dumps(VALID))

    out, accepted = settle_health_insights(
        {"questions_to_ask": ["Should I worry?"]}, SCHEMA, transform)
    assert accepted is True
    assert out["insights"][0]["finding"].startswith("The March upload")
    assert len(seen) == 1
    assert "Do not ask" in seen[0]["rejected_because"]


def test_a_valid_answer_is_not_sent_through_the_repair_model():
    def transform(payload, schema, **_kwargs):
        raise AssertionError("a valid insights answer was rewritten")

    out, accepted = settle_health_insights(VALID, SCHEMA, transform)
    assert accepted is True
    assert out == VALID


def test_a_repair_that_is_still_questions_is_not_accepted():
    shaped = json.loads(json.dumps(VALID))
    shaped["insights"][0]["finding"] = "Is this something to worry about?"
    shaped["insights"][0]["title"] = "Should this be checked?"
    shaped["suggestions"][0]["action"] = "What should I do next?"

    def transform(payload, schema, **_kwargs):
        return shaped

    out, accepted = settle_health_insights(
        {"questions_to_ask": ["Should I worry?"]}, SCHEMA, transform)
    assert accepted is False
    assert out == shaped


def test_the_worker_settles_health_insights_instead_of_always_rewriting():
    worker = (REPO / "src/orchestrator/worker_handler.py").read_text()
    assert "settle_health_insights" in worker
    assert "INSIGHT_TITLES" in worker
    assert "health_insights_v1" in (REPO / "src/orchestrator/schema_validate.py").read_text()


def test_prompts_include_the_kb_grounding_instructions():
    globals_ = TEAM["globals"]
    team = TeamConfig(
        team=TEAM["team"],
        globals=TeamGlobals(
            north_star=globals_["north_star"],
            default_channel=globals_["default_channel"],
            hard_constraints=globals_["hard_constraints"],
            features=globals_["features"],
            rag=globals_["rag"],
            artifact_store=globals_["artifact_store"],
            revision=globals_["revision"],
        ),
        agents=[], workflow=[], schemas={},
    )
    for agent_doc in TEAM["agents"]:
        agent = AgentConfig(
            id=agent_doc["id"],
            name=agent_doc["name"],
            bedrock=BedrockRef(agentId="", aliasId=""),
            goal_template=agent_doc["goal_template"],
            schema_ref=agent_doc["schema_ref"],
        )
        prompt = build_prompt(
            team, agent,
            {"request": {"experiencing": "headaches since March"}},
            {}, "", "", "",
        )
        assert "knowledge base" in prompt
        assert "sourceKey" in prompt
        assert "found false" in prompt
        assert "error" in prompt
        assert "DO THE WORK" in prompt
        assert "not the answer" in prompt
        lowered = agent_doc["goal_template"].lower()
        assert "questions is not" in lowered
    insights = next(a for a in TEAM["agents"] if a["id"] == "HP_insights")
    assert "suggestions" in insights["goal_template"]
    assert "what was searched" in insights["goal_template"]
    joined = " ".join(globals_["hard_constraints"])
    assert "list of questions is not the answer" in joined
    assert "trends" in joined and "out-of-range" in joined


def test_facets_retrieve_trends_labs_meds_sleep_and_the_report(monkeypatch):
    from src.orchestrator import health_kb
    from src.orchestrator.tools import weave_tools

    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    seen = []

    def retrieve(question, *, top_k=6, client=None):
        seen.append((question, top_k))
        if question.startswith("Recent laboratory"):
            return {
                "found": True,
                "source": "bedrock-knowledge-base",
                "excerpts": [{
                    "sourceKey": "s3://contextweave-health/labs.txt",
                    "content": "HbA1c 6.1 on 2026-03-02",
                }],
            }
        if question.startswith("Trends"):
            return {
                "found": True,
                "source": "bedrock-knowledge-base",
                "excerpts": [{
                    "sourceKey": "s3://contextweave-health/labs.txt",
                    "content": "HbA1c 6.1 on 2026-03-02",
                }],
            }
        return {"found": False, "excerpts": [], "source": "bedrock-knowledge-base"}

    monkeypatch.setattr(health_kb, "retrieve", retrieve)
    result = weave_tools.query_health_record(
        "headaches since March", caller_token="tok", top_k=4, facets=True)
    assert [q["facet"] for q in result["queries"]] == [
        "reported", "labs_vitals", "medications", "sleep_activity", "trends"]
    assert len(seen) == 5
    assert all(top_k == 4 for _q, top_k in seen)
    assert seen[0][0] == "headaches since March"
    assert result["found"] is True
    assert result["source"] == "bedrock-knowledge-base"
    assert result["excerpts"] == [{
        "sourceKey": "s3://contextweave-health/labs.txt",
        "content": "HbA1c 6.1 on 2026-03-02",
    }]
    labs = next(q for q in result["queries"] if q["facet"] == "labs_vitals")
    assert labs["found"] is True
    sleep = next(q for q in result["queries"] if q["facet"] == "sleep_activity")
    assert sleep["found"] is False
    assert len(result["searched"]) == 5


def test_empty_facets_say_what_was_searched_and_do_not_look_like_an_error(monkeypatch):
    from src.orchestrator import health_kb
    from src.orchestrator.tools import weave_tools

    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    monkeypatch.setattr(
        health_kb, "retrieve",
        lambda question, *, top_k=6, client=None: {
            "found": False, "excerpts": [], "source": "bedrock-knowledge-base"},
    )
    result = weave_tools.query_health_record(
        "headaches since March", caller_token="tok", facets=True)
    assert result["found"] is False
    assert result["excerpts"] == []
    assert "error" not in result
    assert len(result["searched"]) == 5
    assert any("Sleep and activity" in q for q in result["searched"])


def test_facets_without_a_token_do_not_retrieve(monkeypatch):
    from src.orchestrator import health_kb
    from src.orchestrator.tools import weave_tools

    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    monkeypatch.setattr(
        health_kb, "retrieve",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrieved without a token")),
    )
    result = weave_tools.query_health_record(
        "headaches since March", caller_token="", facets=True)
    assert "error" in result
    assert "found" not in result
    assert "queries" not in result


def test_a_facet_failure_on_every_query_is_not_an_empty_record(monkeypatch):
    from src.orchestrator import health_kb
    from src.orchestrator.tools import weave_tools

    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    monkeypatch.setattr(
        health_kb, "retrieve",
        lambda *a, **k: {"error": "the health knowledge base could not be reached"},
    )
    result = weave_tools.query_health_record("headaches", caller_token="tok", facets=True)
    assert "error" in result
    assert "found" not in result
    assert len(result["searched"]) == 5
    assert all("error" in q for q in result["queries"])
