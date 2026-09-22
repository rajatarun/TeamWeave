"""What the prompt tells an agent to do with grounding -- and without it.

The block used to be dumped under a bare `RAG_CONTEXT:` label with no
instruction at all, leaving the agent to infer its purpose. Two failures follow
from that, in opposite directions:

  * with retrieved experience, the piece drifts into a career recital -- the
    topic becomes a frame for the author rather than the reverse;
  * with none, nothing says "do not invent any", so the agent supplies
    plausible projects and outcomes that never happened. Nothing catches that:
    a fabricated anecdote is exactly as schema-valid as a real one.

The retrieval layer decides *relevance* (contextweave mode drops an answer
below `min_confidence`), so an empty block is a real signal -- "nothing
relevant was found" -- and is worth stating rather than leaving as an absence.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from src.orchestrator.models import AgentConfig, BedrockRef, TeamConfig, TeamGlobals  # noqa: E402
from src.orchestrator.prompt_builder import build_prompt  # noqa: E402


def _team(hard_constraints=()):
    return TeamConfig(
        team={},
        globals=TeamGlobals(
            north_star="ship", default_channel="linkedin",
            hard_constraints=list(hard_constraints), features={},
            rag={"mode": "contextweave"}, artifact_store={}, revision={},
        ),
        agents=[], workflow=[], schemas={},
    )


def _agent():
    return AgentConfig(id="writer", name="Writer",
                       bedrock=BedrockRef(agentId="a", aliasId="b"),
                       goal_template="Draft the post.", schema_ref="draft_pack_v1")


def prompt_with(rag_context: str, **kw) -> str:
    return build_prompt(_team(kw.pop("constraints", ())), _agent(),
                        {"request": {"topic": "t"}}, {}, rag_context, "", "")


# ── grounding found ─────────────────────────────────────────────────────────

def test_retrieved_experience_is_labelled_as_the_authors_own():
    prompt = prompt_with("[RAG #1] SOURCE: contextweave:answer\nBuilt a router.")
    assert "VERIFIED_EXPERIENCE" in prompt
    assert "Built a router." in prompt


def test_experience_is_supporting_evidence_not_the_subject():
    """The whole point of the request: the article must not be a career recital."""
    prompt = prompt_with("[RAG #1] SOURCE: x\nBuilt a router.")
    assert "supporting evidence, not the subject" in prompt
    assert "never heard of the author" in prompt


def test_the_agent_is_told_not_to_exceed_what_was_retrieved():
    prompt = prompt_with("[RAG #1] SOURCE: x\nBuilt a router.")
    assert "do not add experience that is not in it" in prompt


def test_the_no_experience_instruction_is_absent_when_there_is_some():
    """Both branches at once would tell the agent to use and ignore the same
    block, which is worse than either."""
    prompt = prompt_with("[RAG #1] SOURCE: x\nBuilt a router.")
    assert "NO_VERIFIED_EXPERIENCE" not in prompt


# ── nothing relevant found ──────────────────────────────────────────────────

def test_an_empty_result_is_stated_rather_than_left_as_an_absence():
    """`min_confidence` already dropped anything not relevant enough, so empty
    means "nothing relevant", not "retrieval was skipped"."""
    prompt = prompt_with("")
    assert "NO_VERIFIED_EXPERIENCE" in prompt


def test_general_knowledge_is_named_as_an_acceptable_answer():
    """Otherwise the agent treats the gap as something to paper over."""
    prompt = prompt_with("")
    assert "general technical expertise" in prompt
    assert "not a gap to paper over" in prompt


def test_fabricating_experience_is_forbidden_explicitly():
    """The failure this exists to prevent. A fabricated anecdote passes schema
    validation exactly as a real one does, so nothing downstream catches it."""
    prompt = prompt_with("")
    assert "Do NOT invent, imply or imagine personal experience" in prompt
    for forbidden in ("projects", "employers", "incidents", "metrics", "outcomes"):
        assert forbidden in prompt


def test_opinion_is_still_allowed_in_first_person():
    """Banning first person outright would produce stilted, voiceless copy."""
    prompt = prompt_with("")
    assert "First person about analysis and opinion is fine" in prompt


def test_the_use_it_instruction_is_absent_when_there_is_nothing_to_use():
    prompt = prompt_with("")
    # The positive label in full: a bare "VERIFIED_EXPERIENCE" check passes
    # trivially here, because NO_VERIFIED_EXPERIENCE contains it.
    assert "VERIFIED_EXPERIENCE (retrieved" not in prompt
    assert "HOW TO USE IT" not in prompt


# ── the two branches are genuinely different ────────────────────────────────

@pytest.mark.parametrize("context", ["", "[RAG #1] SOURCE: x\nsomething"])
def test_every_prompt_says_something_about_grounding(context):
    """Neither branch may be silent: silence is what produced both failures."""
    prompt = prompt_with(context)
    assert ("VERIFIED_EXPERIENCE" in prompt) or ("NO_VERIFIED_EXPERIENCE" in prompt)


def test_the_branches_do_not_produce_the_same_prompt():
    assert prompt_with("") != prompt_with("[RAG #1] SOURCE: x\nsomething")


# ── the shipped team is actually configured to do this ──────────────────────

def _visibility_globals():
    import json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    doc = json.loads((repo / "config" / "examples" / "teams"
                      / "tarun_visibility_team" / "v1" / "team.json").read_text())
    return doc["globals"]


def test_rag_is_switched_on_for_the_team():
    """`explicit_rag` sits above the mode dispatch and gates *every* mode
    despite its name. False left the declared mode as dead config and the team
    ran with no grounding at all."""
    assert _visibility_globals()["features"]["explicit_rag"] is True


def test_the_team_asks_the_knowledge_layer():
    rag = _visibility_globals()["rag"]
    assert rag["mode"] == "contextweave"


def test_relevance_is_a_retrieval_decision_not_a_request_to_the_model():
    """min_confidence is what makes "only if relevant experience is found"
    real: below it, _contextweave_context returns no context and the prompt
    takes its no-experience branch. Without a floor every answer is injected
    however weak, and the instruction becomes advice the model may ignore."""
    rag = _visibility_globals()["rag"]
    assert rag.get("min_confidence"), "no relevance floor: weak answers would still be injected"
    assert 0 < float(rag["min_confidence"]) < 1


def test_grounding_is_more_than_one_chunk():
    assert int(_visibility_globals()["rag"]["top_k"]) > 1


def test_no_constraint_demands_personal_examples_unconditionally():
    """The old constraint read "Prefer concrete examples from Tarun's work
    using RAG context" with no condition, so a run that retrieved nothing was
    still told to prefer personal examples -- and supplied them."""
    constraints = " ".join(_visibility_globals()["hard_constraints"]).lower()
    assert "prefer concrete examples from tarun's work" not in constraints


def test_a_constraint_makes_the_experience_conditional():
    constraints = " ".join(_visibility_globals()["hard_constraints"]).lower()
    assert "only where verified_experience supplies it" in constraints


def test_a_constraint_says_the_piece_is_not_about_tarun():
    constraints = " ".join(_visibility_globals()["hard_constraints"]).lower()
    assert "about the topic, not about tarun" in constraints
