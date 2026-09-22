import json
from typing import Any, Dict, List
from .models import TeamConfig, AgentConfig


def build_prompt(
    team: TeamConfig,
    agent: AgentConfig,
    step_inputs: Dict[str, Any],
    director_brief: Dict[str, Any],
    rag_context: str,
    owner_profile_context: str,
    gemini_brief: str,
) -> str:
    parts: List[str] = []

    # ── Identity ───────────────────────────────────────────────────────────────
    parts.append(f"ROLE: {agent.name}")
    parts.append(f"TEAM_NORTH_STAR: {team.globals.north_star}")
    parts.append("")

    # ── GOAL FIRST — anchor the agent before any context ──────────────────────
    # Placing STEP_GOAL at the top prevents the agent from treating upstream
    # instructions (e.g. "The task is to generate content...") as its response.
    parts.append("STEP_GOAL:")
    parts.append(agent.goal_template)
    parts.append("")
    parts.append("OUTPUT CONTRACT:")
    parts.append("Return ONLY valid JSON. No markdown. Must match the schema for this step.")
    parts.append("Do not ask follow-up questions. If inputs are incomplete, make reasonable assumptions and continue.")
    parts.append("You are generating content — not describing what you would do. Produce the actual output.")
    parts.append("")

    # ── Request / topic — explicit, not buried in INPUTS_JSON ─────────────────
    request_obj = step_inputs.get("request") or {}
    if request_obj:
        parts.append("REQUEST:")
        parts.append(json.dumps(request_obj, ensure_ascii=False))
        parts.append("")

    # ── Director brief ─────────────────────────────────────────────────────────
    if director_brief:
        parts.append("DIRECTOR_BRIEF_JSON:")
        parts.append(json.dumps(director_brief, ensure_ascii=False))
        parts.append("")
        acc = (
            director_brief.get("acceptance_criteria")
            or director_brief.get("acceptanceCriteria")
            or []
        )
        if acc:
            parts.append("ACCEPTANCE_CRITERIA:")
            for a in acc:
                parts.append(f"- {a}")
            parts.append("")

    # ── Hard constraints ───────────────────────────────────────────────────────
    if team.globals.hard_constraints:
        parts.append("HARD_CONSTRAINTS:")
        for c in team.globals.hard_constraints:
            parts.append(f"- {c}")
        parts.append("")

    # ── Research & profile context ─────────────────────────────────────────────
    if gemini_brief:
        parts.append("GEMINI_RESEARCH_BRIEF:")
        parts.append(gemini_brief)
        parts.append("")

    if owner_profile_context:
        parts.append("OWNER_PROFILE_CONTEXT:")
        parts.append(owner_profile_context)
        parts.append("")

    # ── Grounding — and what to do when there is none ──────────────────────────
    #
    # The block used to be dumped under a bare "RAG_CONTEXT:" label with no
    # instruction at all, which leaves the agent to infer what it is for. Two
    # failure modes follow from that, in opposite directions:
    #
    #   * with retrieved experience, the piece drifts into a career recital --
    #     the topic becomes a frame for the author rather than the reverse;
    #   * with none, nothing says "do not invent any", so the agent supplies
    #     plausible projects and outcomes that never happened. Nothing checks
    #     that, because a fabricated anecdote is exactly as schema-valid as a
    #     real one.
    #
    # The retrieval layer already decides *relevance*: contextweave mode drops
    # an answer below `min_confidence` and returns no context at all. So an
    # empty block here is a real signal -- "nothing relevant was found" -- and
    # is worth saying out loud rather than leaving as an absence.
    if rag_context:
        parts.append("VERIFIED_EXPERIENCE (retrieved from the author's own corpus):")
        parts.append(rag_context)
        parts.append("")
        parts.append("HOW TO USE IT:")
        parts.append("- Draw on it only where it genuinely supports the point being made.")
        parts.append("- It is supporting evidence, not the subject. The piece is about the "
                     "topic; a reader who has never heard of the author must still come away "
                     "with something useful.")
        parts.append("- Do not stretch it to cover claims it does not support, and do not add "
                     "experience that is not in it.")
        parts.append("")
    else:
        parts.append("NO_VERIFIED_EXPERIENCE:")
        parts.append("The knowledge layer returned nothing relevant enough for this topic.")
        parts.append("- Write from general technical expertise instead. That is a complete, "
                     "acceptable answer here — not a gap to paper over.")
        parts.append("- Do NOT invent, imply or imagine personal experience: no projects, "
                     "employers, incidents, metrics or outcomes attributed to the author.")
        parts.append("- First person about analysis and opinion is fine. First person about "
                     "things done is not.")
        parts.append("")

    # ── Prior step outputs — strip fields already shown above ─────────────────
    _SKIP_KEYS = {"request", "owner_profile_context", "rag_context", "rag_meta", "gemini_brief", "owner"}
    inputs_clean = {k: v for k, v in step_inputs.items() if k not in _SKIP_KEYS}
    if inputs_clean:
        parts.append("STEP_INPUTS_JSON:")
        parts.append(json.dumps(inputs_clean, ensure_ascii=False))
        parts.append("")

    return "\n".join(parts)
