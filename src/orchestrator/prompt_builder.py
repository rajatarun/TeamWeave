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

    if rag_context:
        parts.append("RAG_CONTEXT:")
        parts.append(rag_context)
        parts.append("")

    # ── Prior step outputs — strip fields already shown above ─────────────────
    _SKIP_KEYS = {"request", "owner_profile_context", "rag_context", "gemini_brief", "owner"}
    inputs_clean = {k: v for k, v in step_inputs.items() if k not in _SKIP_KEYS}
    if inputs_clean:
        parts.append("STEP_INPUTS_JSON:")
        parts.append(json.dumps(inputs_clean, ensure_ascii=False))
        parts.append("")

    return "\n".join(parts)
