import json
from typing import Any, Dict, List
from .models import TeamConfig, AgentConfig

def build_prompt(team: TeamConfig, agent: AgentConfig, step_inputs: Dict[str, Any],
                 director_brief: Dict[str, Any], rag_context: str,
                 owner_profile_context: str, gemini_brief: str) -> str:
    parts: List[str] = []
    parts.append(f"ROLE: {agent.name}")
    parts.append(f"TEAM_NORTH_STAR: {team.globals.north_star}")
    parts.append("")
    if director_brief:
        parts.append("DIRECTOR_BRIEF_JSON:")
        parts.append(json.dumps(director_brief, ensure_ascii=False))
        parts.append("")
        acc = director_brief.get("acceptance_criteria") or director_brief.get("acceptanceCriteria") or []
        if acc:
            parts.append("ACCEPTANCE_CRITERIA:")
            for a in acc:
                parts.append(f"- {a}")
            parts.append("")
    if team.globals.hard_constraints:
        parts.append("HARD_CONSTRAINTS:")
        for c in team.globals.hard_constraints:
            parts.append(f"- {c}")
        parts.append("")
    parts.append("RAG_CONTEXT:")
    parts.append(rag_context or "")
    parts.append("")
    parts.append("OWNER_PROFILE_CONTEXT:")
    parts.append(owner_profile_context or "")
    parts.append("")
    parts.append("GEMINI_RESEARCH_BRIEF:")
    parts.append(gemini_brief or "")
    parts.append("")
    parts.append("INPUTS_JSON:")
    parts.append(json.dumps(step_inputs, ensure_ascii=False))
    parts.append("")
    parts.append("STEP_GOAL:")
    parts.append(agent.goal_template)
    parts.append("")
    parts.append("OUTPUT CONTRACT:")
    parts.append("Return ONLY valid JSON. No markdown. Must match the schema for this step.")
    return "\n".join(parts)
