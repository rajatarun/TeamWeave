"""TeamWeave as an A2A agent: the Agent Card and the Task mapping.

A2A (Agent2Agent) reached 1.0.0 in January 2026 under the Linux Foundation and
is how agents built on different stacks discover and call each other. That is
the weave platform's whole premise, and until now sibling services reached
TeamWeave through a hardcoded URL and private knowledge of its payload shape.

**One card, many skills.** TeamWeave is one A2A agent; each TeamWeave agent is
an `AgentSkill` on it. That is the third place this platform has had to answer
"what identifies an agent", and the third time the answer is *not* a separate
piece of infrastructure: Classic made one Bedrock agent each, AgentCore was
almost given one endpoint each, and the answer that holds is a name in a
document -- a skill id here, `gen_ai.agent.id` on the span. Adding an agent
costs nothing but a line of config.

Built against the published v1.0 specification rather than recollection, which
matters because v1.0 changed the card's shape: `supportedInterfaces` replaced
the top-level `url` and `preferredTransport` of 0.3.x, and each entry declares
its own `protocolBinding` and `protocolVersion`.

**What this deliberately does not claim.** The card advertises exactly one
interface, HTTP+JSON, because that is the one TeamWeave actually serves.
`message:send` is non-blocking: A2A's default is to block until the task is
terminal, and a TeamWeave pipeline outlives any API Gateway request, so the
card declares no streaming and the task comes back in a working state for the
caller to poll. Declaring a transport that is not there would be a lie a
machine acts on.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# The spec version this card is written against. `supportedInterfaces` carries
# it per interface; a client uses it to decide whether it can talk to us.
A2A_PROTOCOL_VERSION = "1.0"
HTTP_JSON_BINDING = "HTTP+JSON"

# Where every A2A client looks first.
AGENT_CARD_PATH = "/.well-known/agent-card.json"
# The A2A surface itself, kept off the existing routes so the two APIs cannot
# collide as either grows.
A2A_BASE_PATH = "/a2a"

# Named once so the scheme and the requirement that references it cannot
# drift apart -- a `security` entry naming a scheme the card does not define
# is a requirement no client can satisfy.
SECURITY_SCHEME_NAME = "siweBearer"

DEFAULT_INPUT_MODES = ["application/json", "text/plain"]
DEFAULT_OUTPUT_MODES = ["application/json", "text/plain"]

# TaskState, spelled exactly as the protobuf enum serialises to JSON.
TASK_SUBMITTED = "TASK_STATE_SUBMITTED"
TASK_WORKING = "TASK_STATE_WORKING"
TASK_COMPLETED = "TASK_STATE_COMPLETED"
TASK_FAILED = "TASK_STATE_FAILED"
TASK_CANCELED = "TASK_STATE_CANCELED"
TASK_UNKNOWN = "TASK_STATE_UNSPECIFIED"

# TeamWeave run status -> A2A task state. Step Functions has more statuses
# than the four the status handler surfaces, so the unmapped ones resolve to
# UNSPECIFIED rather than to a guess: reporting a timed-out run as completed
# would be worse than reporting it as unknown.
_RUN_STATUS_TO_TASK_STATE = {
    "SUBMITTED": TASK_SUBMITTED,
    "RUNNING": TASK_WORKING,
    "SUCCEEDED": TASK_COMPLETED,
    "FAILED": TASK_FAILED,
    "TIMED_OUT": TASK_FAILED,
    "ABORTED": TASK_CANCELED,
}

TERMINAL_STATES = {TASK_COMPLETED, TASK_FAILED, TASK_CANCELED}

_TAG_SAFE = re.compile(r"[^a-z0-9]+")


def task_state_for(run_status: str) -> str:
    """The A2A state for a TeamWeave run status."""
    return _RUN_STATUS_TO_TASK_STATE.get(str(run_status or "").strip().upper(), TASK_UNKNOWN)


def is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES


def _tag(value: str) -> str:
    return _TAG_SAFE.sub("-", str(value or "").strip().lower()).strip("-")


def skill_from_agent(agent: Dict[str, Any], team_name: str) -> Dict[str, Any]:
    """One TeamWeave agent as an A2A skill.

    `id`, `name`, `description` and `tags` are the required four. The skill id
    is prefixed with the team because the same agent id may appear in more
    than one team and a card's skill ids have to be unique within the card.
    """
    agent_id = str(agent.get("id") or agent.get("name") or "").strip()
    display = str(agent.get("name") or agent_id).strip()
    goal = str(agent.get("goal_template") or "").strip()
    role = str(agent.get("role_id") or "").strip()

    description = goal or f"TeamWeave agent {display} in team {team_name}."
    tags = [t for t in ("teamweave", _tag(team_name), _tag(role)) if t]

    skill: Dict[str, Any] = {
        "id": f"{team_name}.{agent_id}" if team_name else agent_id,
        "name": display,
        "description": description[:1000],
        "tags": tags,
        "inputModes": list(DEFAULT_INPUT_MODES),
        "outputModes": list(DEFAULT_OUTPUT_MODES),
    }
    schema_ref = str(agent.get("schema_ref") or "").strip()
    if schema_ref:
        # A caller needs to know the output is schema-constrained before it
        # sends anything, not after it fails validation.
        skill["description"] = f"{skill['description']} Output is validated against schema '{schema_ref}'."
    return skill


def skills_from_teams(teams: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every agent in every team, as skills, deduplicated and ordered.

    Sorted so the card is byte-stable across deploys: a card that reorders
    itself every deploy defeats the caching the spec asks clients to do.
    """
    skills: Dict[str, Dict[str, Any]] = {}
    for _, team in sorted(teams.items()):
        team_name = str((team.get("team") or {}).get("name") or "").strip()
        for agent in team.get("agents") or []:
            if not str(agent.get("id") or agent.get("name") or "").strip():
                continue
            skill = skill_from_agent(agent, team_name)
            skills.setdefault(skill["id"], skill)
    return [skills[k] for k in sorted(skills)]


def build_agent_card(
    *,
    base_url: str,
    version: str,
    teams: Dict[str, Dict[str, Any]],
    documentation_url: str = "",
    provider: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """The document served at /.well-known/agent-card.json."""
    base = str(base_url or "").rstrip("/")
    card: Dict[str, Any] = {
        "name": "TeamWeave",
        "description": (
            "Config-driven multi-agent orchestration. Each skill is a TeamWeave "
            "agent; sending a message starts a pipeline run and returns a task "
            "to poll."
        ),
        "version": version,
        # v1.0: per-interface url and binding. There is exactly one because
        # there is exactly one that works.
        "supportedInterfaces": [
            {
                "url": f"{base}{A2A_BASE_PATH}",
                "protocolBinding": HTTP_JSON_BINDING,
                "protocolVersion": A2A_PROTOCOL_VERSION,
            }
        ],
        "capabilities": {
            # Both false and both true statements. Claiming streaming without
            # /message:stream would strand every client that preferred it.
            "streaming": False,
            "pushNotifications": False,
        },
        "defaultInputModes": list(DEFAULT_INPUT_MODES),
        "defaultOutputModes": list(DEFAULT_OUTPUT_MODES),
        "skills": skills_from_teams(teams),
        # The card is public; the operations are not. A client that discovers
        # this document and calls message:send without a token gets a 401 from
        # the authorizer, and the card is the only place that can tell it why
        # -- so it says so, rather than leaving the client to guess from a
        # status code. The token is the SIWE-issued JWT the rest of the API
        # takes; the card names the scheme, never a credential.
        "securitySchemes": {
            SECURITY_SCHEME_NAME: {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": (
                    "SIWE-issued session JWT, sent as 'Authorization: Bearer <token>'. "
                    "Obtain one from the SIWE endpoints; see documentationUrl."
                ),
            }
        },
        "security": [{SECURITY_SCHEME_NAME: []}],
    }
    if provider:
        card["provider"] = dict(provider)
    if documentation_url:
        card["documentationUrl"] = documentation_url
    return card


def task_for_run(run_id: str, run_status: str, *, context_id: str = "",
                 result: Any = None, error: str = "") -> Dict[str, Any]:
    """A TeamWeave run as an A2A Task."""
    state = task_state_for(run_status)
    task: Dict[str, Any] = {
        "id": run_id,
        "contextId": context_id or run_id,
        "status": {"state": state},
    }
    if state == TASK_COMPLETED and result is not None:
        task["artifacts"] = [{
            "artifactId": f"{run_id}-result",
            "name": "run-result",
            "parts": [{"text": result if isinstance(result, str) else _json(result)}],
        }]
    if state == TASK_FAILED and error:
        task["status"]["message"] = {
            "role": "ROLE_AGENT",
            "messageId": f"{run_id}-error",
            "parts": [{"text": str(error)[:2000]}],
        }
    return task


def _json(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, default=str)


def text_of_message(message: Dict[str, Any]) -> str:
    """The prompt out of an A2A Message.

    Parts may carry text, files or structured data; only text is honoured
    here, and a caller sending nothing else usable gets an empty string rather
    than a crash.
    """
    parts = (message or {}).get("parts") or []
    chunks = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n".join(chunks)
