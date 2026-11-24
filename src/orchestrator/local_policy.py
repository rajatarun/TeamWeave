"""Local execution policy and weave-telemetry-v1 formatting for TeamWeave.

Provides:
- :class:`LocalTeamPolicy`: step budget, role allowlist, and duration guardrails
  for local multi-agent team runs, with ``agent-policy-v1`` parsing.
- :func:`format_team_telemetry_event`: builds ``weave-telemetry-v1``-conformant
  telemetry payloads with ``source_repo="TeamWeave"`` and a guaranteed non-empty
  ``trace_id``.
- :func:`append_telemetry_event`: appends telemetry events to a JSONL file.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TELEMETRY_SCHEMA_ID = "weave-telemetry-v1"
TELEMETRY_SOURCE_REPO = "TeamWeave"
AGENT_POLICY_SCHEMA_ID = "agent-policy-v1"

_STATUS_MAP: dict[str, str] = {
    "ok": "ok",
    "completed": "ok",
    "succeeded": "ok",
    "success": "ok",
    "error": "error",
    "failed": "error",
    "fail": "error",
    "timeout": "timeout",
    "timed_out": "timeout",
    "aborted": "aborted",
    "policy_denied": "aborted",
    "denied": "aborted",
    "canceled": "aborted",
    "cancelled": "aborted",
    "pending": "pending",
    "running": "pending",
    "queued": "pending",
}


class PolicyDeniedError(PermissionError):
    """Raised when a LocalTeamRunner step violates LocalTeamPolicy with raise_on_error=True."""


def _utc_rfc3339_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def normalize_trace_id(trace_id: str | None) -> str:
    """Return a guaranteed non-empty trace_id string."""
    if isinstance(trace_id, str) and trace_id.strip():
        return trace_id.strip()
    return f"trace-{uuid.uuid4().hex}"


@dataclass
class LocalTeamPolicy:
    """Execution guardrails for local TeamWeave multi-agent workflow runs."""

    max_steps: int = 20
    allowed_roles: set[str] | None = None
    max_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool) or self.max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        if self.allowed_roles is not None:
            self.allowed_roles = {str(r).strip() for r in self.allowed_roles if str(r).strip()}
        if self.max_duration_seconds is not None:
            if isinstance(self.max_duration_seconds, bool) or self.max_duration_seconds < 0:
                raise ValueError("max_duration_seconds must be a non-negative number")
            self.max_duration_seconds = float(self.max_duration_seconds)

    @classmethod
    def from_agent_policy(cls, policy: dict[str, Any]) -> LocalTeamPolicy:
        """Construct a :class:`LocalTeamPolicy` from an ``agent-policy-v1`` dictionary.

        Supports both top-level keys (``max_steps``, ``allowed_roles``,
        ``max_duration_seconds``) and nested ``team_policy`` / ``team`` sections.
        When ``enforcement_mode`` is ``"disabled"``, returns a permissive default
        policy. When ``routing_mode`` is ``"halt"`` (and enforcement is active),
        sets ``max_steps=0`` so execution halts cleanly.
        """
        if not isinstance(policy, dict):
            raise TypeError("policy must be a dictionary")

        schema = policy.get("schema")
        if schema is not None and schema != AGENT_POLICY_SCHEMA_ID:
            raise ValueError(f"Unsupported policy schema {schema!r}; expected {AGENT_POLICY_SCHEMA_ID!r}")

        enforcement_mode = str(policy.get("enforcement_mode") or "enforce").strip().lower()
        if enforcement_mode == "disabled":
            return cls()

        team_section = policy.get("team_policy") or policy.get("team") or {}
        if not isinstance(team_section, dict):
            team_section = {}

        routing_mode = str(policy.get("routing_mode") or "normal").strip().lower()
        raw_max_steps = team_section.get("max_steps", policy.get("max_steps"))
        if routing_mode == "halt":
            max_steps = 0
        elif raw_max_steps is not None:
            max_steps = int(raw_max_steps)
        else:
            max_steps = 20

        raw_roles = team_section.get("allowed_roles", policy.get("allowed_roles"))
        allowed_roles: set[str] | None = None
        if raw_roles is not None:
            if isinstance(raw_roles, str):
                allowed_roles = {raw_roles.strip()} if raw_roles.strip() else set()
            else:
                allowed_roles = {str(r).strip() for r in raw_roles if str(r).strip()}

        raw_duration = team_section.get(
            "max_duration_seconds",
            policy.get("max_duration_seconds", policy.get("timeout_seconds")),
        )
        max_duration_seconds = float(raw_duration) if raw_duration is not None else None

        return cls(
            max_steps=max_steps,
            allowed_roles=allowed_roles,
            max_duration_seconds=max_duration_seconds,
        )

    def validate_step(
        self,
        step_index: int,
        role: str,
        elapsed_seconds: float,
    ) -> tuple[bool, str | None]:
        """Validate whether a step is permitted to execute under this policy.

        Returns ``(True, None)`` when allowed, or ``(False, reason)`` when denied.
        """
        if step_index < 0:
            return False, f"Invalid negative step_index: {step_index}"

        if step_index >= self.max_steps:
            return (
                False,
                f"Step index {step_index} exceeds max_steps budget ({self.max_steps})",
            )

        if self.allowed_roles is not None:
            normalized_role = str(role or "").strip()
            if normalized_role not in self.allowed_roles:
                return (
                    False,
                    f"Role {normalized_role!r} is not permitted by allowed_roles ({sorted(self.allowed_roles)})",
                )

        if self.max_duration_seconds is not None and elapsed_seconds > self.max_duration_seconds:
            return (
                False,
                f"Elapsed duration {elapsed_seconds:.3f}s exceeds max_duration_seconds ({self.max_duration_seconds}s)",
            )

        return True, None


def format_team_telemetry_event(
    *,
    team_name: str,
    run_id: str,
    step_name: str,
    status: str,
    latency_ms: float,
    trace_id: str | None = None,
    attributes: dict[str, Any] | None = None,
    error_message: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    event_type: str = "team_step",
) -> dict[str, Any]:
    """Format a TeamWeave step event strictly conforming to ``weave-telemetry-v1``."""
    raw_status = str(status or "").strip().lower()
    canonical_status = _STATUS_MAP.get(raw_status)
    if canonical_status is None:
        raise ValueError(
            f"Unsupported telemetry status {status!r}; expected one of {sorted(set(_STATUS_MAP.values()))}"
        )

    merged_attributes: dict[str, Any] = {
        "team_name": str(team_name),
        "run_id": str(run_id),
        "step_name": str(step_name),
    }
    if attributes:
        merged_attributes.update(attributes)

    event: dict[str, Any] = {
        "schema": TELEMETRY_SCHEMA_ID,
        "event_id": f"evt-{uuid.uuid4().hex}",
        "source_repo": TELEMETRY_SOURCE_REPO,
        "event_type": str(event_type or "team_step").strip() or "team_step",
        "status": canonical_status,
        "latency_ms": round(max(0.0, float(latency_ms)), 3),
        "tokens_in": max(0, int(tokens_in)),
        "tokens_out": max(0, int(tokens_out)),
        "cost_usd": max(0.0, float(cost_usd)),
        "trace_id": normalize_trace_id(trace_id),
        "timestamp": _utc_rfc3339_now(),
        "attributes": merged_attributes,
    }
    if error_message is not None:
        event["error_message"] = str(error_message)
    return event


def append_telemetry_event(telemetry_path: str | Path, event: dict[str, Any]) -> None:
    """Append a single ``weave-telemetry-v1`` event as a JSONL line."""
    path = Path(telemetry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
