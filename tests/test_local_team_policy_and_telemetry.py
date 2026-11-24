"""Unit tests for TeamWeave LocalTeamPolicy and weave-telemetry-v1 step emission."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from src.orchestrator.local_policy import (
    LocalTeamPolicy,
    PolicyDeniedError,
    format_team_telemetry_event,
)
from src.orchestrator.local_runner import LocalTeamRunner

_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
)


def _sample_team_config() -> dict[str, Any]:
    return {
        "team": {"team_id": "POLICY_DEMO", "name": "policy_demo_team", "version": "v1"},
        "agents": [
            {
                "id": "researcher",
                "name": "Researcher",
                "role": "research",
                "goal_template": "Gather facts.",
            },
            {
                "id": "writer",
                "name": "Writer",
                "role": "write",
                "goal_template": "Draft summary.",
            },
            {
                "id": "publisher",
                "name": "Publisher",
                "role": "publish",
                "goal_template": "Publish output.",
            },
        ],
        "workflow": [
            {"step": "researcher", "inputs": ["request"]},
            {"step": "writer", "inputs": ["researcher.output"]},
            {"step": "publisher", "inputs": ["writer.output"]},
        ],
        "schemas": {},
    }


def _assert_weave_telemetry_v1(event: dict[str, Any]) -> None:
    assert event["schema"] == "weave-telemetry-v1"
    assert isinstance(event["event_id"], str) and event["event_id"]
    assert event["source_repo"] == "TeamWeave"
    assert isinstance(event["event_type"], str) and event["event_type"]
    assert event["status"] in {"ok", "error", "timeout", "aborted", "pending"}
    assert isinstance(event["latency_ms"], (int, float)) and event["latency_ms"] >= 0
    assert isinstance(event["tokens_in"], int) and event["tokens_in"] >= 0
    assert isinstance(event["tokens_out"], int) and event["tokens_out"] >= 0
    assert isinstance(event["cost_usd"], (int, float)) and event["cost_usd"] >= 0
    assert isinstance(event["trace_id"], str) and len(event["trace_id"].strip()) > 0
    assert isinstance(event["timestamp"], str) and _RFC3339_RE.match(event["timestamp"])
    assert isinstance(event["attributes"], dict)


def test_local_team_policy_validate_step_budgets() -> None:
    policy = LocalTeamPolicy(
        max_steps=2,
        allowed_roles={"research", "write"},
        max_duration_seconds=5.0,
    )

    assert policy.validate_step(0, "research", 0.1) == (True, None)
    assert policy.validate_step(1, "write", 4.9) == (True, None)

    ok_steps, reason_steps = policy.validate_step(2, "research", 0.5)
    assert ok_steps is False
    assert reason_steps is not None and "max_steps" in reason_steps

    ok_role, reason_role = policy.validate_step(0, "publish", 0.5)
    assert ok_role is False
    assert reason_role is not None and "allowed_roles" in reason_role

    ok_dur, reason_dur = policy.validate_step(0, "research", 6.5)
    assert ok_dur is False
    assert reason_dur is not None and "max_duration_seconds" in reason_dur


def test_local_team_policy_from_agent_policy() -> None:
    policy = LocalTeamPolicy.from_agent_policy(
        {
            "schema": "agent-policy-v1",
            "policy_id": "pol-team-1",
            "enforcement_mode": "enforce",
            "team_policy": {
                "max_steps": 3,
                "allowed_roles": ["research", "write"],
                "max_duration_seconds": 30,
            },
        }
    )
    assert policy.max_steps == 3
    assert policy.allowed_roles == {"research", "write"}
    assert policy.max_duration_seconds == 30.0

    halted = LocalTeamPolicy.from_agent_policy(
        {
            "schema": "agent-policy-v1",
            "enforcement_mode": "enforce",
            "routing_mode": "halt",
            "max_steps": 10,
        }
    )
    assert halted.max_steps == 0
    assert halted.validate_step(0, "research", 0.0)[0] is False

    disabled = LocalTeamPolicy.from_agent_policy(
        {
            "schema": "agent-policy-v1",
            "enforcement_mode": "disabled",
            "routing_mode": "halt",
            "max_steps": 0,
        }
    )
    assert disabled.max_steps == 20

    with pytest.raises(ValueError, match="Unsupported policy schema"):
        LocalTeamPolicy.from_agent_policy({"schema": "other-v1"})


def test_format_team_telemetry_event_conforms_to_weave_telemetry_v1() -> None:
    ev_auto = format_team_telemetry_event(
        team_name="alpha",
        run_id="run-1",
        step_name="researcher",
        status="completed",
        latency_ms=42.5,
        trace_id="   ",
        attributes={"role": "research"},
    )
    _assert_weave_telemetry_v1(ev_auto)
    assert ev_auto["status"] == "ok"
    assert ev_auto["trace_id"].startswith("trace-")
    assert ev_auto["attributes"]["team_name"] == "alpha"
    assert ev_auto["attributes"]["role"] == "research"

    ev_denied = format_team_telemetry_event(
        team_name="alpha",
        run_id="run-1",
        step_name="publisher",
        status="policy_denied",
        latency_ms=1.2,
        trace_id="trace-fixed-99",
        error_message="Role not allowed",
    )
    _assert_weave_telemetry_v1(ev_denied)
    assert ev_denied["status"] == "aborted"
    assert ev_denied["trace_id"] == "trace-fixed-99"
    assert ev_denied["error_message"] == "Role not allowed"


def test_runner_halts_on_policy_violation_and_emits_telemetry(tmp_path: Path) -> None:
    telemetry_file = tmp_path / "team_telemetry.jsonl"
    executed: list[str] = []

    def _executor(
        agent: dict[str, Any],
        inputs: dict[str, Any],
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        executed.append(agent["id"])
        return {"step": agent["id"], "done": True}

    policy = LocalTeamPolicy(max_steps=2, allowed_roles={"research", "write"})
    runner = LocalTeamRunner(
        _sample_team_config(),
        db_path=tmp_path / "runs.db",
        agent_executor=_executor,
        policy=policy,
        telemetry_path=telemetry_file,
    )

    result = runner.run({"topic": "edge-ai"}, trace_id="   ")
    assert result.status == "policy_denied"
    assert result.failed_step == "publisher"
    assert result.error is not None and "max_steps" in result.error
    assert executed == ["researcher", "writer"]

    # Telemetry JSONL has 2 ok steps + 1 aborted (policy_denied) step sharing one trace_id
    lines = [
        json.loads(line)
        for line in telemetry_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 3
    for ev in lines:
        _assert_weave_telemetry_v1(ev)
    assert [ev["status"] for ev in lines] == ["ok", "ok", "aborted"]
    trace_ids = {ev["trace_id"] for ev in lines}
    assert len(trace_ids) == 1
    assert next(iter(trace_ids)).startswith("trace-")

    # raise_on_error=True raises PolicyDeniedError
    role_policy = LocalTeamPolicy(max_steps=10, allowed_roles={"research"})
    strict_runner = LocalTeamRunner(
        _sample_team_config(),
        db_path=tmp_path / "runs_strict.db",
        agent_executor=_executor,
        policy=role_policy,
    )
    with pytest.raises(PolicyDeniedError, match="allowed_roles"):
        strict_runner.run({"topic": "edge-ai"}, raise_on_error=True)
