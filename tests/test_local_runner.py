"""Unit tests for optional amp_metrics imports and LocalTeamRunner + SQLiteTeamRunStore."""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from src.orchestrator.local_runner import (
    LocalTeamRunner,
    SQLiteTeamRunStore,
    StepSchemaError,
    WorkflowValidationError,
    validate_and_order_workflow,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sample_team_config() -> dict[str, Any]:
    return {
        "team": {"team_id": "LOCAL_DEMO", "name": "local_demo_team", "version": "v1"},
        "agents": [
            {
                "id": "researcher",
                "name": "Researcher",
                "goal_template": "Gather facts for the request.",
                "schema_ref": "facts_v1",
            },
            {
                "id": "writer",
                "name": "Writer",
                "goal_template": "Draft a summary using facts.",
                "schema_ref": "draft_v1",
            },
            {
                "id": "reviewer",
                "name": "Reviewer",
                "goal_template": "Approve or reject the draft.",
                "schema_ref": "review_v1",
            },
        ],
        # Intentionally list 'writer' after 'reviewer' to verify DAG topological ordering.
        "workflow": [
            {"step": "researcher", "inputs": ["request"]},
            {"step": "reviewer", "inputs": ["writer.output", "researcher.output"]},
            {"step": "writer", "inputs": ["researcher.output"]},
        ],
        "schemas": {
            "facts_v1": {
                "type": "object",
                "required": ["topic", "bullets"],
                "properties": {
                    "topic": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "additionalProperties": False,
            },
            "draft_v1": {
                "type": "object",
                "required": ["headline", "body"],
                "properties": {
                    "headline": {"type": "string"},
                    "body": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "review_v1": {
                "type": "object",
                "required": ["approved", "score"],
                "properties": {
                    "approved": {"type": "boolean"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "additionalProperties": False,
            },
        },
    }


def test_amp_metrics_imports_cleanly_without_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """amp_metrics.py must import cleanly and no-op push_to_amp when requests is absent."""
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "requests" or name.startswith("requests."):
            raise ImportError("Simulated missing requests package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.delitem(sys.modules, "requests", raising=False)

    spec = importlib.util.spec_from_file_location(
        "src.orchestrator._amp_metrics_no_requests",
        REPO_ROOT / "src" / "orchestrator" / "amp_metrics.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "src.orchestrator"
    spec.loader.exec_module(mod)

    assert mod.requests is None
    monkeypatch.setenv("AMP_WORKSPACE_ID", "ws-offline-test")
    # Must gracefully no-op rather than raising AttributeError/ImportError.
    mod.push_to_amp([([("__name__", "test_metric")], 1.0, 1700000000000)])


def test_local_team_runner_end_to_end_dag_and_sqlite_wal(tmp_path: Path) -> None:
    db_file = tmp_path / "team_runs.db"
    executed_order: list[str] = []
    seen_inputs: dict[str, dict[str, Any]] = {}

    def _executor(
        agent: dict[str, Any],
        inputs: dict[str, Any],
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        aid = agent["id"]
        executed_order.append(aid)
        seen_inputs[aid] = inputs
        if aid == "researcher":
            return {"topic": inputs["request"]["topic"], "bullets": ["SQLite WAL is local-first"]}
        if aid == "writer":
            bullets = inputs["researcher.output"]["bullets"]
            return {"headline": "Local-First Teams", "body": f"Key point: {bullets[0]}"}
        if aid == "reviewer":
            assert "Local-First Teams" in inputs["writer.output"]["headline"]
            return {"approved": True, "score": 5}
        raise AssertionError(f"Unexpected agent: {aid}")

    runner = LocalTeamRunner(_sample_team_config(), db_path=db_file, agent_executor=_executor)
    try:
        assert runner.store.journal_mode() == "wal"
        result = runner.run({"topic": "WeaveOS"}, run_id="run-e2e-001")
    finally:
        runner.close()

    assert result.status == "COMPLETED"
    assert result.failed_step is None
    assert result.error is None
    assert executed_order == ["researcher", "writer", "reviewer"]
    assert result.step_outputs["reviewer"] == {"approved": True, "score": 5}

    # Re-open SQLite store to verify durable persistence of run and step artifacts.
    reopened = SQLiteTeamRunStore(db_file)
    try:
        saved = reopened.get_run("run-e2e-001")
        assert saved is not None
        assert saved["status"] == "COMPLETED"
        assert saved["team_id"] == "LOCAL_DEMO"
        assert [s["step_id"] for s in saved["steps"]] == ["researcher", "writer", "reviewer"]
        assert all(s["status"] == "COMPLETED" for s in saved["steps"])
        assert saved["steps"][1]["output"]["headline"] == "Local-First Teams"
        assert len(reopened.list_runs()) == 1
    finally:
        reopened.close()


def test_local_team_runner_halts_cleanly_on_schema_violation(tmp_path: Path) -> None:
    db_file = tmp_path / "schema_fail.db"
    executed: list[str] = []

    def _bad_writer_executor(
        agent: dict[str, Any],
        inputs: dict[str, Any],
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        aid = agent["id"]
        executed.append(aid)
        if aid == "researcher":
            return {"topic": "Edge AI", "bullets": ["Runs offline"]}
        if aid == "writer":
            # Missing required 'body' field -> must fail draft_v1 schema validation.
            return {"headline": "Incomplete draft"}
        return {"approved": True, "score": 5}

    runner = LocalTeamRunner(_sample_team_config(), db_path=db_file, agent_executor=_bad_writer_executor)
    try:
        result = runner.run({"topic": "Edge AI"}, run_id="run-bad-schema")
        saved = runner.store.get_run("run-bad-schema")
    finally:
        runner.close()

    assert result.status == "FAILED"
    assert result.failed_step == "writer"
    assert result.error is not None and "draft_v1" in result.error
    # Reviewer must never run after writer fails schema validation.
    assert executed == ["researcher", "writer"]

    assert saved is not None
    assert saved["status"] == "FAILED"
    assert [s["step_id"] for s in saved["steps"]] == ["researcher", "writer"]
    assert saved["steps"][0]["status"] == "COMPLETED"
    assert saved["steps"][1]["status"] == "FAILED"
    assert saved["steps"][1]["output"] is None
    assert "draft_v1" in (saved["steps"][1]["error"] or "")


def test_local_team_runner_raise_on_error_mode() -> None:
    def _bad_executor(agent: dict[str, Any], inputs: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
        return {"wrong_key": 123}

    runner = LocalTeamRunner(_sample_team_config(), agent_executor=_bad_executor)
    try:
        with pytest.raises(StepSchemaError):
            runner.run({"topic": "Strict"}, raise_on_error=True)
    finally:
        runner.close()


def test_workflow_validation_rejects_cycles_and_unknown_inputs() -> None:
    cyclic = _sample_team_config()
    cyclic["workflow"] = [
        {"step": "researcher", "inputs": ["writer.output"]},
        {"step": "writer", "inputs": ["researcher.output"]},
    ]
    with pytest.raises(WorkflowValidationError, match="cycle"):
        validate_and_order_workflow(cyclic)

    unknown = _sample_team_config()
    unknown["workflow"] = [
        {"step": "researcher", "inputs": ["ghost_step.output"]},
    ]
    with pytest.raises(WorkflowValidationError, match="unknown step"):
        validate_and_order_workflow(unknown)


def test_local_team_runner_loads_file_team_json_and_external_schema(tmp_path: Path) -> None:
    schema_path = tmp_path / "schemas" / "out_v1.json"
    schema_path.parent.mkdir(parents=True)
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
            }
        ),
        encoding="utf-8",
    )
    team_path = tmp_path / "team.json"
    team_path.write_text(
        json.dumps(
            {
                "team": {"team_id": "FILE_TEAM", "name": "file_team"},
                "agents": [{"id": "solo", "name": "Solo", "schema_ref": "out_v1"}],
                "workflow": [{"step": "solo", "inputs": ["request"]}],
                "schemas": {"out_v1": {"path": "schemas/out_v1.json"}},
            }
        ),
        encoding="utf-8",
    )

    runner = LocalTeamRunner(
        team_path,
        db_path=tmp_path / "file_team.db",
        agent_executor=lambda agent, inputs, schema: {"answer": f"ok:{inputs['request']}"},
    )
    try:
        res = runner.run("hello")
    finally:
        runner.close()

    assert res.status == "COMPLETED"
    assert res.step_outputs["solo"] == {"answer": "ok:hello"}
