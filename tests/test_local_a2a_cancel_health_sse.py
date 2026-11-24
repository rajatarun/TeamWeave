"""Unit and HTTP integration tests for Cycle 3 LocalA2AService features:
- JSON-RPC ``tasks/cancel``
- ``get_health()`` & ``GET /health``
- ``GET /tasks/{run_id}/events`` Server-Sent Events (SSE) progress replay
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.orchestrator import a2a
from src.orchestrator.local_a2a_server import (
    ERR_INVALID_PARAMS,
    ERR_TASK_NOT_FOUND,
    LocalA2AService,
    create_http_server,
)
from src.orchestrator.local_runner import LocalTeamRunner, SQLiteTeamRunStore


def _team_config(name: str = "demo_team") -> dict[str, Any]:
    return {
        "team": {"team_id": f"{name.upper()}_ID", "name": name, "version": "v1"},
        "agents": [
            {
                "id": "planner",
                "name": "Planner",
                "role_id": "PLAN-1",
                "goal_template": "Create a concise plan from the request.",
                "schema_ref": "plan_v1",
            },
            {
                "id": "writer",
                "name": "Writer",
                "role_id": "WRITE-1",
                "goal_template": "Write final output from the plan.",
                "schema_ref": "final_v1",
            },
        ],
        "workflow": [
            {"step": "planner", "inputs": ["request"]},
            {"step": "writer", "inputs": ["planner.output"]},
        ],
        "schemas": {
            "plan_v1": {
                "type": "object",
                "required": ["steps"],
                "properties": {"steps": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
                "additionalProperties": False,
            },
            "final_v1": {
                "type": "object",
                "required": ["summary"],
                "properties": {"summary": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    }


def _demo_executor(
    agent: dict[str, Any],
    inputs: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    if agent["id"] == "planner":
        req = inputs["request"]
        topic = req["topic"] if isinstance(req, dict) else str(req)
        return {"steps": [f"research:{topic}", f"draft:{topic}"]}
    if agent["id"] == "writer":
        steps = inputs["planner.output"]["steps"]
        return {"summary": " -> ".join(steps)}
    raise AssertionError(f"Unexpected agent: {agent['id']}")


def _http_get_raw(url: str) -> tuple[int, dict[str, str], str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, headers, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        headers = {k.lower(): v for k, v in exc.headers.items()}
        return exc.code, headers, exc.read().decode("utf-8")


def _http_post_json(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _parse_sse(body: str) -> list[dict[str, Any]]:
    """Parse a text/event-stream payload into structured event dicts."""
    events: list[dict[str, Any]] = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        ev: dict[str, Any] = {}
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("id:"):
                ev["id"] = line.split(":", 1)[1].strip()
            elif line.startswith("event:"):
                ev["event"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if data_lines:
            ev["data"] = json.loads("\n".join(data_lines))
        events.append(ev)
    return events


def test_get_health_unit_and_http_reports_teams_and_run_counts(tmp_path: Path) -> None:
    store = SQLiteTeamRunStore(tmp_path / "health.db")
    alpha = _team_config("alpha_team")
    beta = _team_config("beta_team")
    runner = LocalTeamRunner(alpha, store=store, agent_executor=_demo_executor)
    service = LocalA2AService(
        {"beta_team": beta, "alpha_team": alpha},
        runner,
        store,
        agent_executor=_demo_executor,
    )
    server = create_http_server(service, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Initial health before any runs.
        unit_health = service.get_health()
        assert unit_health == {
            "status": "ok",
            "teams_loaded": 2,
            "team_names": ["alpha_team", "beta_team"],
            "total_runs": 0,
        }

        status, headers, raw = _http_get_raw(f"http://127.0.0.1:{port}/health")
        assert status == 200
        assert "application/json" in headers["content-type"]
        assert json.loads(raw) == unit_health

        # Execute two runs via tasks/send and confirm total_runs increments accurately.
        for idx, team_name in enumerate(("alpha_team", "beta_team"), start=1):
            post_status, resp = _http_post_json(
                f"http://127.0.0.1:{port}/a2a",
                {
                    "jsonrpc": "2.0",
                    "id": f"send-{idx}",
                    "method": "tasks/send",
                    "params": {
                        "id": f"run-health-{idx}",
                        "team": team_name,
                        "input": {"topic": f"topic-{idx}"},
                    },
                },
            )
            assert post_status == 200
            assert "error" not in resp

        status2, _, raw2 = _http_get_raw(f"http://127.0.0.1:{port}/health")
        assert status2 == 200
        health_after = json.loads(raw2)
        assert health_after["status"] == "ok"
        assert health_after["teams_loaded"] == 2
        assert health_after["team_names"] == ["alpha_team", "beta_team"]
        assert health_after["total_runs"] == 2
    finally:
        server.shutdown()
        server.server_close()
        store.close()


def test_tasks_cancel_transitions_run_to_canceled_and_reflected_in_tasks_get(tmp_path: Path) -> None:
    store = SQLiteTeamRunStore(tmp_path / "cancel.db")
    team_cfg = _team_config("ops_team")
    runner = LocalTeamRunner(team_cfg, store=store, agent_executor=_demo_executor)
    service = LocalA2AService({"ops_team": team_cfg}, runner, store)
    server = create_http_server(service, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Seed a RUNNING task with one completed step in SQLiteTeamRunStore.
        store.create_run("run-cancel-001", "OPS_TEAM_ID", "ops_team", {"topic": "Edge"})
        store.record_step(
            "run-cancel-001",
            "planner",
            0,
            "COMPLETED",
            {"request": {"topic": "Edge"}},
            output={"steps": ["research:Edge"]},
            schema_ref="plan_v1",
        )

        # Cancel via HTTP JSON-RPC POST /a2a.
        status, cancel_resp = _http_post_json(
            f"http://127.0.0.1:{port}/a2a",
            {
                "jsonrpc": "2.0",
                "id": "cancel-req-1",
                "method": "tasks/cancel",
                "params": {"run_id": "run-cancel-001", "reason": "operator requested stop"},
            },
        )
        assert status == 200
        assert cancel_resp["id"] == "cancel-req-1"
        assert "error" not in cancel_resp
        canceled_task = cancel_resp["result"]
        assert canceled_task["id"] == "run-cancel-001"
        assert canceled_task["status"]["state"] == a2a.TASK_CANCELED
        assert a2a.is_terminal(canceled_task["status"]["state"])

        # Verify persisted SQLite state is updated to "canceled".
        persisted = store.get_run("run-cancel-001")
        assert persisted is not None
        assert persisted["status"] == "canceled"
        assert persisted["error"] == "operator requested stop"

        # Verify tasks/get reflects the canceled state.
        get_status, get_resp = _http_post_json(
            f"http://127.0.0.1:{port}/",
            {
                "jsonrpc": "2.0",
                "id": "get-req-1",
                "method": "tasks/get",
                "params": {"id": "run-cancel-001"},
            },
        )
        assert get_status == 200
        assert "error" not in get_resp
        assert get_resp["result"]["status"]["state"] == a2a.TASK_CANCELED
    finally:
        server.shutdown()
        server.server_close()
        store.close()


def test_tasks_cancel_errors_for_unknown_and_missing_run_id() -> None:
    store = SQLiteTeamRunStore(":memory:")
    team_cfg = _team_config("demo_team")
    runner = LocalTeamRunner(team_cfg, store=store, agent_executor=_demo_executor)
    service = LocalA2AService({"demo_team": team_cfg}, runner, store)
    try:
        missing_param = service.handle_jsonrpc(
            {"jsonrpc": "2.0", "id": 10, "method": "tasks/cancel", "params": {}}
        )
        assert missing_param["error"]["code"] == ERR_INVALID_PARAMS

        blank_param = service.handle_jsonrpc(
            {"jsonrpc": "2.0", "id": 11, "method": "tasks/cancel", "params": {"run_id": "   "}}
        )
        assert blank_param["error"]["code"] == ERR_INVALID_PARAMS

        unknown = service.handle_jsonrpc(
            {"jsonrpc": "2.0", "id": 12, "method": "tasks/cancel", "params": {"run_id": "no-such-run"}}
        )
        assert unknown["error"]["code"] == ERR_TASK_NOT_FOUND
        assert "no-such-run" in unknown["error"]["message"]
    finally:
        store.close()


def test_sse_events_endpoint_streams_steps_and_terminal_event_over_http(tmp_path: Path) -> None:
    store = SQLiteTeamRunStore(tmp_path / "sse.db")
    team_cfg = _team_config("demo_team")
    runner = LocalTeamRunner(team_cfg, store=store, agent_executor=_demo_executor)
    service = LocalA2AService({"demo_team": team_cfg}, runner, store)
    server = create_http_server(service, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # 1. Run a full 2-step team via tasks/send.
        send_status, send_resp = _http_post_json(
            f"http://127.0.0.1:{port}/a2a",
            {
                "jsonrpc": "2.0",
                "id": "sse-send-1",
                "method": "tasks/send",
                "params": {"id": "run-sse-001", "team": "demo_team", "input": {"topic": "Telemetry"}},
            },
        )
        assert send_status == 200
        assert "error" not in send_resp

        # 2. GET /tasks/run-sse-001/events returns text/event-stream with ordered step & terminal events.
        status, headers, body = _http_get_raw(f"http://127.0.0.1:{port}/tasks/run-sse-001/events")
        assert status == 200
        assert headers["content-type"].startswith("text/event-stream")
        assert headers.get("cache-control") == "no-cache"

        events = _parse_sse(body)
        assert len(events) == 3
        assert [ev["event"] for ev in events] == ["task.step", "task.step", "task.terminal"]

        step0 = events[0]["data"]
        assert step0["run_id"] == "run-sse-001"
        assert step0["step_id"] == "planner"
        assert step0["seq"] == 0
        assert step0["status"] == "COMPLETED"
        assert step0["output"] == {"steps": ["research:Telemetry", "draft:Telemetry"]}

        step1 = events[1]["data"]
        assert step1["run_id"] == "run-sse-001"
        assert step1["step_id"] == "writer"
        assert step1["seq"] == 1
        assert step1["status"] == "COMPLETED"
        assert step1["output"] == {"summary": "research:Telemetry -> draft:Telemetry"}

        term = events[2]["data"]
        assert term["run_id"] == "run-sse-001"
        assert term["status"] == "COMPLETED"
        assert term["state"] == a2a.TASK_COMPLETED
        assert term["steps_count"] == 2

        # 3. In-progress (RUNNING) task emits persisted steps without a terminal event.
        store.create_run("run-sse-running", "DEMO_TEAM_ID", "demo_team", {"topic": "Live"})
        store.record_step(
            "run-sse-running",
            "planner",
            0,
            "COMPLETED",
            {"request": {"topic": "Live"}},
            output={"steps": ["research:Live"]},
        )
        run_status, run_headers, run_body = _http_get_raw(
            f"http://127.0.0.1:{port}/tasks/run-sse-running/events"
        )
        assert run_status == 200
        assert run_headers["content-type"].startswith("text/event-stream")
        running_events = _parse_sse(run_body)
        assert [ev["event"] for ev in running_events] == ["task.step"]
        assert running_events[0]["data"]["step_id"] == "planner"

        # 4. Canceling the running task makes SSE emit task.terminal with canceled status.
        cancel_status, cancel_resp = _http_post_json(
            f"http://127.0.0.1:{port}/a2a",
            {
                "jsonrpc": "2.0",
                "id": "sse-cancel-1",
                "method": "tasks/cancel",
                "params": {"id": "run-sse-running"},
            },
        )
        assert cancel_status == 200
        assert "error" not in cancel_resp

        _, _, canceled_body = _http_get_raw(f"http://127.0.0.1:{port}/tasks/run-sse-running/events")
        canceled_events = _parse_sse(canceled_body)
        assert [ev["event"] for ev in canceled_events] == ["task.step", "task.terminal"]
        assert canceled_events[1]["data"]["status"] == "canceled"
        assert canceled_events[1]["data"]["state"] == a2a.TASK_CANCELED

        # 5. Unknown run_id returns 404 JSON error.
        nf_status, nf_headers, nf_body = _http_get_raw(f"http://127.0.0.1:{port}/tasks/unknown-run/events")
        assert nf_status == 404
        assert "application/json" in nf_headers["content-type"]
        assert "unknown-run" in json.loads(nf_body)["error"]
    finally:
        server.shutdown()
        server.server_close()
        store.close()
