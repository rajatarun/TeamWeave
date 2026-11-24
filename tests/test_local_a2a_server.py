"""Unit and HTTP integration tests for TeamWeave LocalA2AService and stdlib HTTP server."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from src.orchestrator import a2a
from src.orchestrator.local_a2a_server import (
    ERR_INVALID_PARAMS,
    ERR_METHOD_NOT_FOUND,
    ERR_PARSE,
    ERR_TASK_NOT_FOUND,
    LocalA2AService,
    create_http_server,
)
from src.orchestrator.local_runner import LocalTeamRunner, SQLiteTeamRunStore, main


def _multi_step_team() -> dict[str, Any]:
    return {
        "team": {"team_id": "A2A_DEMO", "name": "demo_team", "version": "v1"},
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


def _http_get_json(url: str) -> tuple[int, dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: Any) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_http_agent_card_tasks_send_and_tasks_get_end_to_end(tmp_path: Path) -> None:
    db_path = tmp_path / "a2a_runs.db"
    store = SQLiteTeamRunStore(db_path)
    team_cfg = _multi_step_team()
    runner = LocalTeamRunner(team_cfg, store=store, agent_executor=_demo_executor)

    service = LocalA2AService({"demo_team": team_cfg}, runner, store, base_url="http://127.0.0.1:0")
    server = create_http_server(service, "127.0.0.1", 0)
    port = server.server_address[1]
    service.base_url = f"http://127.0.0.1:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # 1. GET /.well-known/agent-card.json returns a valid A2A v1.0 card.
        status, card = _http_get_json(f"http://127.0.0.1:{port}/.well-known/agent-card.json")
        assert status == 200
        assert card["name"] == "TeamWeave"
        assert card["supportedInterfaces"][0]["protocolVersion"] == a2a.A2A_PROTOCOL_VERSION
        assert card["supportedInterfaces"][0]["protocolBinding"] == a2a.HTTP_JSON_BINDING
        assert card["supportedInterfaces"][0]["url"] == f"http://127.0.0.1:{port}/a2a"
        assert {s["id"] for s in card["skills"]} == {"demo_team.planner", "demo_team.writer"}

        # 2. POST / tasks/send executes the multi-step pipeline and returns terminal A2A task.
        status, send_resp = _http_post_json(
            f"http://127.0.0.1:{port}/",
            {
                "jsonrpc": "2.0",
                "id": "req-1",
                "method": "tasks/send",
                "params": {
                    "id": "run-a2a-001",
                    "team": "demo_team",
                    "input": {"topic": "EdgeOS"},
                },
            },
        )
        assert status == 200
        assert send_resp["id"] == "req-1"
        assert "error" not in send_resp
        task = send_resp["result"]
        assert task["id"] == "run-a2a-001"
        assert task["status"]["state"] == a2a.TASK_COMPLETED
        assert a2a.is_terminal(task["status"]["state"])
        artifact_payload = json.loads(task["artifacts"][0]["parts"][0]["text"])
        assert artifact_payload["planner"] == {"steps": ["research:EdgeOS", "draft:EdgeOS"]}
        assert artifact_payload["writer"] == {"summary": "research:EdgeOS -> draft:EdgeOS"}

        # 3. POST /a2a tasks/get retrieves persisted task by run_id.
        status, get_resp = _http_post_json(
            f"http://127.0.0.1:{port}/a2a",
            {
                "jsonrpc": "2.0",
                "id": "req-2",
                "method": "tasks/get",
                "params": {"run_id": "run-a2a-001"},
            },
        )
        assert status == 200
        assert "error" not in get_resp
        fetched = get_resp["result"]
        assert fetched["id"] == "run-a2a-001"
        assert fetched["status"]["state"] == a2a.TASK_COMPLETED
        fetched_outputs = json.loads(fetched["artifacts"][0]["parts"][0]["text"])
        assert fetched_outputs["writer"]["summary"] == "research:EdgeOS -> draft:EdgeOS"

        # 4. Invalid JSON body returns HTTP 400 with JSON-RPC parse error.
        bad_status, bad_body = _http_post_json(f"http://127.0.0.1:{port}/", b"{not-json")
        assert bad_status == 400
        assert bad_body["error"]["code"] == ERR_PARSE
    finally:
        server.shutdown()
        server.server_close()
        store.close()


def test_tasks_send_accepts_a2a_message_parts() -> None:
    store = SQLiteTeamRunStore(":memory:")
    team_cfg = _multi_step_team()
    runner = LocalTeamRunner(team_cfg, store=store, agent_executor=_demo_executor)
    service = LocalA2AService({"demo_team": team_cfg}, runner, store)
    try:
        resp = service.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tasks/send",
                "params": {
                    "message": {
                        "role": "ROLE_USER",
                        "parts": [{"text": "WeaveA2A"}],
                    }
                },
            }
        )
        assert "error" not in resp
        task = resp["result"]
        assert task["status"]["state"] == a2a.TASK_COMPLETED
        outputs = json.loads(task["artifacts"][0]["parts"][0]["text"])
        assert outputs["writer"]["summary"] == "research:WeaveA2A -> draft:WeaveA2A"
    finally:
        store.close()


def test_jsonrpc_errors_for_unknown_team_method_and_task() -> None:
    store = SQLiteTeamRunStore(":memory:")
    team_cfg = _multi_step_team()
    runner = LocalTeamRunner(team_cfg, store=store, agent_executor=_demo_executor)
    service = LocalA2AService({"demo_team": team_cfg}, runner, store)
    try:
        unknown_method = service.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tasks/unknown", "params": {}})
        assert unknown_method["error"]["code"] == ERR_METHOD_NOT_FOUND

        unknown_team = service.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tasks/send",
                "params": {"team": "nonexistent_team", "input": {"topic": "x"}},
            }
        )
        assert unknown_team["error"]["code"] == ERR_INVALID_PARAMS
        assert "Unknown team" in unknown_team["error"]["message"]

        missing_input = service.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tasks/send",
                "params": {"team": "demo_team"},
            }
        )
        assert missing_input["error"]["code"] == ERR_INVALID_PARAMS

        missing_task = service.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tasks/get",
                "params": {"id": "does-not-exist"},
            }
        )
        assert missing_task["error"]["code"] == ERR_TASK_NOT_FOUND
    finally:
        store.close()


def test_cli_serve_subcommand_and_flag_wire_local_a2a_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team_path = tmp_path / "team.json"
    team_path.write_text(json.dumps(_multi_step_team()), encoding="utf-8")
    db_path = tmp_path / "cli_a2a.db"

    started: list[tuple[str, int]] = []

    class _FakeServer:
        def __init__(self, host: str, port: int) -> None:
            self._host = host
            self._port = port
            self.closed = False

        def serve_forever(self) -> None:
            started.append((self._host, self._port))

        def server_close(self) -> None:
            self.closed = True

    def _fake_create_http_server(service: LocalA2AService, host: str, port: int) -> _FakeServer:
        card = service.get_agent_card()
        assert card["skills"]
        return _FakeServer(host, port)

    monkeypatch.setattr("src.orchestrator.local_a2a_server.create_http_server", _fake_create_http_server)

    rc1 = main(["serve", "--team", str(team_path), "--db", str(db_path), "--host", "127.0.0.1", "--port", "9001"])
    assert rc1 == 0

    rc2 = main(["--team", str(team_path), "--db", str(db_path), "--serve", "--host", "127.0.0.1", "--port", "9002"])
    assert rc2 == 0

    assert started == [("127.0.0.1", 9001), ("127.0.0.1", 9002)]
