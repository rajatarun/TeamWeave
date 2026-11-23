"""Standalone local A2A v1.0 HTTP/JSON-RPC server over LocalTeamRunner.

Serves ``GET /.well-known/agent-card.json`` and ``POST /`` (or ``POST /a2a``)
JSON-RPC 2.0 requests (``tasks/send``, ``tasks/get``) backed by
``LocalTeamRunner`` and ``SQLiteTeamRunStore`` with zero AWS dependencies.
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from . import a2a
from .local_runner import (
    AgentExecutor,
    LocalTeamRunner,
    SQLiteTeamRunStore,
    WorkflowValidationError,
)

JSONRPC_VERSION = "2.0"

ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_TASK_NOT_FOUND = -32004


def _jsonrpc_result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": result}


def _jsonrpc_error(
    req_id: Any,
    code: int,
    message: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "error": err}


def _normalize_run_status(status: str) -> str:
    """Map LocalTeamRunner statuses (COMPLETED) to canonical a2a.py statuses (SUCCEEDED)."""
    upper = str(status or "").strip().upper()
    if upper == "COMPLETED":
        return "SUCCEEDED"
    return upper


def _extract_input_payload(params: dict[str, Any]) -> dict[str, Any] | str | None:
    """Extract task input from A2A ``params`` (``input``, ``request``, or ``message``)."""
    if "input" in params and params["input"] is not None:
        val = params["input"]
        if isinstance(val, (dict, str)) and (isinstance(val, dict) or val.strip()):
            return val
    if "request" in params and params["request"] is not None:
        val = params["request"]
        if isinstance(val, (dict, str)) and (isinstance(val, dict) or val.strip()):
            return val
    message = params.get("message")
    if isinstance(message, dict):
        for part in message.get("parts") or []:
            if isinstance(part, dict) and isinstance(part.get("data"), dict):
                return part["data"]
        text = a2a.text_of_message(message)
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            return text
    return None


class LocalA2AService:
    """Local A2A v1.0 service wrapping ``LocalTeamRunner`` and ``SQLiteTeamRunStore``."""

    def __init__(
        self,
        teams_by_name: dict[str, dict[str, Any]],
        runner: LocalTeamRunner | None,
        store: SQLiteTeamRunStore,
        *,
        base_url: str = "http://127.0.0.1:8787",
        version: str = "0.2.0",
        agent_executor: AgentExecutor | None = None,
    ) -> None:
        self._teams_by_name: dict[str, dict[str, Any]] = dict(teams_by_name)
        self.runner = runner
        self.store = store
        self.base_url = base_url.rstrip("/")
        self.version = version
        self._agent_executor = agent_executor or (runner.agent_executor if runner is not None else None)
        self._lock = threading.Lock()

    def _teams_for_card(self) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for key, cfg in self._teams_by_name.items():
            team_meta = dict(cfg.get("team") or {})
            if not str(team_meta.get("name") or "").strip():
                team_meta["name"] = key
            normalized[key] = {**cfg, "team": team_meta}
        return normalized

    def get_agent_card(self) -> dict[str, Any]:
        """Build the A2A v1.0 Agent Card advertising all registered teams/agents."""
        return a2a.build_agent_card(
            base_url=self.base_url,
            version=self.version,
            teams=self._teams_for_card(),
        )

    def _resolve_team_name(self, params: dict[str, Any]) -> str:
        explicit = params.get("team")
        if explicit is None and isinstance(params.get("metadata"), dict):
            explicit = params["metadata"].get("team")
        if explicit is None and isinstance(params.get("skill_id"), str) and "." in params["skill_id"]:
            explicit = params["skill_id"].split(".", 1)[0]

        if explicit is not None:
            team_name = str(explicit).strip()
            if team_name not in self._teams_by_name:
                raise KeyError(f"Unknown team: {team_name!r}")
            return team_name

        if len(self._teams_by_name) == 1:
            return next(iter(self._teams_by_name))

        raise ValueError("tasks/send requires 'team' when multiple teams are registered")

    def _runner_for_team(self, team_name: str) -> LocalTeamRunner:
        team_cfg = self._teams_by_name[team_name]
        if self.runner is not None:
            runner_team_meta = self.runner.config.get("team") or {}
            runner_names = {
                str(runner_team_meta.get("name") or ""),
                str(runner_team_meta.get("team_id") or ""),
            }
            if (self.runner.config is team_cfg or team_name in runner_names) and self.runner.store is self.store:
                return self.runner
        return LocalTeamRunner(
            team_cfg,
            store=self.store,
            agent_executor=self._agent_executor,
        )

    def _handle_tasks_send(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        try:
            team_name = self._resolve_team_name(params)
        except KeyError as exc:
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, str(exc.args[0]))
        except ValueError as exc:
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, str(exc))

        payload = _extract_input_payload(params)
        if payload is None:
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "tasks/send requires 'input', 'request', or 'message'")

        run_id = params.get("id") or params.get("run_id") or params.get("task_id")
        if run_id is not None:
            run_id = str(run_id).strip() or None
        context_id = str(params.get("contextId") or params.get("sessionId") or "")
        ambient = params.get("ambient_context") if isinstance(params.get("ambient_context"), dict) else None

        with self._lock:
            active_runner = self._runner_for_team(team_name)
            try:
                result = active_runner.run(
                    payload,
                    run_id=run_id,
                    ambient_context=ambient,
                )
            except WorkflowValidationError as exc:
                return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, str(exc))

        task = a2a.task_for_run(
            result.run_id,
            _normalize_run_status(result.status),
            context_id=context_id,
            result=result.step_outputs,
            error=result.error or "",
        )
        return _jsonrpc_result(req_id, task)

    def _handle_tasks_get(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        run_id = params.get("id") or params.get("run_id") or params.get("task_id")
        if not isinstance(run_id, str) or not run_id.strip():
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "tasks/get requires a non-empty 'id' or 'run_id'")

        with self._lock:
            saved = self.store.get_run(run_id.strip())
        if saved is None:
            return _jsonrpc_error(req_id, ERR_TASK_NOT_FOUND, f"Task not found: {run_id!r}")

        step_outputs = {
            step["step_id"]: step["output"]
            for step in saved.get("steps") or []
            if step.get("output") is not None
        }
        task = a2a.task_for_run(
            saved["run_id"],
            _normalize_run_status(saved["status"]),
            result=step_outputs,
            error=saved.get("error") or "",
        )
        return _jsonrpc_result(req_id, task)

    def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a JSON-RPC 2.0 A2A request (``tasks/send`` or ``tasks/get``)."""
        if not isinstance(request, dict):
            return _jsonrpc_error(None, ERR_INVALID_REQUEST, "JSON-RPC request must be a JSON object")

        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params")
        if params is None:
            params = {}

        if not isinstance(method, str) or not method.strip():
            return _jsonrpc_error(req_id, ERR_INVALID_REQUEST, "Missing JSON-RPC 'method'")
        if not isinstance(params, dict):
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "JSON-RPC 'params' must be an object")

        if method == "tasks/send":
            return self._handle_tasks_send(req_id, params)
        if method == "tasks/get":
            return self._handle_tasks_get(req_id, params)

        return _jsonrpc_error(req_id, ERR_METHOD_NOT_FOUND, f"Unknown A2A method: {method!r}")


def create_http_server(
    service: LocalA2AService,
    host: str = "127.0.0.1",
    port: int = 8787,
) -> HTTPServer:
    """Create a lightweight stdlib HTTPServer serving the Agent Card and JSON-RPC A2A endpoints."""

    class _A2AHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == a2a.AGENT_CARD_PATH:
                self._send_json(HTTPStatus.OK, service.get_agent_card())
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Not found: {path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path not in ("/", a2a.A2A_BASE_PATH, f"{a2a.A2A_BASE_PATH}/"):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Not found: {path}"})
                return

            length_raw = self.headers.get("Content-Length", "0")
            try:
                content_length = int(length_raw)
            except ValueError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    _jsonrpc_error(None, ERR_INVALID_REQUEST, "Invalid Content-Length header"),
                )
                return

            raw_body = self.rfile.read(content_length) if content_length > 0 else b""
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    _jsonrpc_error(None, ERR_PARSE, "Invalid JSON payload"),
                )
                return

            response = service.handle_jsonrpc(payload)
            self._send_json(HTTPStatus.OK, response)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return HTTPServer((host, port), _A2AHandler)
