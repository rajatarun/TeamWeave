"""Local SQLite-backed ``team.json`` workflow runner (zero AWS required).

Loads a TeamWeave ``team.json`` configuration, validates step references and
DAG ordering, executes each agent step via an injectable ``agent_executor``
(defaulting to a stdlib ``urllib`` OpenAI-compatible chat completions client),
validates JSON Schema handoffs between steps, and persists run/step state to
SQLite WAL.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import urllib.request
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema.exceptions import ValidationError

from .local_policy import (
    LocalTeamPolicy,
    PolicyDeniedError,
    append_telemetry_event,
    format_team_telemetry_event,
    normalize_trace_id,
)
from .schema_validate import validate_output

AMBIENT_INPUTS: frozenset[str] = frozenset(
    {
        "request",
        "rag_context",
        "research_context",
        "owner_profile_context",
        "gemini_brief",
        "owner",
        "rag_meta",
    }
)

AgentExecutor = Callable[[dict[str, Any], dict[str, Any], dict[str, Any] | None], dict[str, Any]]


class WorkflowValidationError(ValueError):
    """Raised when a team.json workflow has invalid agents, inputs, or cycles."""


class StepSchemaError(ValueError):
    """Raised when a step's output fails JSON Schema validation."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_and_order_workflow(
    config: dict[str, Any],
    *,
    extra_ambient_keys: set[str] | frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Validate step/agent/input references and return steps in topological order."""
    agents_list = config.get("agents")
    workflow = config.get("workflow")
    if not isinstance(agents_list, list) or not agents_list:
        raise WorkflowValidationError("team.json must define a non-empty 'agents' list")
    if not isinstance(workflow, list) or not workflow:
        raise WorkflowValidationError("team.json must define a non-empty 'workflow' list")

    agent_ids: set[str] = set()
    for agent in agents_list:
        aid = agent.get("id") if isinstance(agent, dict) else None
        if not isinstance(aid, str) or not aid.strip():
            raise WorkflowValidationError("Every agent entry must have a non-empty string 'id'")
        if aid in agent_ids:
            raise WorkflowValidationError(f"Duplicate agent id: {aid!r}")
        agent_ids.add(aid)

    step_map: dict[str, dict[str, Any]] = {}
    declared_order: dict[str, int] = {}
    for idx, step in enumerate(workflow):
        sid = step.get("step") if isinstance(step, dict) else None
        if not isinstance(sid, str) or not sid.strip():
            raise WorkflowValidationError(f"Workflow entry at index {idx} is missing string 'step'")
        if sid in step_map:
            raise WorkflowValidationError(f"Duplicate workflow step: {sid!r}")
        if sid not in agent_ids:
            raise WorkflowValidationError(f"Workflow step {sid!r} has no matching agent id")
        step_map[sid] = step
        declared_order[sid] = idx

    ambient = set(AMBIENT_INPUTS) | set(extra_ambient_keys)
    deps: dict[str, set[str]] = {sid: set() for sid in step_map}
    dependents: dict[str, list[str]] = {sid: [] for sid in step_map}

    for sid, step in step_map.items():
        for raw_ref in step.get("inputs") or []:
            base = str(raw_ref).split(".")[0]
            if base in ambient:
                continue
            if base not in step_map:
                raise WorkflowValidationError(
                    f"Step {sid!r} input {raw_ref!r} references unknown step {base!r}"
                )
            if base == sid:
                raise WorkflowValidationError(f"Step {sid!r} cannot depend on itself")
            if base not in deps[sid]:
                deps[sid].add(base)
                dependents[base].append(sid)

    in_degree = {sid: len(dep_set) for sid, dep_set in deps.items()}
    ready = deque(sorted((sid for sid, deg in in_degree.items() if deg == 0), key=declared_order.__getitem__))
    ordered_ids: list[str] = []

    while ready:
        current = ready.popleft()
        ordered_ids.append(current)
        newly_ready: list[str] = []
        for nxt in dependents[current]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                newly_ready.append(nxt)
        if newly_ready:
            merged = sorted(list(ready) + newly_ready, key=declared_order.__getitem__)
            ready = deque(merged)

    if len(ordered_ids) != len(step_map):
        remaining = sorted(set(step_map) - set(ordered_ids), key=declared_order.__getitem__)
        raise WorkflowValidationError(f"Workflow dependency cycle detected involving steps: {remaining}")

    return [step_map[sid] for sid in ordered_ids]


class SQLiteTeamRunStore:
    """SQLite WAL persistence store for local TeamWeave runs and step artifacts."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._init_schema()

    def _configure(self) -> None:
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS team_runs (
                run_id TEXT PRIMARY KEY,
                team_id TEXT NOT NULL,
                team_name TEXT NOT NULL,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS team_steps (
                run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                status TEXT NOT NULL,
                inputs_json TEXT NOT NULL,
                output_json TEXT,
                schema_ref TEXT,
                error TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                PRIMARY KEY (run_id, step_id),
                FOREIGN KEY (run_id) REFERENCES team_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        self._conn.commit()

    def journal_mode(self) -> str:
        row = self._conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower() if row else ""

    def create_run(
        self,
        run_id: str,
        team_id: str,
        team_name: str,
        request: dict[str, Any] | str,
    ) -> None:
        now = _utc_now()
        self._conn.execute(
            """
            INSERT INTO team_runs (run_id, team_id, team_name, status, request_json, error, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (run_id, team_id, team_name, "RUNNING", json.dumps(request), now, now),
        )
        self._conn.commit()

    def update_run_status(self, run_id: str, status: str, *, error: str | None = None) -> None:
        self._conn.execute(
            "UPDATE team_runs SET status = ?, error = ?, updated_at = ? WHERE run_id = ?",
            (status, error, _utc_now(), run_id),
        )
        self._conn.commit()

    def record_step(
        self,
        run_id: str,
        step_id: str,
        seq: int,
        status: str,
        inputs: dict[str, Any],
        *,
        output: dict[str, Any] | None = None,
        schema_ref: str | None = None,
        error: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        now = _utc_now()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO team_steps (
                run_id, step_id, seq, status, inputs_json, output_json,
                schema_ref, error, started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                step_id,
                seq,
                status,
                json.dumps(inputs),
                json.dumps(output) if output is not None else None,
                schema_ref,
                error,
                started_at or now,
                finished_at or now,
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM team_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        step_rows = self._conn.execute(
            "SELECT * FROM team_steps WHERE run_id = ? ORDER BY seq ASC",
            (run_id,),
        ).fetchall()
        steps = [
            {
                "step_id": sr["step_id"],
                "seq": sr["seq"],
                "status": sr["status"],
                "inputs": json.loads(sr["inputs_json"]),
                "output": json.loads(sr["output_json"]) if sr["output_json"] is not None else None,
                "schema_ref": sr["schema_ref"],
                "error": sr["error"],
                "started_at": sr["started_at"],
                "finished_at": sr["finished_at"],
            }
            for sr in step_rows
        ]
        return {
            "run_id": row["run_id"],
            "team_id": row["team_id"],
            "team_name": row["team_name"],
            "status": row["status"],
            "request": json.loads(row["request_json"]),
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "steps": steps,
        }

    def list_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT run_id, team_id, team_name, status, error, created_at, updated_at "
            "FROM team_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_runs(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM team_runs").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()


class OpenAICompatExecutor:
    """Zero-dependency OpenAI-compatible HTTP executor using stdlib ``urllib``."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        model: str = "local",
        timeout_sec: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec

    def __call__(
        self,
        agent: dict[str, Any],
        inputs: dict[str, Any],
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        system_prompt = str(agent.get("goal_template") or f"You are {agent.get('name', agent.get('id'))}.")
        user_payload = {
            "inputs": inputs,
            "required_json_schema": schema,
        }
        body = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, dict):
            return content
        parsed = json.loads(str(content))
        if not isinstance(parsed, dict):
            raise ValueError("LLM response content must decode to a JSON object")
        return parsed


@dataclass
class TeamRunResult:
    run_id: str
    status: str
    step_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_step: str | None = None
    error: str | None = None


class LocalTeamRunner:
    """Executes a TeamWeave ``team.json`` workflow locally with SQLite persistence."""

    def __init__(
        self,
        team_config: dict[str, Any] | str | Path,
        *,
        store: SQLiteTeamRunStore | None = None,
        db_path: str | Path | None = None,
        agent_executor: AgentExecutor | None = None,
        base_dir: Path | None = None,
        policy: LocalTeamPolicy | None = None,
        telemetry_path: str | Path | None = None,
    ) -> None:
        if isinstance(team_config, (str, Path)):
            cfg_path = Path(team_config)
            self.config: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.base_dir = base_dir or cfg_path.parent
        else:
            self.config = team_config
            self.base_dir = base_dir or Path.cwd()

        self._owns_store = store is None
        self.store = store if store is not None else SQLiteTeamRunStore(db_path or ":memory:")
        self.agent_executor = agent_executor or self._default_executor()
        self.policy = policy
        self.telemetry_path = Path(telemetry_path) if telemetry_path is not None else None
        self.agents_by_id: dict[str, dict[str, Any]] = {
            a["id"]: a for a in (self.config.get("agents") or []) if isinstance(a, dict) and "id" in a
        }

    def _default_executor(self) -> AgentExecutor:
        base_url = os.environ.get("TEAMWEAVE_LLM_BASE_URL", "").strip()
        if base_url:
            return OpenAICompatExecutor(
                base_url=base_url,
                api_key=os.environ.get("TEAMWEAVE_LLM_API_KEY", ""),
                model=os.environ.get("TEAMWEAVE_LLM_MODEL", "local"),
            )

        def _missing_executor(
            agent: dict[str, Any],
            inputs: dict[str, Any],
            schema: dict[str, Any] | None,
        ) -> dict[str, Any]:
            raise RuntimeError(
                "No agent_executor provided and TEAMWEAVE_LLM_BASE_URL is not set"
            )

        return _missing_executor

    def _resolve_schema(self, schema_ref: str | None) -> dict[str, Any] | None:
        if not schema_ref:
            return None
        schemas = self.config.get("schemas") or {}
        if schema_ref not in schemas:
            raise WorkflowValidationError(f"Schema reference {schema_ref!r} is not defined in team.json")
        entry = schemas[schema_ref]
        if isinstance(entry, dict) and "path" in entry and "type" not in entry and "properties" not in entry:
            raw_path = Path(entry["path"])
            candidates = [raw_path, self.base_dir / raw_path]
            for cand in candidates:
                if cand.is_file():
                    return json.loads(cand.read_text(encoding="utf-8"))
            raise WorkflowValidationError(f"Schema file for {schema_ref!r} not found at {entry['path']!r}")
        if not isinstance(entry, dict):
            raise WorkflowValidationError(f"Schema {schema_ref!r} must be a JSON object")
        return entry

    def _build_step_inputs(
        self,
        step: dict[str, Any],
        ambient: dict[str, Any],
        outputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for ref in step.get("inputs") or []:
            ref_str = str(ref)
            if "." in ref_str:
                base, attr = ref_str.split(".", 1)
                if base in outputs:
                    val = outputs[base]
                    resolved[ref_str] = val if attr == "output" else val.get(attr)
                elif base in ambient:
                    resolved[ref_str] = ambient[base]
            elif ref_str in outputs:
                resolved[ref_str] = outputs[ref_str]
            elif ref_str in ambient:
                resolved[ref_str] = ambient[ref_str]
        return resolved

    def _emit_telemetry(
        self,
        *,
        team_name: str,
        run_id: str,
        step_name: str,
        status: str,
        latency_ms: float,
        trace_id: str,
        attributes: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        if self.telemetry_path is None:
            return
        event = format_team_telemetry_event(
            team_name=team_name,
            run_id=run_id,
            step_name=step_name,
            status=status,
            latency_ms=latency_ms,
            trace_id=trace_id,
            attributes=attributes,
            error_message=error_message,
        )
        append_telemetry_event(self.telemetry_path, event)

    def run(
        self,
        request: dict[str, Any] | str,
        *,
        run_id: str | None = None,
        ambient_context: dict[str, Any] | None = None,
        raise_on_error: bool = False,
        trace_id: str | None = None,
    ) -> TeamRunResult:
        """Validate workflow, execute all steps in DAG order, and persist to SQLite."""
        ambient: dict[str, Any] = {"request": request, **(ambient_context or {})}
        ordered_steps = validate_and_order_workflow(
            self.config, extra_ambient_keys=set(ambient.keys())
        )

        team_meta = self.config.get("team") or {}
        team_id = str(team_meta.get("team_id") or team_meta.get("name") or "local_team")
        team_name = str(team_meta.get("name") or team_id)
        active_run_id = run_id or str(uuid.uuid4())
        run_trace_id = normalize_trace_id(trace_id)
        run_started_monotonic = time.monotonic()

        self.store.create_run(active_run_id, team_id, team_name, request)
        outputs: dict[str, dict[str, Any]] = {}

        for seq, step in enumerate(ordered_steps):
            step_id = step["step"]
            agent = self.agents_by_id[step_id]
            step_role = str(agent.get("role") or agent.get("id") or step_id)
            schema_ref = agent.get("schema_ref")
            step_inputs = self._build_step_inputs(step, ambient, outputs)
            step_started_monotonic = time.monotonic()
            started_at = _utc_now()
            elapsed_seconds = step_started_monotonic - run_started_monotonic

            if self.policy is not None:
                allowed, denial_reason = self.policy.validate_step(seq, step_role, elapsed_seconds)
                if not allowed:
                    err_msg = denial_reason or f"Step {step_id!r} denied by LocalTeamPolicy"
                    latency_ms = (time.monotonic() - step_started_monotonic) * 1000.0
                    self.store.record_step(
                        active_run_id,
                        step_id,
                        seq,
                        "policy_denied",
                        step_inputs,
                        output=None,
                        schema_ref=schema_ref,
                        error=err_msg,
                        started_at=started_at,
                        finished_at=_utc_now(),
                    )
                    self.store.update_run_status(active_run_id, "policy_denied", error=err_msg)
                    self._emit_telemetry(
                        team_name=team_name,
                        run_id=active_run_id,
                        step_name=step_id,
                        status="policy_denied",
                        latency_ms=latency_ms,
                        trace_id=run_trace_id,
                        attributes={"role": step_role, "step_index": seq, "policy_denied": True},
                        error_message=err_msg,
                    )
                    if raise_on_error:
                        raise PolicyDeniedError(err_msg)
                    return TeamRunResult(
                        run_id=active_run_id,
                        status="policy_denied",
                        step_outputs=outputs,
                        failed_step=step_id,
                        error=err_msg,
                    )

            try:
                schema = self._resolve_schema(schema_ref)
                output = self.agent_executor(agent, step_inputs, schema)
                if not isinstance(output, dict):
                    raise StepSchemaError(
                        f"Step {step_id!r} returned {type(output).__name__}; expected dict"
                    )
                if schema is not None:
                    try:
                        validate_output(output, schema)
                    except ValidationError as exc:
                        raise StepSchemaError(
                            f"Step {step_id!r} failed schema {schema_ref!r}: {exc.message}"
                        ) from exc
            except Exception as exc:  # noqa: BLE001
                err_msg = str(exc)
                latency_ms = (time.monotonic() - step_started_monotonic) * 1000.0
                self.store.record_step(
                    active_run_id,
                    step_id,
                    seq,
                    "FAILED",
                    step_inputs,
                    output=None,
                    schema_ref=schema_ref,
                    error=err_msg,
                    started_at=started_at,
                    finished_at=_utc_now(),
                )
                self.store.update_run_status(active_run_id, "FAILED", error=err_msg)
                self._emit_telemetry(
                    team_name=team_name,
                    run_id=active_run_id,
                    step_name=step_id,
                    status="error",
                    latency_ms=latency_ms,
                    trace_id=run_trace_id,
                    attributes={"role": step_role, "step_index": seq},
                    error_message=err_msg,
                )
                if raise_on_error:
                    raise
                return TeamRunResult(
                    run_id=active_run_id,
                    status="FAILED",
                    step_outputs=outputs,
                    failed_step=step_id,
                    error=err_msg,
                )

            outputs[step_id] = output
            latency_ms = (time.monotonic() - step_started_monotonic) * 1000.0
            self.store.record_step(
                active_run_id,
                step_id,
                seq,
                "COMPLETED",
                step_inputs,
                output=output,
                schema_ref=schema_ref,
                started_at=started_at,
                finished_at=_utc_now(),
            )
            self._emit_telemetry(
                team_name=team_name,
                run_id=active_run_id,
                step_name=step_id,
                status="ok",
                latency_ms=latency_ms,
                trace_id=run_trace_id,
                attributes={"role": step_role, "step_index": seq},
            )

        self.store.update_run_status(active_run_id, "COMPLETED")
        return TeamRunResult(
            run_id=active_run_id,
            status="COMPLETED",
            step_outputs=outputs,
        )

    def close(self) -> None:
        if self._owns_store:
            self.store.close()


def _run_serve(
    team_path: str,
    *,
    db_path: str,
    host: str,
    port: int,
    base_url: str,
    model: str,
) -> int:
    from .local_a2a_server import LocalA2AService, create_http_server

    executor: AgentExecutor | None = None
    if base_url:
        executor = OpenAICompatExecutor(
            base_url=base_url,
            api_key=os.environ.get("TEAMWEAVE_LLM_API_KEY", ""),
            model=model,
        )

    store = SQLiteTeamRunStore(db_path)
    runner = LocalTeamRunner(team_path, store=store, agent_executor=executor)
    team_meta = runner.config.get("team") or {}
    team_name = str(team_meta.get("name") or team_meta.get("team_id") or Path(team_path).stem)
    service = LocalA2AService(
        {team_name: runner.config},
        runner,
        store,
        base_url=f"http://{host}:{port}",
    )
    server = create_http_server(service, host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    import sys

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "serve":
        serve_parser = argparse.ArgumentParser(
            prog="teamweave-local serve",
            description="Serve a TeamWeave team.json over a local A2A v1.0 HTTP/JSON-RPC server.",
        )
        serve_parser.add_argument("--team", required=True, help="Path to team.json")
        serve_parser.add_argument("--db", default="teamweave_runs.db", help="SQLite database path")
        serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host")
        serve_parser.add_argument("--port", type=int, default=8787, help="Bind port")
        serve_parser.add_argument("--base-url", default="", help="OpenAI-compatible API base URL")
        serve_parser.add_argument("--model", default="local", help="Model name for OpenAI-compatible API")
        s_args = serve_parser.parse_args(raw_argv[1:])
        return _run_serve(
            s_args.team,
            db_path=s_args.db,
            host=s_args.host,
            port=s_args.port,
            base_url=s_args.base_url,
            model=s_args.model,
        )

    parser = argparse.ArgumentParser(description="Run a TeamWeave team.json locally with SQLite persistence.")
    parser.add_argument("--team", required=True, help="Path to team.json")
    parser.add_argument("--request", default=None, help="Input request (JSON string or plain text)")
    parser.add_argument("--db", default="teamweave_runs.db", help="SQLite database path")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default="local", help="Model name for OpenAI-compatible API")
    parser.add_argument("--serve", action="store_true", help="Start local A2A v1.0 HTTP/JSON-RPC server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host when --serve is set")
    parser.add_argument("--port", type=int, default=8787, help="Bind port when --serve is set")
    args = parser.parse_args(raw_argv)

    if args.serve:
        return _run_serve(
            args.team,
            db_path=args.db,
            host=args.host,
            port=args.port,
            base_url=args.base_url,
            model=args.model,
        )

    if args.request is None:
        parser.error("the following arguments are required: --request (unless --serve or 'serve' is used)")

    try:
        req_payload: dict[str, Any] | str = json.loads(args.request)
    except json.JSONDecodeError:
        req_payload = args.request

    executor: AgentExecutor | None = None
    if args.base_url:
        executor = OpenAICompatExecutor(
            base_url=args.base_url,
            api_key=os.environ.get("TEAMWEAVE_LLM_API_KEY", ""),
            model=args.model,
        )

    runner = LocalTeamRunner(args.team, db_path=args.db, agent_executor=executor)
    try:
        result = runner.run(req_payload)
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "failed_step": result.failed_step,
                    "error": result.error,
                    "steps_completed": list(result.step_outputs.keys()),
                },
                indent=2,
            )
        )
        return 0 if result.status == "COMPLETED" else 1
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
