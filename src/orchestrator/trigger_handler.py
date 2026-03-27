import json
import os
import uuid
from typing import Any, Dict, Optional
from urllib.parse import unquote

import boto3
from botocore.exceptions import ClientError

from .config_loader import load_team_config
from .db import DbDao
from .logger import get_logger

sfn = boto3.client("stepfunctions", endpoint_url=os.environ.get("STEPFUNCTIONS_ENDPOINT_URL"))
lambda_client = boto3.client("lambda")
log = get_logger("trigger_handler")


# ASSUMPTION: Existing non-/team/task routes remain on this Lambda to avoid breaking current API consumers.
def _cors() -> Dict[str, str]:
    return {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "Content-Type,Authorization",
        "access-control-allow-methods": "OPTIONS,GET,POST,PUT,DELETE",
    }


def _resp(code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {"statusCode": code, "headers": _cors(), "body": json.dumps(body, ensure_ascii=False, default=str)}


def _method(event: Dict[str, Any]) -> str:
    return (event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod") or "").upper()


def _path(event: Dict[str, Any]) -> str:
    return event.get("rawPath") or event.get("path") or ""


def _qs(event: Dict[str, Any]) -> Dict[str, Any]:
    return event.get("queryStringParameters") or {}


def _json_body(event: Dict[str, Any]) -> Dict[str, Any]:
    b = event.get("body")
    if not b:
        return {}
    try:
        return json.loads(b) if isinstance(b, str) else b
    except Exception:
        return {}


def _start_async_execution(state_machine_arn: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    payload_with_run_id = {**payload, "run_id": run_id}
    log.info(
        "stepfunctions_start_execution_requested",
        extra={
            "run_id": run_id,
            "state_machine_arn": state_machine_arn,
            "operation": payload_with_run_id.get("operation", "team_task"),
            "method": payload_with_run_id.get("method"),
            "path": payload_with_run_id.get("path"),
            "team": payload_with_run_id.get("team"),
            "version": payload_with_run_id.get("version"),
        },
    )
    execution = sfn.start_execution(stateMachineArn=state_machine_arn, input=json.dumps(payload_with_run_id))
    execution_arn = execution.get("executionArn")
    execution_id = execution_arn.rsplit(":", 1)[-1] if isinstance(execution_arn, str) and execution_arn else None
    log.info(
        "stepfunctions_start_execution_succeeded",
        extra={
            "run_id": run_id,
            "execution_arn": execution_arn,
            "execution_id": execution_id,
        },
    )
    return _resp(
        202,
        {
            "run_id": run_id,
            "state_fn_execution_id": execution_id,
            "state_fn_execution_arn": execution_arn,
        },
    )


def _invoke_provision_lambda_sync(method: str, path: str, body: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
    function_name = os.environ.get("PROVISION_FUNCTION_NAME")
    if not function_name:
        return _resp(500, {"error": "PROVISION_FUNCTION_NAME is not configured"})

    invoke_payload = {
        "httpMethod": method,
        "path": path,
        "rawPath": path,
        "queryStringParameters": query,
        "body": json.dumps(body or {}),
    }
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(invoke_payload).encode("utf-8"),
    )
    payload_bytes = response.get("Payload").read()
    payload_text = payload_bytes.decode("utf-8") if payload_bytes else "{}"
    payload = json.loads(payload_text or "{}")

    status_code = int(payload.get("statusCode", 500))
    raw_body = payload.get("body")
    try:
        parsed_body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body
    except Exception:
        parsed_body = {"raw": raw_body}

    if isinstance(parsed_body, dict):
        return _resp(status_code, parsed_body)
    return _resp(status_code, {"result": parsed_body})


def _proxy_path_for_provision_compat(method: str, body: Dict[str, Any]) -> str:
    if method == "POST":
        return "/teams"
    team_name = body.get("team_name") or body.get("team") or body.get("name")
    return f"/teams/{team_name}" if team_name else "/teams"


def _is_agent_mgmt_route(path: str) -> bool:
    if not path:
        return False
    parts = [p for p in path.split("/") if p]
    if not parts:
        return False
    if parts[0] not in {"agents", "teams", "roles", "departments"}:
        return False
    return len(parts) <= 2


def _normalize_proxy_path(path: str) -> str:
    parts = [unquote(p) for p in path.split("/") if p]
    return "/" + "/".join(parts)


def _dao_from_optional_team(team: Optional[str], version: Optional[str]) -> DbDao:
    if team and version:
        _, team_raw = load_team_config(team, version)
        return DbDao.from_team_config(team_raw)
    return DbDao.from_team_config({})


def handler(event, context):
    m = _method(event)
    p = _path(event)

    if m == "OPTIONS":
        return {"statusCode": 200, "headers": _cors(), "body": ""}

    if p == "/improve/tasks" and m == "GET":
        qs = _qs(event)
        owner = (qs.get("owner") or "Tarun Raja")
        limit = int(qs.get("limit") or 50)
        team = qs.get("team")
        version = qs.get("version")
        dao = _dao_from_optional_team(team, version)
        return _resp(200, {"items": dao.list_tasks(owner, limit=limit)})

    if p == "/improve/task/done" and m == "POST":
        body = _json_body(event)
        owner = body.get("owner") or "Tarun Raja"
        task_id = body.get("task_id") or ""
        team = body.get("team")
        version = body.get("version")
        if not task_id:
            return _resp(400, {"error": "task_id required"})
        dao = _dao_from_optional_team(team, version)
        return _resp(200, dao.mark_task_done(owner, task_id))

    if p == "/team/task" and m == "POST":
        body = _json_body(event)
        team = body.get("team")
        version = body.get("version")
        request_obj = body.get("request") or {}

        if not team or not version:
            return _resp(400, {"error": "team and version required"})

        state_machine_arn = os.environ.get("STATE_MACHINE_ARN")
        if not state_machine_arn:
            return _resp(500, {"error": "STATE_MACHINE_ARN is not configured"})

        try:
            return _start_async_execution(
                state_machine_arn,
                {"team": team, "version": version, "request": request_obj},
            )
        except ClientError as exc:
            return _resp(500, {"error": exc.response.get("Error", {}).get("Message", str(exc))})

    if _is_agent_mgmt_route(p) and m in {"GET", "POST", "PUT", "DELETE"}:
        body = _json_body(event)
        query = _qs(event)
        normalized_path = _normalize_proxy_path(p)

        try:
            if m == "GET":
                return _invoke_provision_lambda_sync(m, normalized_path, body, query)

            state_machine_arn = os.environ.get("STATE_MACHINE_ARN")
            if not state_machine_arn:
                return _resp(500, {"error": "STATE_MACHINE_ARN is not configured"})

            payload = {
                "operation": "agent_management",
                "method": m,
                "path": normalized_path,
                "body": body,
                "query": query,
            }
            return _start_async_execution(state_machine_arn, payload)
        except ClientError as exc:
            return _resp(500, {"error": exc.response.get("Error", {}).get("Message", str(exc))})

    if p == "/provision" and m in {"POST", "DELETE"}:
        body = _json_body(event)
        state_machine_arn = os.environ.get("STATE_MACHINE_ARN")
        if not state_machine_arn:
            return _resp(500, {"error": "STATE_MACHINE_ARN is not configured"})

        try:
            return _start_async_execution(
                state_machine_arn,
                {
                    "operation": "agent_management",
                    "method": m,
                    "path": _proxy_path_for_provision_compat(m, body),
                    "body": body,
                    "query": _qs(event),
                },
            )
        except ClientError as exc:
            return _resp(500, {"error": exc.response.get("Error", {}).get("Message", str(exc))})

    return _resp(404, {"error": "Route not found"})
