import json
import os
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from .config_loader import load_team_config
from .db import DbDao

sfn = boto3.client("stepfunctions")


# ASSUMPTION: Existing non-/team/task routes remain on this Lambda to avoid breaking current API consumers.
def _cors() -> Dict[str, str]:
    return {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "content-type",
        "access-control-allow-methods": "POST,GET,OPTIONS",
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
            started = sfn.start_execution(
                stateMachineArn=state_machine_arn,
                input=json.dumps({"team": team, "version": version, "request": request_obj}),
            )
            return _resp(202, {"run_id": started["executionArn"]})
        except ClientError as exc:
            return _resp(500, {"error": exc.response.get("Error", {}).get("Message", str(exc))})

    return _resp(404, {"error": "Route not found"})
