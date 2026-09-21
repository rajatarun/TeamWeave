"""The A2A surface: the Agent Card, message:send and tasks/{id}.

Thin on purpose. Everything A2A-shaped lives in ``a2a.py`` as pure functions,
and everything TeamWeave-shaped already exists -- starting a run is the same
Step Functions execution ``POST /team/task`` starts, and reading one is the
same DescribeExecution the status handler reads. This module is the
translation between the two vocabularies and nothing else, so a client using
A2A and a client using the native API cannot drift into different behaviour.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

from . import a2a, run_ids
from .logger import get_logger

log = get_logger("a2a_handler")

sfn = boto3.client("stepfunctions")
s3 = boto3.client("s3")

# A2A's own media type. Clients content-negotiate on it.
A2A_MEDIA_TYPE = "application/a2a+json"


def _cors(media_type: str = A2A_MEDIA_TYPE) -> Dict[str, str]:
    return {
        "content-type": media_type,
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "Content-Type,Authorization",
        "access-control-allow-methods": "OPTIONS,GET,POST",
    }


def _resp(code: int, body: Dict[str, Any], media_type: str = A2A_MEDIA_TYPE) -> Dict[str, Any]:
    return {
        "statusCode": code,
        "headers": _cors(media_type),
        "body": json.dumps(body, ensure_ascii=False, default=str),
    }


def _method(event: Dict[str, Any]) -> str:
    return (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or ""
    ).upper()


def _path(event: Dict[str, Any]) -> str:
    return (
        event.get("requestContext", {}).get("http", {}).get("path")
        or event.get("rawPath")
        or event.get("path")
        or ""
    )


def _json_body(event: Dict[str, Any]) -> Dict[str, Any]:
    raw = event.get("body")
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return {}


def _load_teams() -> Dict[str, Dict[str, Any]]:
    """Team configs from S3 -- the same objects the orchestrator runs from.

    The card is generated from live config rather than written by hand, so a
    skill cannot advertise an agent that no longer exists. An unreadable
    bucket yields an empty skill list rather than a 500: a card with no skills
    is a true statement about what this deploy can prove it offers.
    """
    bucket = os.environ.get("CONFIG_BUCKET", "")
    prefix = (os.environ.get("CONFIG_PREFIX") or os.environ.get("TEAM_CONFIG_PREFIX") or "teams").strip("/")
    if not bucket:
        return {}
    teams: Dict[str, Dict[str, Any]] = {}
    try:
        token = None
        while True:
            kwargs = {"Bucket": bucket, "Prefix": f"{prefix}/"}
            if token:
                kwargs["ContinuationToken"] = token
            page = s3.list_objects_v2(**kwargs)
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/team.json"):
                    continue
                try:
                    teams[key] = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
                except (ClientError, ValueError) as exc:
                    log.warning("a2a_team_unreadable", extra={"key": key, "err": str(exc)[:200]})
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
    except ClientError as exc:
        log.warning("a2a_teams_unreadable", extra={"err": str(exc)[:200]})
    return teams


def _base_url(event: Dict[str, Any]) -> str:
    configured = (os.environ.get("A2A_BASE_URL") or "").strip().rstrip("/")
    if configured:
        return configured
    domain = event.get("requestContext", {}).get("domainName") or ""
    stage = event.get("requestContext", {}).get("stage") or ""
    if not domain:
        return ""
    # A stage-prefixed REST API serves every route under /{stage}; omitting it
    # would publish interface URLs that 404.
    suffix = f"/{stage}" if stage and stage != "$default" else ""
    return f"https://{domain}{suffix}"


def agent_card(event: Dict[str, Any]) -> Dict[str, Any]:
    return a2a.build_agent_card(
        base_url=_base_url(event),
        version=os.environ.get("A2A_AGENT_VERSION", "1.0.0"),
        teams=_load_teams(),
        documentation_url=os.environ.get("A2A_DOCUMENTATION_URL", ""),
        provider={"organization": os.environ.get("A2A_PROVIDER", "TeamWeave")},
    )


def _send_message(event: Dict[str, Any]) -> Dict[str, Any]:
    body = _json_body(event)
    message = body.get("message") or {}
    prompt = a2a.text_of_message(message)
    if not prompt:
        return _resp(400, {"error": {"code": "INVALID_ARGUMENT",
                                     "message": "message.parts carried no text"}})

    state_machine_arn = os.environ.get("STATE_MACHINE_ARN")
    if not state_machine_arn:
        return _resp(500, {"error": {"code": "INTERNAL", "message": "STATE_MACHINE_ARN is not configured"}})

    run_id = run_ids.new_run_id()
    # `skillId` selects the agent; without it the team's own workflow decides,
    # which is the native behaviour of POST /team/task.
    payload = {
        "operation": "team_task",
        "run_id": run_id,
        "brief": prompt,
        "skill_id": str(body.get("skillId") or "").strip(),
        "a2a": True,
    }
    try:
        # Named with the task id for the same reason the trigger names its
        # execution: `_get_task` resolves the id straight back to an ARN.
        sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=run_id,
            input=json.dumps(payload),
        )
    except ClientError as exc:
        log.error("a2a_start_failed", extra={"err": str(exc)[:300]})
        return _resp(502, {"error": {"code": "UNAVAILABLE",
                                     "message": exc.response.get("Error", {}).get("Message", str(exc))}})

    # Non-blocking, as the card declares. A2A's default is to block until the
    # task is terminal; a TeamWeave pipeline outlives any API Gateway request,
    # so returning a working task the caller polls is the honest answer rather
    # than a request that times out and loses the run id.
    log.info("a2a_message_sent", extra={"run_id": run_id, "skill_id": payload["skill_id"]})
    return _resp(200, {"task": a2a.task_for_run(run_id, "SUBMITTED")})


def _get_task(event: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    from .status_handler import _to_execution_arn  # same id->ARN rule, one definition

    try:
        execution_arn = _to_execution_arn(task_id)
    except ValueError as exc:
        return _resp(400, {"error": {"code": "INVALID_ARGUMENT", "message": str(exc)}})

    try:
        desc = sfn.describe_execution(executionArn=execution_arn)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ExecutionDoesNotExist":
            return _resp(404, {"error": {"code": "NOT_FOUND", "message": "task not found"}})
        return _resp(500, {"error": {"code": "INTERNAL",
                                     "message": exc.response.get("Error", {}).get("Message", str(exc))}})

    status = desc.get("status", "UNKNOWN")
    result = None
    if status == "SUCCEEDED":
        try:
            result = json.loads(desc.get("output") or "null")
        except ValueError:
            result = desc.get("output")
    return _resp(200, {"task": a2a.task_for_run(
        task_id, status, result=result, error=desc.get("cause") or "")})


def handler(event, context):
    method = _method(event)
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": _cors(), "body": ""}

    path = _path(event)

    if path.endswith(a2a.AGENT_CARD_PATH):
        if method != "GET":
            return _resp(405, {"error": {"code": "UNIMPLEMENTED", "message": "GET only"}})
        # Plain JSON: a well-known document is fetched by tooling that does not
        # negotiate on A2A's own media type.
        return _resp(200, agent_card(event), media_type="application/json")

    if path.endswith("/message:send"):
        if method != "POST":
            return _resp(405, {"error": {"code": "UNIMPLEMENTED", "message": "POST only"}})
        return _send_message(event)

    if "/tasks/" in path:
        if method != "GET":
            return _resp(405, {"error": {"code": "UNIMPLEMENTED", "message": "GET only"}})
        task_id = (event.get("pathParameters") or {}).get("task_id") or path.rsplit("/", 1)[-1]
        return _get_task(event, task_id)

    return _resp(404, {"error": {"code": "NOT_FOUND", "message": f"no A2A route for {path}"}})
