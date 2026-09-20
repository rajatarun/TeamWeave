import json
from typing import Any, Dict

from botocore.exceptions import ClientError

from .agent_runtime import AgentRef, get_runtime
from .bedrock_invoke import invoke_agent
from .logger import get_logger
from .models import StepFailed

log = get_logger("conversation_handler")


def _cors() -> Dict[str, str]:
    return {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "Content-Type,Authorization",
        "access-control-allow-methods": "OPTIONS,POST",
    }


def _resp(code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {"statusCode": code, "headers": _cors(), "body": json.dumps(body, ensure_ascii=False, default=str)}


def _method(event: Dict[str, Any]) -> str:
    return (event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod") or "").upper()


def _json_body(event: Dict[str, Any]) -> Dict[str, Any]:
    b = event.get("body")
    if not b:
        return {}
    try:
        return json.loads(b) if isinstance(b, str) else b
    except Exception:
        return {}


def handler(event, context):
    m = _method(event)

    if m == "OPTIONS":
        return {"statusCode": 200, "headers": _cors(), "body": ""}

    if m != "POST":
        return _resp(405, {"error": "Method not allowed"})

    body = _json_body(event)
    agent_id = (body.get("agent_id") or "").strip()
    alias_id = (body.get("alias_id") or "").strip()
    runtime_arn = (body.get("runtime_arn") or "").strip()
    qualifier = (body.get("qualifier") or "").strip()
    session_id = (body.get("session_id") or "").strip()
    message = (body.get("message") or "").strip()

    # session_id and message are required on every substrate. The agent's
    # coordinates are not: which ones are needed depends on AGENT_RUNTIME, so
    # the runtime is asked rather than Classic's pair being demanded of
    # everyone. Demanding them was what silently took the admin console's chat
    # away the moment the substrate changed -- every AgentCore agent has a
    # blank agentId/aliasId, so every request was a 400 for missing fields
    # that no longer exist.
    missing = [f for f, v in [("session_id", session_id), ("message", message)] if not v]
    if missing:
        return _resp(400, {"error": f"Missing required fields: {', '.join(missing)}"})

    runtime = get_runtime()
    problem = runtime.missing_fields(
        AgentRef(agent_id=agent_id, alias_id=alias_id, runtime_arn=runtime_arn, qualifier=qualifier)
    )
    if problem:
        return _resp(400, {"error": problem, "runtime": runtime.name})

    log.info(
        "agent_converse_request",
        extra={
            "runtime": runtime.name,
            "agent_id": agent_id,
            "alias_id": alias_id,
            "runtime_arn": runtime_arn,
            "session_id": session_id,
        },
    )

    try:
        response_text = invoke_agent(
            agent_id,
            alias_id,
            session_id,
            message,
            runtime_arn=runtime_arn,
            qualifier=qualifier,
        )
    except StepFailed as exc:
        log.error("agent_converse_failed", extra={"error": str(exc)})
        return _resp(502, {"error": str(exc), "runtime": runtime.name})
    except ClientError as exc:
        log.error("agent_converse_client_error", extra={"error": str(exc)})
        return _resp(502, {
            "error": exc.response.get("Error", {}).get("Message", str(exc)),
            "runtime": runtime.name,
        })

    log.info(
        "agent_converse_response",
        extra={
            "runtime": runtime.name,
            "agent_id": agent_id,
            "alias_id": alias_id,
            "session_id": session_id,
            "response_length": len(response_text),
        },
    )

    # `runtime` is in the response so a console can say which substrate
    # answered instead of inferring it from fields that are now often blank.
    return _resp(200, {
        "agent_id": agent_id,
        "alias_id": alias_id,
        "runtime": runtime.name,
        "session_id": session_id,
        "response": response_text,
    })
