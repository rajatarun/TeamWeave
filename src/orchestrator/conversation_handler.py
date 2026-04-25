import json
from typing import Any, Dict

from botocore.exceptions import ClientError

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
    session_id = (body.get("session_id") or "").strip()
    message = (body.get("message") or "").strip()

    missing = [f for f, v in [("agent_id", agent_id), ("alias_id", alias_id), ("session_id", session_id), ("message", message)] if not v]
    if missing:
        return _resp(400, {"error": f"Missing required fields: {', '.join(missing)}"})

    log.info(
        "agent_converse_request",
        extra={"agent_id": agent_id, "alias_id": alias_id, "session_id": session_id},
    )

    try:
        response_text = invoke_agent(agent_id, alias_id, session_id, message)
    except StepFailed as exc:
        log.error("agent_converse_failed", extra={"error": str(exc)})
        return _resp(502, {"error": str(exc)})
    except ClientError as exc:
        log.error("agent_converse_client_error", extra={"error": str(exc)})
        return _resp(502, {"error": exc.response.get("Error", {}).get("Message", str(exc))})

    log.info(
        "agent_converse_response",
        extra={"agent_id": agent_id, "alias_id": alias_id, "session_id": session_id, "response_length": len(response_text)},
    )

    return _resp(200, {
        "agent_id": agent_id,
        "alias_id": alias_id,
        "session_id": session_id,
        "response": response_text,
    })
