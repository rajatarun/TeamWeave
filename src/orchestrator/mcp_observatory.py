import time
from dataclasses import dataclass
from typing import Optional

from .bedrock_wrappers import invoke_agent_request, invoke_model_request
from .logger import get_logger

log = get_logger("mcp_observatory")


@dataclass
class ObservedInvocation:
    """Captures observability data for a single Bedrock API call."""

    operation: str
    duration_ms: float
    success: bool
    error: Optional[str] = None


def observe_agent_request(
    runtime_client,
    *,
    agent_id: str,
    alias_id: str,
    session_id: str,
    input_text: str,
) -> dict:
    """Observatory wrapper around invoke_agent_request.

    Delegates to the bedrock wrapper and emits a structured log record with
    timing, success/failure status, and key request identifiers so that every
    agent invocation is observable without changing caller behaviour.
    """
    started = time.monotonic()
    error: Optional[str] = None
    try:
        result = invoke_agent_request(
            runtime_client,
            agent_id=agent_id,
            alias_id=alias_id,
            session_id=session_id,
            input_text=input_text,
        )
        return result
    except Exception as exc:
        error = str(exc)[:240]
        raise
    finally:
        duration_ms = round((time.monotonic() - started) * 1000, 2)
        extra: dict = {
            "operation": "invoke_agent",
            "agent_id": agent_id,
            "alias_id": alias_id,
            "session_id": session_id,
            "input_len": len(input_text),
            "duration_ms": duration_ms,
            "success": error is None,
        }
        if error is not None:
            extra["error"] = error
        log.info("mcp_observatory", extra=extra)


def observe_model_request(
    runtime_client,
    *,
    model_id: str,
    body: str,
    content_type: Optional[str] = None,
    accept: Optional[str] = None,
) -> dict:
    """Observatory wrapper around invoke_model_request.

    Delegates to the bedrock wrapper and emits a structured log record with
    timing, success/failure status, and key request identifiers.
    """
    started = time.monotonic()
    error: Optional[str] = None
    try:
        result = invoke_model_request(
            runtime_client,
            model_id=model_id,
            body=body,
            content_type=content_type,
            accept=accept,
        )
        return result
    except Exception as exc:
        error = str(exc)[:240]
        raise
    finally:
        duration_ms = round((time.monotonic() - started) * 1000, 2)
        extra = {
            "operation": "invoke_model",
            "model_id": model_id,
            "body_len": len(body),
            "duration_ms": duration_ms,
            "success": error is None,
        }
        if error is not None:
            extra["error"] = error
        log.info("mcp_observatory", extra=extra)
