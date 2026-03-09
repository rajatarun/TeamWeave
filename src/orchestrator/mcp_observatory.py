"""MCP Observatory integration for Bedrock wrapper observability.

Wraps Bedrock agent and model invocations with the mcp-observatory
InvocationWrapperAPI, recording per-call telemetry (trace ID, token
estimates, cost, latency, policy decision) via structured log records
and persisting each span as an item in the ObservatoryMetrics DynamoDB
table (pk=OBSERVATORY#{operation}, sk={iso_timestamp}#{trace_id}).
Items expire automatically via a 90-day TTL attribute.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional

import boto3
from mcp_observatory.instrument import instrument_wrapper_api

from .bedrock_wrappers import invoke_agent_request, invoke_model_request
from .logger import get_logger

log = get_logger("mcp_observatory")

# One shared wrapper for all Bedrock calls made by this service.
_wrapper = instrument_wrapper_api("teamweave-bedrock")

# Lazy DynamoDB table resource; initialised on first write.
_ddb_table = None

_TTL_SECONDS = 90 * 24 * 60 * 60  # 90 days


def _get_ddb_table():
    """Return the DynamoDB Table resource, creating it once per process."""
    global _ddb_table
    table_name = os.environ.get("OBSERVATORY_METRICS_TABLE")
    if not table_name:
        return None
    if _ddb_table is None:
        _ddb_table = boto3.resource("dynamodb").Table(table_name)
    return _ddb_table


def _to_decimal(value: float) -> Decimal:
    """Convert a float to Decimal for DynamoDB, rounding to 8 d.p."""
    try:
        return Decimal(str(round(value, 8)))
    except InvalidOperation:
        return Decimal("0")


def _push_metric(operation: str, span, decision, extra: dict) -> None:
    """Best-effort write of a telemetry span to ObservatoryMetricsTable.

    Failures are logged as warnings and never propagate to the caller.
    """
    table = _get_ddb_table()
    if table is None:
        return

    try:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
        expiry = int(time.time()) + _TTL_SECONDS

        item: dict = {
            "pk": f"OBSERVATORY#{operation}",
            "sk": f"{now_iso}#{span.trace_id}",
            "trace_id": span.trace_id,
            "operation": operation,
            "timestamp": now_iso,
            "prompt_tokens": Decimal(span.prompt_tokens),
            "completion_tokens": Decimal(span.completion_tokens),
            "cost_usd": _to_decimal(span.cost_usd),
            "decision": decision.action,
            "decision_reason": decision.reason or "none",
            "ttl": Decimal(expiry),
        }
        item.update({k: str(v) if isinstance(v, float) else v for k, v in extra.items()})

        table.put_item(Item=item)
    except Exception as exc:  # noqa: BLE001
        log.warning("observatory_metric_write_failed", extra={"err": str(exc)})


def observe_agent_request(
    runtime_client,
    *,
    agent_id: str,
    alias_id: str,
    session_id: str,
    input_text: str,
) -> dict:
    """Invoke a Bedrock agent through the mcp-observatory wrapper.

    Delegates to ``invoke_agent_request`` while recording a telemetry span
    (trace_id, token estimates, cost_usd, latency, policy decision),
    emitting a structured log record, and persisting the span to DynamoDB.
    """
    result = asyncio.run(
        _wrapper.invoke(
            source="agent",
            model="bedrock-agent",
            prompt=input_text,
            input_payload={
                "agent_id": agent_id,
                "alias_id": alias_id,
                "session_id": session_id,
            },
            call=lambda: invoke_agent_request(
                runtime_client,
                agent_id=agent_id,
                alias_id=alias_id,
                session_id=session_id,
                input_text=input_text,
            ),
        )
    )

    log.info(
        "mcp_observatory",
        extra={
            "operation": "invoke_agent",
            "agent_id": agent_id,
            "alias_id": alias_id,
            "session_id": session_id,
            "input_len": len(input_text),
            "trace_id": result.span.trace_id,
            "prompt_tokens": result.span.prompt_tokens,
            "completion_tokens": result.span.completion_tokens,
            "cost_usd": result.span.cost_usd,
            "decision": result.decision.action,
            "decision_reason": result.decision.reason,
        },
    )

    _push_metric(
        "invoke_agent",
        result.span,
        result.decision,
        {
            "agent_id": agent_id,
            "alias_id": alias_id,
            "session_id": session_id,
            "input_len": Decimal(len(input_text)),
        },
    )

    return result.output


def observe_model_request(
    runtime_client,
    *,
    model_id: str,
    body: str,
    content_type: Optional[str] = None,
    accept: Optional[str] = None,
) -> dict:
    """Invoke a Bedrock model through the mcp-observatory wrapper.

    Delegates to ``invoke_model_request`` while recording a telemetry span
    and emitting a single structured log record and DynamoDB item per call.
    """
    result = asyncio.run(
        _wrapper.invoke(
            source="model",
            model=model_id,
            prompt=body,
            input_payload={
                "model_id": model_id,
                "content_type": content_type,
                "accept": accept,
            },
            call=lambda: invoke_model_request(
                runtime_client,
                model_id=model_id,
                body=body,
                content_type=content_type,
                accept=accept,
            ),
        )
    )

    log.info(
        "mcp_observatory",
        extra={
            "operation": "invoke_model",
            "model_id": model_id,
            "body_len": len(body),
            "trace_id": result.span.trace_id,
            "prompt_tokens": result.span.prompt_tokens,
            "completion_tokens": result.span.completion_tokens,
            "cost_usd": result.span.cost_usd,
            "decision": result.decision.action,
            "decision_reason": result.decision.reason,
        },
    )

    _push_metric(
        "invoke_model",
        result.span,
        result.decision,
        {
            "model_id": model_id,
            "body_len": Decimal(len(body)),
        },
    )

    return result.output
