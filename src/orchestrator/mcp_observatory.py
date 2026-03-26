"""MCP Observatory integration for Bedrock wrapper observability.

Wraps Bedrock agent and model invocations with the mcp-observatory
InvocationWrapperAPI, recording per-call telemetry (trace ID, token
estimates, cost, latency, policy decision) via structured log records
and persisting each span as an item in the ObservatoryMetrics DynamoDB
table (pk=OBSERVATORY#{operation}, sk={iso_timestamp}#{trace_id}).

When a ``shadow_alias_id`` is supplied to ``observe_agent_request``,
``dual_invoke=True`` is activated: the shadow alias is invoked in
parallel, and shadow telemetry (disagreement score, numeric variance)
is captured and written to DynamoDB alongside the primary span.

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

from . import amp_metrics as _amp
from .bedrock_wrappers import invoke_agent_request, invoke_model_request
from .logger import get_logger

log = get_logger("mcp_observatory")

# One shared wrapper for all Bedrock calls made by this service.
_wrapper = instrument_wrapper_api("teamweave-bedrock")

# Lazy DynamoDB table resource; initialised on first write.
_ddb_table = None

_TTL_SECONDS = 90 * 24 * 60 * 60  # 90 days

# Sentinel for distinguishing "attribute absent" from "attribute is None"
_MISSING = object()

# TraceContext field groups used by _extract_span_fields
_SPAN_FLOAT_FIELDS = (
    "confidence", "grounding_score", "verifier_score", "self_consistency_score",
    "numeric_variance_score", "hallucination_risk_score",
    "grounding_risk", "self_consistency_risk", "numeric_instability_risk",
    "tool_mismatch_risk", "drift_risk", "composite_risk_score",
)
_SPAN_INT_FIELDS = ("retries", "prompt_size_chars", "exec_token_ttl_ms")
_SPAN_BOOL_FIELDS = (
    "fallback_used", "is_shadow", "gate_blocked",
    "tool_claim_mismatch", "exec_token_verified",
)
_SPAN_STR_FIELDS = (
    "span_id", "parent_span_id", "risk_tier", "prompt_template_id", "prompt_hash",
    "normalized_prompt_hash", "answer_hash", "hallucination_risk_level",
    "shadow_parent_trace_id", "fallback_type", "fallback_reason",
    "request_id", "method", "tool_name", "tool_args_hash", "tool_criticality",
    "policy_decision", "policy_id", "policy_version",
    "composite_risk_level", "exec_token_id", "exec_token_hash",
)
_SPAN_DATETIME_FIELDS = ("start_time", "end_time")


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


def _extract_span_fields(span) -> dict:
    """Extract all TraceContext fields from an mcp-observatory span into a DynamoDB-ready dict.

    Only fields present on the span object and not None are included.  Fields
    already written explicitly by ``_push_metric`` (trace_id, prompt_tokens,
    completion_tokens, cost_usd, shadow_disagreement_score,
    shadow_numeric_variance) are intentionally excluded here to avoid
    duplication.
    """
    result: dict = {}

    for field in _SPAN_FLOAT_FIELDS:
        val = getattr(span, field, _MISSING)
        if val is not _MISSING and val is not None:
            result[field] = _to_decimal(val)

    for field in _SPAN_INT_FIELDS:
        val = getattr(span, field, _MISSING)
        if val is not _MISSING and val is not None:
            result[field] = Decimal(int(val))

    for field in _SPAN_BOOL_FIELDS:
        val = getattr(span, field, _MISSING)
        if val is not _MISSING and val is not None:
            result[field] = bool(val)

    for field in _SPAN_STR_FIELDS:
        val = getattr(span, field, _MISSING)
        if val is not _MISSING and val is not None:
            result[field] = str(val)

    for field in _SPAN_DATETIME_FIELDS:
        val = getattr(span, field, _MISSING)
        if val is not _MISSING and val is not None:
            try:
                result[field] = val.isoformat() if hasattr(val, "isoformat") else str(val)
            except Exception:  # noqa: BLE001
                pass

    return result


def _push_metric(operation: str, span, decision, extra: dict) -> None:
    """Best-effort write of a telemetry span to ObservatoryMetricsTable and AMP.

    Includes shadow disagreement and numeric variance when a dual_invoke
    was performed (fields are set on the primary span by mcp_observatory).

    Both the DynamoDB write and the AMP remote_write are best-effort:
    failures are logged as warnings and never propagate to the caller.
    """
    # --- DynamoDB write (guarded by table availability) ---
    table = _get_ddb_table()
    if table is not None:
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

            # Shadow comparison fields (set when dual_invoke=True)
            if span.shadow_disagreement_score is not None:
                item["shadow_disagreement_score"] = _to_decimal(span.shadow_disagreement_score)
            if span.shadow_numeric_variance is not None:
                item["shadow_numeric_variance"] = _to_decimal(span.shadow_numeric_variance)

            # Merge remaining TraceContext fields (don't overwrite already-set keys)
            for k, v in _extract_span_fields(span).items():
                if k not in item:
                    item[k] = v

            item.update({k: str(v) if isinstance(v, float) else v for k, v in extra.items()})

            table.put_item(Item=item)
        except Exception as exc:  # noqa: BLE001
            log.warning("observatory_metric_write_failed", extra={"err": str(exc)})

    # --- AMP remote_write (independent of DynamoDB availability) ---
    if operation == "invoke_agent":
        _amp.record_agent_span(span, decision, extra)
    else:
        _amp.record_model_span(span, decision, extra)


def observe_agent_request(
    runtime_client,
    *,
    agent_id: str,
    alias_id: str,
    session_id: str,
    input_text: str,
    shadow_alias_id: Optional[str] = None,
) -> dict:
    """Invoke a Bedrock agent through the mcp-observatory wrapper.

    When ``shadow_alias_id`` is provided the call is dual-invoked: the
    primary alias and the shadow alias are both executed and telemetry
    (disagreement score, numeric variance) is captured.
    """
    has_shadow = bool(shadow_alias_id)

    shadow_call = None
    if has_shadow:
        _sid = shadow_alias_id  # captured for closure
        shadow_call = lambda: invoke_agent_request(  # noqa: E731
            runtime_client,
            agent_id=agent_id,
            alias_id=_sid,
            session_id=session_id,
            input_text=input_text,
        )

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
            dual_invoke=has_shadow,
            shadow_source="agent" if has_shadow else None,
            shadow_model=f"bedrock-agent/{shadow_alias_id}" if has_shadow else None,
            shadow_prompt=input_text if has_shadow else None,
            shadow_input_payload=(
                {"agent_id": agent_id, "alias_id": shadow_alias_id, "session_id": session_id}
                if has_shadow else None
            ),
            shadow_call=shadow_call,
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
            "shadow_alias_id": shadow_alias_id,
            "shadow_disagreement_score": result.span.shadow_disagreement_score,
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
            **({"shadow_alias_id": shadow_alias_id} if shadow_alias_id else {}),
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
