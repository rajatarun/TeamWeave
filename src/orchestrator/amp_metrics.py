"""Amazon Managed Prometheus (AMP) remote_write integration.

Pushes Prometheus metrics from Bedrock invocation telemetry directly to an
AMP workspace via the Prometheus remote_write protocol (protobuf + snappy)
signed with SigV4 (service=``aps``).

Configuration
-------------
Set the following environment variables:

  AMP_WORKSPACE_ID  -- AMP workspace ID (required; push is skipped if absent)
  AMP_REGION        -- AWS region for the AMP workspace (default: us-east-1)

Public API
----------
  record_agent_span(span, decision, extra)  -- invoke_agent telemetry
  record_model_span(span, decision, extra)  -- invoke_model telemetry

Both functions swallow all exceptions and log a warning on failure so that a
broken AMP endpoint never affects the primary invocation path.
"""

from __future__ import annotations

import os
import struct
import time
from typing import Any

import boto3
import botocore.auth
import botocore.awsrequest
import requests
import snappy

from .logger import get_logger

log = get_logger("amp_metrics")

# ---------------------------------------------------------------------------
# Prometheus metric names
# ---------------------------------------------------------------------------

_M_PROMPT_TOKENS = "teamweave_bedrock_prompt_tokens_total"
_M_COMPLETION_TOKENS = "teamweave_bedrock_completion_tokens_total"
_M_COST_USD = "teamweave_bedrock_cost_usd_total"
_M_REQUESTS = "teamweave_bedrock_requests_total"
_M_INPUT_LEN = "teamweave_bedrock_input_length_chars"
_M_SHADOW_DISAGREEMENT = "teamweave_bedrock_shadow_disagreement_score"
_M_SHADOW_VARIANCE = "teamweave_bedrock_shadow_numeric_variance"

# Type aliases
_LabelList = list[tuple[str, str]]
_Series = list[tuple[_LabelList, float, int]]  # (labels, value, ts_ms)

# ---------------------------------------------------------------------------
# Minimal protobuf encoder for Prometheus WriteRequest
# ---------------------------------------------------------------------------
# Proto schema:
#   message Label       { string name = 1; string value = 2; }
#   message Sample      { double value = 1; int64 timestamp = 2; }
#   message TimeSeries  { repeated Label labels = 1; repeated Sample samples = 2; }
#   message WriteRequest{ repeated TimeSeries timeseries = 1; }
#
# Wire types: 0=varint  1=64-bit  2=length-delimited
# Tag = (field_number << 3) | wire_type


def _encode_varint(value: int) -> bytes:
    """Encode a non-negative integer as a protobuf base-128 varint."""
    result: list[int] = []
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            result.append(bits | 0x80)
        else:
            result.append(bits)
            break
    return bytes(result)


def _ld(data: bytes) -> bytes:
    """Prefix *data* with its varint-encoded length (length-delimited body)."""
    return _encode_varint(len(data)) + data


def _encode_label(name: str, value: str) -> bytes:
    """Return wire bytes for a Label{name, value} message."""
    n, v = name.encode(), value.encode()
    # field 1 (name)  tag=0x0A  field 2 (value) tag=0x12
    return b"\x0a" + _ld(n) + b"\x12" + _ld(v)


def _encode_sample(value: float, ts_ms: int) -> bytes:
    """Return wire bytes for a Sample{value, timestamp} message."""
    # field 1 (double, wire 1) tag=0x09; field 2 (int64 varint, wire 0) tag=0x10
    return b"\x09" + struct.pack("<d", value) + b"\x10" + _encode_varint(ts_ms)


def _encode_timeseries(labels: _LabelList, value: float, ts_ms: int) -> bytes:
    """Return wire bytes for a TimeSeries message.

    Labels are sorted lexicographically by name as required by AMP.
    """
    data = b""
    for lname, lval in sorted(labels, key=lambda x: x[0]):
        lb = _encode_label(lname, lval)
        data += b"\x0a" + _ld(lb)          # field 1: labels (repeated message)
    sp = _encode_sample(value, ts_ms)
    data += b"\x12" + _ld(sp)              # field 2: samples (repeated message)
    return data


def _encode_write_request(series: _Series) -> bytes:
    """Return wire bytes for a WriteRequest (the full remote_write body)."""
    data = b""
    for labels, value, ts_ms in series:
        ts = _encode_timeseries(labels, value, ts_ms)
        data += b"\x0a" + _ld(ts)          # field 1: timeseries (repeated message)
    return data


# ---------------------------------------------------------------------------
# AMP remote_write push
# ---------------------------------------------------------------------------


def push_to_amp(series: _Series) -> None:
    """Encode *series* as protobuf+snappy and POST to AMP remote_write.

    No-op when ``AMP_WORKSPACE_ID`` is not configured.
    Raises on HTTP errors (caller is responsible for swallowing).
    """
    workspace_id = os.environ.get("AMP_WORKSPACE_ID")
    if not workspace_id:
        return

    region = os.environ.get("AMP_REGION", "us-east-1")
    url = (
        f"https://aps-workspaces.{region}.amazonaws.com"
        f"/workspaces/{workspace_id}/api/v1/remote_write"
    )

    proto_bytes = _encode_write_request(series)
    compressed = snappy.compress(proto_bytes)

    headers: dict[str, str] = {
        "Content-Type": "application/x-protobuf",
        "Content-Encoding": "snappy",
        "X-Prometheus-Remote-Write-Version": "0.1.0",
    }

    session = boto3.session.Session()
    creds = session.get_credentials().get_frozen_credentials()
    aws_req = botocore.awsrequest.AWSRequest(
        method="POST", url=url, data=compressed, headers=headers
    )
    botocore.auth.SigV4Auth(creds, "aps", region).add_auth(aws_req)

    resp = requests.post(
        url, data=compressed, headers=dict(aws_req.headers), timeout=5
    )
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Internal label helpers
# ---------------------------------------------------------------------------


def _make_labels(operation: str, extras: dict[str, str]) -> _LabelList:
    """Build a label list with ``operation`` prepended then *extras*."""
    return [("operation", operation)] + [(k, v) for k, v in extras.items()]


# ---------------------------------------------------------------------------
# Public metric recording helpers
# ---------------------------------------------------------------------------


def record_agent_span(span: Any, decision: Any, extra: dict) -> None:
    """Push invoke_agent telemetry to AMP.

    Metrics pushed per invocation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    teamweave_bedrock_prompt_tokens_total       {operation, agent_id, alias_id, trace_id}
    teamweave_bedrock_completion_tokens_total   {operation, agent_id, alias_id, trace_id}
    teamweave_bedrock_cost_usd_total            {operation, agent_id, alias_id, trace_id}
    teamweave_bedrock_input_length_chars        {operation, agent_id, alias_id, trace_id}
    teamweave_bedrock_requests_total            {operation, agent_id, alias_id, trace_id,
                                                 decision, decision_reason}
    teamweave_bedrock_shadow_disagreement_score {operation, agent_id, alias_id, trace_id}
                                                 (only when shadow was invoked)
    teamweave_bedrock_shadow_numeric_variance   {operation, agent_id, alias_id, trace_id}
                                                 (only when shadow was invoked)
    """
    if not os.environ.get("AMP_WORKSPACE_ID"):
        return

    try:
        ts_ms = int(time.time() * 1000)
        agent_id = str(extra.get("agent_id", ""))
        alias_id = str(extra.get("alias_id", ""))
        trace_id = str(getattr(span, "trace_id", ""))

        base_extra = {"agent_id": agent_id, "alias_id": alias_id, "trace_id": trace_id}
        base_labels = _make_labels("invoke_agent", base_extra)

        series: _Series = [
            ([("__name__", _M_PROMPT_TOKENS)] + base_labels, float(span.prompt_tokens), ts_ms),
            ([("__name__", _M_COMPLETION_TOKENS)] + base_labels, float(span.completion_tokens), ts_ms),
            ([("__name__", _M_COST_USD)] + base_labels, float(span.cost_usd), ts_ms),
            ([("__name__", _M_INPUT_LEN)] + base_labels, float(extra.get("input_len", 0)), ts_ms),
            (
                [("__name__", _M_REQUESTS)]
                + _make_labels(
                    "invoke_agent",
                    {
                        **base_extra,
                        "decision": str(decision.action),
                        "decision_reason": str(decision.reason or "none"),
                    },
                ),
                1.0,
                ts_ms,
            ),
        ]

        if span.shadow_disagreement_score is not None:
            series.append(
                ([("__name__", _M_SHADOW_DISAGREEMENT)] + base_labels,
                 float(span.shadow_disagreement_score), ts_ms)
            )

        if span.shadow_numeric_variance is not None:
            series.append(
                ([("__name__", _M_SHADOW_VARIANCE)] + base_labels,
                 float(span.shadow_numeric_variance), ts_ms)
            )

        push_to_amp(series)

    except Exception as exc:  # noqa: BLE001
        log.warning("amp_agent_metric_push_failed", extra={"err": str(exc)})


def record_model_span(span: Any, decision: Any, extra: dict) -> None:
    """Push invoke_model telemetry to AMP.

    Metrics pushed per invocation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    teamweave_bedrock_prompt_tokens_total      {operation, model_id, trace_id}
    teamweave_bedrock_completion_tokens_total  {operation, model_id, trace_id}
    teamweave_bedrock_cost_usd_total           {operation, model_id, trace_id}
    teamweave_bedrock_input_length_chars       {operation, model_id, trace_id}
    teamweave_bedrock_requests_total           {operation, model_id, trace_id,
                                                decision, decision_reason}
    """
    if not os.environ.get("AMP_WORKSPACE_ID"):
        return

    try:
        ts_ms = int(time.time() * 1000)
        model_id = str(extra.get("model_id", ""))
        trace_id = str(getattr(span, "trace_id", ""))

        base_extra = {"model_id": model_id, "trace_id": trace_id}
        base_labels = _make_labels("invoke_model", base_extra)

        series: _Series = [
            ([("__name__", _M_PROMPT_TOKENS)] + base_labels, float(span.prompt_tokens), ts_ms),
            ([("__name__", _M_COMPLETION_TOKENS)] + base_labels, float(span.completion_tokens), ts_ms),
            ([("__name__", _M_COST_USD)] + base_labels, float(span.cost_usd), ts_ms),
            ([("__name__", _M_INPUT_LEN)] + base_labels, float(extra.get("body_len", 0)), ts_ms),
            (
                [("__name__", _M_REQUESTS)]
                + _make_labels(
                    "invoke_model",
                    {
                        **base_extra,
                        "decision": str(decision.action),
                        "decision_reason": str(decision.reason or "none"),
                    },
                ),
                1.0,
                ts_ms,
            ),
        ]

        push_to_amp(series)

    except Exception as exc:  # noqa: BLE001
        log.warning("amp_model_metric_push_failed", extra={"err": str(exc)})
