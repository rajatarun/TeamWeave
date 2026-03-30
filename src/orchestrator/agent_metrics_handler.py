"""Agent Observability Metrics Dashboard Lambda handler.

Serves ``GET /observability/agent-metrics`` by querying the ObservatoryMetricsTable
DynamoDB table that stores mcp-observatory telemetry spans for every Bedrock
agent/model invocation.

Query parameters
----------------
operation    (optional) -- ``invoke_agent``, ``invoke_model``, or ``all`` (default: ``all``)
agent_id     (optional) -- filter by Bedrock agent ID (uses AgentIdTimestampIndex GSI)
model_id     (optional) -- filter by model ID (FilterExpression on query results)
decision     (optional) -- filter by policy decision action (FilterExpression)
start        (optional) -- ISO 8601 or Unix epoch; lower bound on timestamp
end          (optional) -- ISO 8601 or Unix epoch; upper bound on timestamp
sort_by      (optional) -- ``timestamp`` (default), ``cost_usd``, ``prompt_tokens``,
                           ``completion_tokens``
sort_order   (optional) -- ``desc`` (default) or ``asc``
limit        (optional) -- 1-1000, default 100 (list mode only)
next_token   (optional) -- base64-encoded DynamoDB LastEvaluatedKey (list mode only)
aggregate    (optional) -- ``none`` (default), ``by_agent``, ``by_model``,
                           ``by_operation``, ``by_decision``, ``by_hour``, ``by_day``

Responses
---------
200  List mode:      {"items": [...], "count": N, "scanned_count": N, "next_token": "..."}
200  Aggregate mode: {"aggregate": "...", "groups": [...], "total_count": N, "scanned_count": N}
400  {"error": "..."}
500  {"error": "..."}
"""

from __future__ import annotations

import base64
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import boto3
from boto3.dynamodb.conditions import Attr, Key

from .logger import get_logger

log = get_logger("agent_metrics_handler")

_VALID_OPERATIONS = {"invoke_agent", "invoke_model", "classify_question", "synthesize_answer", "all"}

# All known operation PK suffixes stored under OBSERVATORY#{op}.
# classify_question / synthesize_answer were used by earlier versions of
# mcp_observatory before the schema was unified to invoke_model.
_ALL_OPERATION_PKS = ["invoke_agent", "invoke_model", "classify_question", "synthesize_answer"]
_VALID_SORT_BY = {
    "timestamp", "cost_usd", "prompt_tokens", "completion_tokens",
    "composite_risk_score", "hallucination_risk_score", "retries", "grounding_score",
}
_VALID_AGGREGATES = {
    "none", "by_agent", "by_model", "by_operation", "by_decision", "by_hour", "by_day",
    "by_risk_tier", "by_composite_risk_level", "by_hallucination_risk_level", "by_policy_decision",
}
_DEFAULT_LIMIT = 100
_MAX_LIMIT = 1000
_AGGREGATE_SCAN_LIMIT = 5000  # max items scanned per aggregate request

_CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key",
    "Access-Control-Allow-Methods": "GET,OPTIONS",
}

# Lazy DynamoDB table resource
_ddb_table = None


def _get_table():
    global _ddb_table
    table_name = os.environ.get("OBSERVATORY_METRICS_TABLE")
    if not table_name:
        return None
    if _ddb_table is None:
        _ddb_table = boto3.resource("dynamodb").Table(table_name)
    return _ddb_table


def _resp(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": _CORS_HEADERS,
        "body": json.dumps(body, default=_json_default),
    }


def _json_default(obj):
    """JSON serializer for Decimal (DynamoDB returns Decimal for numbers)."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _parse_timestamp(value: str) -> str:
    """Normalise a Unix epoch or ISO 8601 string to ISO 8601 format.

    The DynamoDB SK prefix is ``{iso_timestamp}#{trace_id}`` so range queries
    use ISO string comparison.
    """
    try:
        epoch = float(value)
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        pass
    # Assume already ISO 8601 — return as-is (strip trailing Z if present)
    return value.replace("Z", "+00:00").replace("+00:00", "")


def _decode_next_token(token: str) -> Optional[dict]:
    try:
        return json.loads(base64.b64decode(token.encode()).decode())
    except Exception:
        return None


def _encode_next_token(last_key: dict) -> str:
    return base64.b64encode(json.dumps(last_key, default=_json_default).encode()).decode()


def _parse_bool_param(value: Optional[str]) -> Optional[bool]:
    """Parse a query-string boolean param ('true'/'false') to Python bool or None."""
    if value is None:
        return None
    return value.lower() == "true"


def _build_filter_expression(
    model_id: Optional[str],
    decision: Optional[str],
    risk_tier: Optional[str] = None,
    policy_decision: Optional[str] = None,
    composite_risk_level: Optional[str] = None,
    hallucination_risk_level: Optional[str] = None,
    is_shadow: Optional[bool] = None,
    gate_blocked: Optional[bool] = None,
    fallback_used: Optional[bool] = None,
):
    """Build a FilterExpression for non-key attribute filters."""
    expr = None

    def _and(cond):
        nonlocal expr
        expr = cond if expr is None else expr & cond

    if model_id:
        _and(Attr("model_id").eq(model_id))
    if decision:
        _and(Attr("decision").eq(decision))
    if risk_tier:
        _and(Attr("risk_tier").eq(risk_tier))
    if policy_decision:
        _and(Attr("policy_decision").eq(policy_decision))
    if composite_risk_level:
        _and(Attr("composite_risk_level").eq(composite_risk_level))
    if hallucination_risk_level:
        _and(Attr("hallucination_risk_level").eq(hallucination_risk_level))
    if is_shadow is not None:
        _and(Attr("is_shadow").eq(is_shadow))
    if gate_blocked is not None:
        _and(Attr("gate_blocked").eq(gate_blocked))
    if fallback_used is not None:
        _and(Attr("fallback_used").eq(fallback_used))

    return expr


def _query_by_pk(
    table,
    pk_value: str,
    start_iso: Optional[str],
    end_iso: Optional[str],
    filter_expr,
    limit: int,
    exclusive_start_key: Optional[dict],
) -> tuple[list[dict], int, Optional[dict]]:
    """Query by primary key (pk=OBSERVATORY#{operation}) with optional SK range."""
    key_cond = Key("pk").eq(pk_value)
    if start_iso and end_iso:
        key_cond = key_cond & Key("sk").between(start_iso, end_iso + "~")
    elif start_iso:
        key_cond = key_cond & Key("sk").gte(start_iso)
    elif end_iso:
        key_cond = key_cond & Key("sk").lte(end_iso + "~")

    kwargs: dict[str, Any] = {
        "KeyConditionExpression": key_cond,
        "Limit": limit,
        "ScanIndexForward": True,  # ascending by SK (timestamp); caller re-sorts
    }
    if filter_expr is not None:
        kwargs["FilterExpression"] = filter_expr
    if exclusive_start_key:
        kwargs["ExclusiveStartKey"] = exclusive_start_key

    resp = table.query(**kwargs)
    return resp.get("Items", []), resp.get("ScannedCount", 0), resp.get("LastEvaluatedKey")


def _query_by_agent_id(
    table,
    agent_id: str,
    start_iso: Optional[str],
    end_iso: Optional[str],
    filter_expr,
    limit: int,
    exclusive_start_key: Optional[dict],
) -> tuple[list[dict], int, Optional[dict]]:
    """Query the AgentIdTimestampIndex GSI by agent_id."""
    key_cond = Key("agent_id").eq(agent_id)
    if start_iso and end_iso:
        key_cond = key_cond & Key("timestamp").between(start_iso, end_iso + "~")
    elif start_iso:
        key_cond = key_cond & Key("timestamp").gte(start_iso)
    elif end_iso:
        key_cond = key_cond & Key("timestamp").lte(end_iso + "~")

    kwargs: dict[str, Any] = {
        "IndexName": "AgentIdTimestampIndex",
        "KeyConditionExpression": key_cond,
        "Limit": limit,
        "ScanIndexForward": True,
    }
    if filter_expr is not None:
        kwargs["FilterExpression"] = filter_expr
    if exclusive_start_key:
        kwargs["ExclusiveStartKey"] = exclusive_start_key

    resp = table.query(**kwargs)
    return resp.get("Items", []), resp.get("ScannedCount", 0), resp.get("LastEvaluatedKey")


def _unwrap_ddb_value(value: Any) -> Any:
    """Unwrap low-level DynamoDB AttributeValue maps to plain Python values."""
    if not isinstance(value, dict) or len(value) != 1:
        return value

    attr_type, attr_val = next(iter(value.items()))
    if attr_type == "S":
        return attr_val
    if attr_type == "N":
        return Decimal(str(attr_val))
    if attr_type == "BOOL":
        return bool(attr_val)
    if attr_type == "NULL":
        return None
    if attr_type == "M" and isinstance(attr_val, dict):
        return {k: _unwrap_ddb_value(v) for k, v in attr_val.items()}
    if attr_type == "L" and isinstance(attr_val, list):
        return [_unwrap_ddb_value(v) for v in attr_val]
    return value


def _normalize_item(item: dict) -> dict:
    """Normalize DynamoDB item shape (resource format or low-level AttributeValue format)."""
    return {k: _unwrap_ddb_value(v) for k, v in item.items()}


def _fetch_all_for_aggregate(
    table,
    operation: str,
    agent_id: Optional[str],
    start_iso: Optional[str],
    end_iso: Optional[str],
    filter_expr,
) -> tuple[list[dict], int]:
    """Fetch all matching items for in-memory aggregation (no pagination)."""
    all_items: list[dict] = []
    total_scanned = 0

    if agent_id:
        # Use GSI
        last_key = None
        while True:
            items, scanned, last_key = _query_by_agent_id(
                table, agent_id, start_iso, end_iso, filter_expr,
                limit=_AGGREGATE_SCAN_LIMIT, exclusive_start_key=last_key
            )
            all_items.extend(items)
            total_scanned += scanned
            if not last_key or len(all_items) >= _AGGREGATE_SCAN_LIMIT:
                break
    else:
        ops = _ALL_OPERATION_PKS if operation == "all" else [operation]
        for op in ops:
            pk = f"OBSERVATORY#{op}"
            last_key = None
            while True:
                items, scanned, last_key = _query_by_pk(
                    table, pk, start_iso, end_iso, filter_expr,
                    limit=_AGGREGATE_SCAN_LIMIT, exclusive_start_key=last_key
                )
                all_items.extend(items)
                total_scanned += scanned
                if not last_key or len(all_items) >= _AGGREGATE_SCAN_LIMIT:
                    break

    return [_normalize_item(item) for item in all_items], total_scanned


def _aggregate_items(items: list[dict], mode: str) -> list[dict]:
    """Group items by the requested dimension and compute aggregates."""
    _NUMERIC_FIELDS = [
        "prompt_tokens", "completion_tokens", "cost_usd",
        "shadow_disagreement_score", "shadow_numeric_variance",
        "retries", "prompt_size_chars", "exec_token_ttl_ms",
        "confidence", "grounding_score", "verifier_score",
        "self_consistency_score", "numeric_variance_score",
        "hallucination_risk_score", "grounding_risk", "self_consistency_risk",
        "numeric_instability_risk", "tool_mismatch_risk", "drift_risk",
        "composite_risk_score",
    ]

    def _operation_for_item(item: dict) -> str:
        op = item.get("operation")
        if op:
            return str(op)
        pk = item.get("pk", "")
        if isinstance(pk, str) and pk.startswith("OBSERVATORY#"):
            return pk.split("#", 1)[1]
        return ""

    def _key_for(item: dict) -> tuple:
        if mode == "by_agent":
            return (item.get("agent_id", ""),)
        if mode == "by_model":
            return (item.get("model_id", ""),)
        if mode == "by_operation":
            return (_operation_for_item(item),)
        if mode == "by_decision":
            return (item.get("decision", ""),)
        if mode == "by_hour":
            ts = item.get("timestamp", "")
            return (ts[:13],)  # "2024-01-15T10"
        if mode == "by_day":
            ts = item.get("timestamp", "")
            return (ts[:10],)  # "2024-01-15"
        if mode == "by_risk_tier":
            return (item.get("risk_tier", ""),)
        if mode == "by_composite_risk_level":
            return (item.get("composite_risk_level", ""),)
        if mode == "by_hallucination_risk_level":
            return (item.get("hallucination_risk_level", ""),)
        if mode == "by_policy_decision":
            return (item.get("policy_decision", ""),)
        return ("",)

    def _key_dict(key_tuple: tuple) -> dict:
        if mode == "by_agent":
            return {"agent_id": key_tuple[0]}
        if mode == "by_model":
            return {"model_id": key_tuple[0]}
        if mode == "by_operation":
            return {"operation": key_tuple[0]}
        if mode == "by_decision":
            return {"decision": key_tuple[0]}
        if mode == "by_hour":
            return {"hour": key_tuple[0]}
        if mode == "by_day":
            return {"day": key_tuple[0]}
        if mode == "by_risk_tier":
            return {"risk_tier": key_tuple[0]}
        if mode == "by_composite_risk_level":
            return {"composite_risk_level": key_tuple[0]}
        if mode == "by_hallucination_risk_level":
            return {"hallucination_risk_level": key_tuple[0]}
        if mode == "by_policy_decision":
            return {"policy_decision": key_tuple[0]}
        return {}

    buckets: dict[tuple, dict] = defaultdict(lambda: {
        "count": 0,
        "sums": defaultdict(float),
        "mins": {},
        "maxs": {},
    })

    for item in items:
        k = _key_for(item)
        b = buckets[k]
        b["count"] += 1
        for field in _NUMERIC_FIELDS:
            raw = item.get(field)
            if raw is None:
                continue
            val = float(raw)
            b["sums"][field] += val
            if field not in b["mins"] or val < b["mins"][field]:
                b["mins"][field] = val
            if field not in b["maxs"] or val > b["maxs"][field]:
                b["maxs"][field] = val

    groups = []
    for key_tuple, b in sorted(buckets.items()):
        grp: dict = {"key": _key_dict(key_tuple), "count": b["count"]}
        for field in _NUMERIC_FIELDS:
            if field in b["sums"]:
                total = b["sums"][field]
                grp[f"sum_{field}"] = round(total, 8)
                grp[f"avg_{field}"] = round(total / b["count"], 8)
                grp[f"min_{field}"] = round(b["mins"][field], 8)
                grp[f"max_{field}"] = round(b["maxs"][field], 8)
        groups.append(grp)

    return groups


def _sort_items(items: list[dict], sort_by: str, sort_order: str) -> list[dict]:
    reverse = sort_order == "desc"

    def _key(item: dict):
        val = item.get(sort_by)
        if val is None:
            return (1, 0)  # push None values to the end
        return (0, float(val) if isinstance(val, (Decimal, float, int)) else val)

    return sorted(items, key=_key, reverse=reverse)


def handler(event: dict, context: object) -> dict:  # noqa: C901
    table = _get_table()
    if table is None:
        return _resp(500, {"error": "OBSERVATORY_METRICS_TABLE environment variable not set"})

    params: dict[str, str] = event.get("queryStringParameters") or {}

    # --- Parameter parsing & validation ---
    operation = params.get("operation", "all").lower()
    if operation not in _VALID_OPERATIONS:
        return _resp(400, {"error": f"operation must be one of {sorted(_VALID_OPERATIONS)}"})

    aggregate = params.get("aggregate", "none").lower()
    if aggregate not in _VALID_AGGREGATES:
        return _resp(400, {"error": f"aggregate must be one of {sorted(_VALID_AGGREGATES)}"})

    sort_by = params.get("sort_by", "timestamp").lower()
    if sort_by not in _VALID_SORT_BY:
        return _resp(400, {"error": f"sort_by must be one of {sorted(_VALID_SORT_BY)}"})

    sort_order = params.get("sort_order", "desc").lower()
    if sort_order not in {"asc", "desc"}:
        return _resp(400, {"error": "sort_order must be 'asc' or 'desc'"})

    try:
        limit = int(params.get("limit", _DEFAULT_LIMIT))
    except ValueError:
        return _resp(400, {"error": "limit must be an integer"})
    limit = max(1, min(limit, _MAX_LIMIT))

    agent_id = params.get("agent_id") or None
    model_id = params.get("model_id") or None
    decision = params.get("decision") or None
    risk_tier = params.get("risk_tier") or None
    policy_decision = params.get("policy_decision") or None
    composite_risk_level = params.get("composite_risk_level") or None
    hallucination_risk_level = params.get("hallucination_risk_level") or None
    is_shadow = _parse_bool_param(params.get("is_shadow"))
    gate_blocked = _parse_bool_param(params.get("gate_blocked"))
    fallback_used = _parse_bool_param(params.get("fallback_used"))
    next_token_raw = params.get("next_token") or None

    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    if params.get("start"):
        try:
            start_iso = _parse_timestamp(params["start"])
        except Exception:
            return _resp(400, {"error": "start must be a Unix epoch or ISO 8601 timestamp"})
    if params.get("end"):
        try:
            end_iso = _parse_timestamp(params["end"])
        except Exception:
            return _resp(400, {"error": "end must be a Unix epoch or ISO 8601 timestamp"})

    filter_expr = _build_filter_expression(
        model_id, decision, risk_tier, policy_decision,
        composite_risk_level, hallucination_risk_level,
        is_shadow, gate_blocked, fallback_used,
    )

    # --- Aggregate mode: fetch all, group, return ---
    if aggregate != "none":
        try:
            items, scanned = _fetch_all_for_aggregate(
                table, operation, agent_id, start_iso, end_iso, filter_expr
            )
        except Exception as exc:
            log.error("agent_metrics_aggregate_error", extra={"err": str(exc)})
            return _resp(500, {"error": "Failed to query metrics"})

        groups = _aggregate_items(items, aggregate)
        return _resp(200, {
            "aggregate": aggregate,
            "groups": groups,
            "total_count": len(items),
            "scanned_count": scanned,
        })

    # --- List mode: query, sort, paginate ---
    exclusive_start_key = _decode_next_token(next_token_raw) if next_token_raw else None

    items: list[dict] = []
    scanned = 0
    last_key: Optional[dict] = None

    try:
        if agent_id:
            items, scanned, last_key = _query_by_agent_id(
                table, agent_id, start_iso, end_iso, filter_expr,
                limit=limit, exclusive_start_key=exclusive_start_key
            )
            items = [_normalize_item(item) for item in items]
            # Apply operation filter if specified
            if operation != "all":
                items = [
                    i for i in items
                    if (
                        i.get("operation") == operation
                        or i.get("pk") == f"OBSERVATORY#{operation}"
                    )
                ]
        elif operation == "all":
            # Query all known operation PKs and merge results.
            # Pagination is not supported for merged queries.
            for op in _ALL_OPERATION_PKS:
                op_items, op_scanned, _ = _query_by_pk(
                    table, f"OBSERVATORY#{op}", start_iso, end_iso, filter_expr,
                    limit=limit, exclusive_start_key=None
                )
                items.extend(_normalize_item(item) for item in op_items)
                scanned += op_scanned
            last_key = None  # merged queries; pagination not supported for all+merged
        else:
            pk = f"OBSERVATORY#{operation}"
            items, scanned, last_key = _query_by_pk(
                table, pk, start_iso, end_iso, filter_expr,
                limit=limit, exclusive_start_key=exclusive_start_key
            )
            items = [_normalize_item(item) for item in items]
    except Exception as exc:
        log.error("agent_metrics_query_error", extra={"err": str(exc)})
        return _resp(500, {"error": "Failed to query metrics"})

    # Sort
    items = _sort_items(items, sort_by, sort_order)[:limit]

    response_body: dict = {
        "items": items,
        "count": len(items),
        "scanned_count": scanned,
    }
    if last_key:
        response_body["next_token"] = _encode_next_token(last_key)

    return _resp(200, response_body)
