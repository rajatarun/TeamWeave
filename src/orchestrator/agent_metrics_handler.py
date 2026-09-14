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

Read path
---------
Spans are read through the ``SpanTimelineIndex`` GSI (``span_date`` HASH,
``timestamp`` RANGE), one query per UTC day in the requested range, with
``operation`` applied as a FilterExpression.  Before this, reads enumerated the
partition key (``OBSERVATORY#{operation}``), which meant a writer's rows showed
up here only if it had guessed a prefix this module happened to list -- three
of the shared table's five writers had not, and their telemetry was durable,
billable and permanently invisible.  Nothing about the query now depends on
what any writer chose for its pk.

Two consequences for callers, both deliberate:

* ``start``/``end`` absent no longer means "everything".  The index is queried
  per day, so an open range is an open-ended fan-out; the default window is the
  last 7 days (``_DEFAULT_LOOKBACK_DAYS``).
* Rows written before the contract added ``span_date`` are not in the index and
  will not appear.  They stay reachable by their original pk; backfilling them
  is a separate migration.

While ``SpanTimelineIndex`` does not exist -- it ships in a different stack, and
a new GSI is not queryable until its backfill finishes -- the old pk
enumeration is used instead and a warning is logged once per process.  That
fallback is triggered by index-absence alone (see ``_is_index_missing_error``);
every other failure propagates, because answering a throttled query out of the
legacy partitions would report "no traffic" for most writers.

The OBSERVATORY_METRICS access and aggregation helpers below are public
(``get_table``, ``fetch_all_for_aggregate``, ``aggregate_items``,
``query_by_pk``) along with the response helpers (``json_response``,
``json_default``, ``CORS_HEADERS``) because ``unified_observability_handler``
composes this same data into ``GET /observability``.  There is one source of
truth for the DynamoDB query and aggregation logic: this module.
"""

from __future__ import annotations

import base64
import json
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
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
#
# This list is now only the *legacy* read path (see _legacy_list_query and
# _fetch_all_via_pk below).  It is exactly the enumeration the v2 contract
# exists to abolish: a writer whose rows land under a prefix absent from this
# list is invisible to this dashboard, which is how three of the shared
# table's five writers came to emit telemetry nobody could see.
_ALL_OPERATION_PKS = ["invoke_agent", "invoke_model", "classify_question", "synthesize_answer"]

# ---------------------------------------------------------------------------
# SpanTimelineIndex -- the supported read path (shared contract v2.0.0)
# ---------------------------------------------------------------------------
# Keyed span_date (HASH, "YYYY-MM-DD" UTC) + timestamp (RANGE, ISO 8601 UTC).
# Reading through it removes the pk-prefix agreement entirely: this handler
# queries the days the caller asked about and filters in memory, so it never
# needs to know what partition-key grammar any writer chose.  The names below
# must equal contracts/observatory_metrics_item.json's "gsi" block; a
# conformance test in tests/test_shared_table_contract.py reads them from that
# file and asserts the query really uses them.
SPAN_TIMELINE_INDEX = "SpanTimelineIndex"
SPAN_TIMELINE_PARTITION_KEY = "span_date"
SPAN_TIMELINE_SORT_KEY = "timestamp"

# One query is issued per day in the requested range, so an unbounded range is
# an unbounded number of queries.  With neither `start` nor `end` given the
# window is the last 7 days: enough to cover the weekly rhythm these dashboards
# are read on, small enough that the default request is 8 partition queries
# rather than a fan-out that grows without limit as the table ages.  Callers
# who want more say so with `start`/`end`.
_DEFAULT_LOOKBACK_DAYS = 7

# Upper bound on the day fan-out for an explicit range.  Items carry a 90-day
# TTL (see mcp_observatory._TTL_SECONDS), so days older than that hold nothing
# to find and querying them only buys latency.  The newest _MAX_SPAN_DATES days
# of the requested range are the ones queried.
_MAX_SPAN_DATES = 92

# The timestamp spelling the writer uses, and what _parse_timestamp normalises
# user input to; defaults are generated in the same shape so that range
# comparisons stay lexicographic.
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

# A DynamoDB error means "this index does not exist yet" only in these two
# shapes.  Anything else -- throttling, credentials, a malformed expression --
# must propagate, or the fallback becomes a mechanism for turning real outages
# into quietly wrong dashboards.
_INDEX_ABSENCE_MARKERS = (
    SPAN_TIMELINE_INDEX.lower(),
    "specified index",
    "index not found",
)

# The fallback warns once per process rather than once per query: it is a
# deploy-ordering condition that persists for minutes, and one line per
# dashboard refresh would bury it.
_index_fallback_warned = False
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

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key",
    "Access-Control-Allow-Methods": "GET,OPTIONS",
}

# Lazy DynamoDB table resource
_ddb_table = None


def get_table():
    global _ddb_table
    table_name = os.environ.get("OBSERVATORY_METRICS_TABLE")
    if not table_name:
        return None
    if _ddb_table is None:
        _ddb_table = boto3.resource("dynamodb").Table(table_name)
    return _ddb_table


def json_response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": CORS_HEADERS,
        "body": json.dumps(body, default=json_default),
    }


def json_default(obj):
    """JSON serializer for Decimal (DynamoDB returns Decimal for numbers)."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _parse_timestamp(value: str) -> str:
    """Normalise a Unix epoch or ISO 8601 string to a UTC ISO 8601 string.

    The DynamoDB SK prefix is ``{iso_timestamp}#{trace_id}`` so range queries
    use lexicographic string comparison.  All inputs are normalised to UTC so
    that timezone-offset strings like ``2026-04-28T14:34:37-05:00`` are not
    compared literally against UTC-stored SKs (which would produce wrong
    results because ``"20" < "23"`` lexicographically).
    """
    try:
        epoch = float(value)
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        pass
    # Parse ISO 8601 with any timezone offset (Z, +HH:MM, -HH:MM) and
    # convert to UTC so the result is safe for lexicographic SK comparison.
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        pass
    return value


def _decode_next_token(token: str) -> Optional[dict]:
    try:
        return json.loads(base64.b64decode(token.encode()).decode())
    except Exception:
        return None


def _encode_next_token(last_key: dict) -> str:
    return base64.b64encode(json.dumps(last_key, default=json_default).encode()).decode()


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


def query_by_pk(
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
        "ScanIndexForward": False,  # descending by SK so Limit returns the most recent items
    }
    if filter_expr is not None:
        kwargs["FilterExpression"] = filter_expr
    if exclusive_start_key:
        kwargs["ExclusiveStartKey"] = exclusive_start_key

    resp = table.query(**kwargs)
    return resp.get("Items", []), resp.get("ScannedCount", 0), resp.get("LastEvaluatedKey")


def _resolve_window(start_iso: Optional[str], end_iso: Optional[str]) -> tuple[str, str]:
    """Fill in the ends of the requested time range that the caller left open.

    The index partition is a day, so a query needs both ends to know which days
    to ask for.  An absent `end` means "up to now"; an absent `start` means
    "_DEFAULT_LOOKBACK_DAYS before the end of the window".
    """
    now = datetime.now(timezone.utc)
    resolved_end = end_iso or now.strftime(_TS_FORMAT)
    if start_iso:
        return start_iso, resolved_end
    anchor = now if end_iso is None else _iso_to_datetime(end_iso) or now
    return (anchor - timedelta(days=_DEFAULT_LOOKBACK_DAYS)).strftime(_TS_FORMAT), resolved_end


def _iso_to_datetime(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def span_dates(start_iso: str, end_iso: str) -> list[str]:
    """The SpanTimelineIndex partitions covering [start, end], newest day first.

    Newest first so that a list-mode query with a Limit fills up from the most
    recent spans, which is what ScanIndexForward=False gave the old single-
    partition query.  An unparseable bound yields no days -- and therefore no
    rows -- rather than a guess at what the caller meant.
    """
    try:
        first = date.fromisoformat(str(start_iso)[:10])
        last = date.fromisoformat(str(end_iso)[:10])
    except ValueError:
        return []
    if last < first:
        return []
    span = min((last - first).days + 1, _MAX_SPAN_DATES)
    return [(last - timedelta(days=offset)).isoformat() for offset in range(span)]


def _with_operation_filter(filter_expr, operation: Optional[str]):
    """Apply `operation` as a filter rather than as a partition.

    Under v1 this was the partition key, so "all" meant one query per known
    operation and an operation nobody had enumerated was unreachable.  On the
    index it is an ordinary attribute: "all" is simply the absence of a filter,
    and an operation this repository has never heard of still comes back.
    """
    if not operation or operation == "all":
        return filter_expr
    condition = Attr("operation").eq(operation)
    return condition if filter_expr is None else filter_expr & condition


def query_span_timeline(
    table,
    span_date: str,
    start_iso: str,
    end_iso: str,
    filter_expr,
    limit: int,
    exclusive_start_key: Optional[dict],
) -> tuple[list[dict], int, Optional[dict]]:
    """Query one day of SpanTimelineIndex, bounded by the timestamp range.

    The same range condition is applied to every day: interior days pass it
    whole, and the two end days are trimmed by DynamoDB rather than in memory.
    """
    key_cond = Key(SPAN_TIMELINE_PARTITION_KEY).eq(span_date) & Key(
        SPAN_TIMELINE_SORT_KEY
    ).between(start_iso, end_iso + "~")

    kwargs: dict[str, Any] = {
        "IndexName": SPAN_TIMELINE_INDEX,
        "KeyConditionExpression": key_cond,
        "Limit": limit,
        "ScanIndexForward": False,  # descending by timestamp: newest first
    }
    if filter_expr is not None:
        kwargs["FilterExpression"] = filter_expr
    if exclusive_start_key:
        kwargs["ExclusiveStartKey"] = exclusive_start_key

    resp = table.query(**kwargs)
    return resp.get("Items", []), resp.get("ScannedCount", 0), resp.get("LastEvaluatedKey")


def _is_index_missing_error(exc: BaseException) -> bool:
    """True only for "SpanTimelineIndex does not exist (yet)".

    The GSI is created by a separate stack whose deploy may land after this
    code, and a new index is not queryable until its backfill completes, so a
    reader that cannot tolerate its absence makes deploy order load-bearing.
    Tolerating *any* failure instead would be worse than the bug this migration
    fixes: a throttle or an expired credential would silently serve whatever
    the legacy partitions happen to hold, which for most writers is nothing.
    So the test is narrow and structural -- the error must be a DynamoDB
    ClientError whose code is ResourceNotFoundException, or a
    ValidationException that names an index.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error") or {}
    code = str(error.get("Code") or "")
    if code == "ResourceNotFoundException":
        return True
    if code != "ValidationException":
        return False
    message = str(error.get("Message") or "").lower()
    return any(marker in message for marker in _INDEX_ABSENCE_MARKERS)


def _warn_index_missing_once(exc: BaseException) -> None:
    global _index_fallback_warned
    if _index_fallback_warned:
        return
    _index_fallback_warned = True
    log.warning(
        "observatory_span_timeline_index_unavailable",
        extra={
            "index": SPAN_TIMELINE_INDEX,
            "err": str(exc),
            "fallback": "legacy pk enumeration (rows from writers using other "
                        "pk prefixes are not visible on this path)",
        },
    )


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
        "ScanIndexForward": False,  # descending by timestamp so Limit returns the most recent items
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


def _dedupe(items: list[dict]) -> list[dict]:
    """Drop repeats of the same row, identified by its base-table key.

    Day partitions are disjoint, so in production this changes nothing.  It is
    here because an aggregate that double-counts is indistinguishable from real
    traffic: a retried page, or a cursor resumed across an overlapping
    boundary, would otherwise inflate every sum on the dashboard with no error
    anywhere.  Rows missing a key are kept as-is rather than collapsed
    together, since nothing identifies them.
    """
    seen: set = set()
    unique: list[dict] = []
    for position, item in enumerate(items):
        pk, sk = item.get("pk"), item.get("sk")
        key = (pk, sk) if isinstance(pk, str) and isinstance(sk, str) else ("\x00", position)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _fetch_all_via_index(
    table,
    operation: str,
    start_iso: Optional[str],
    end_iso: Optional[str],
    filter_expr,
) -> tuple[list[dict], int]:
    """Aggregate-mode fetch over SpanTimelineIndex: every day, fully paged."""
    window_start, window_end = _resolve_window(start_iso, end_iso)
    expr = _with_operation_filter(filter_expr, operation)

    all_items: list[dict] = []
    total_scanned = 0
    for day in span_dates(window_start, window_end):
        last_key = None
        while True:
            items, scanned, last_key = query_span_timeline(
                table, day, window_start, window_end, expr,
                limit=_AGGREGATE_SCAN_LIMIT, exclusive_start_key=last_key,
            )
            all_items.extend(items)
            total_scanned += scanned
            if not last_key or len(all_items) >= _AGGREGATE_SCAN_LIMIT:
                break
        if len(all_items) >= _AGGREGATE_SCAN_LIMIT:
            break
    return all_items, total_scanned


def _fetch_all_via_pk(
    table,
    operation: str,
    start_iso: Optional[str],
    end_iso: Optional[str],
    filter_expr,
) -> tuple[list[dict], int]:
    """Legacy aggregate-mode fetch: one pass per enumerated OBSERVATORY# pk.

    Reached only when SpanTimelineIndex is absent.  Unlike the index path it
    does not impose a default time window, because it is deliberately the
    pre-migration behaviour, unchanged: while the index is missing, a caller
    who asked for no range gets what this endpoint has always given them.
    """
    all_items: list[dict] = []
    total_scanned = 0
    ops = _ALL_OPERATION_PKS if operation == "all" else [operation]
    for op in ops:
        pk = f"OBSERVATORY#{op}"
        last_key = None
        while True:
            items, scanned, last_key = query_by_pk(
                table, pk, start_iso, end_iso, filter_expr,
                limit=_AGGREGATE_SCAN_LIMIT, exclusive_start_key=last_key
            )
            all_items.extend(items)
            total_scanned += scanned
            if not last_key or len(all_items) >= _AGGREGATE_SCAN_LIMIT:
                break
    return all_items, total_scanned


def fetch_all_for_aggregate(
    table,
    operation: str,
    agent_id: Optional[str],
    start_iso: Optional[str],
    end_iso: Optional[str],
    filter_expr,
) -> tuple[list[dict], int]:
    """Fetch all matching items for in-memory aggregation (no pagination)."""
    if agent_id:
        # AgentIdTimestampIndex: unrelated to this migration and already keyed
        # on an attribute every writer sets, so it is untouched.
        all_items: list[dict] = []
        total_scanned = 0
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
        try:
            all_items, total_scanned = _fetch_all_via_index(
                table, operation, start_iso, end_iso, filter_expr
            )
        except Exception as exc:  # noqa: BLE001 -- re-raised unless it is the index
            if not _is_index_missing_error(exc):
                raise
            _warn_index_missing_once(exc)
            all_items, total_scanned = _fetch_all_via_pk(
                table, operation, start_iso, end_iso, filter_expr
            )

    return _dedupe([_normalize_item(item) for item in all_items]), total_scanned


def _list_via_index(
    table,
    operation: str,
    start_iso: Optional[str],
    end_iso: Optional[str],
    filter_expr,
    limit: int,
    exclusive_start_key: Optional[dict],
) -> tuple[list[dict], int, Optional[dict]]:
    """List-mode fetch over SpanTimelineIndex, newest day first.

    A page ends at the first day DynamoDB could not finish: that day's
    LastEvaluatedKey becomes the next_token, and because it carries span_date
    the resumed request knows which day to pick up on and which days it has
    already returned.
    """
    window_start, window_end = _resolve_window(start_iso, end_iso)
    days = span_dates(window_start, window_end)
    expr = _with_operation_filter(filter_expr, operation)

    resume_key = None
    if exclusive_start_key:
        resume_day = _unwrap_ddb_value(exclusive_start_key.get(SPAN_TIMELINE_PARTITION_KEY))
        if isinstance(resume_day, str) and resume_day:
            days = [day for day in days if day <= resume_day]
            resume_key = exclusive_start_key

    items: list[dict] = []
    scanned = 0
    last_key: Optional[dict] = None
    for position, day in enumerate(days):
        page, page_scanned, page_last_key = query_span_timeline(
            table, day, window_start, window_end, expr,
            limit=limit, exclusive_start_key=resume_key if position == 0 else None,
        )
        items.extend(page)
        scanned += page_scanned
        if page_last_key:
            last_key = page_last_key
            break
        if len(items) >= limit:
            break
    return items, scanned, last_key


def _list_via_pk(
    table,
    operation: str,
    start_iso: Optional[str],
    end_iso: Optional[str],
    filter_expr,
    limit: int,
    exclusive_start_key: Optional[dict],
) -> tuple[list[dict], int, Optional[dict]]:
    """Legacy list-mode fetch: the pre-migration pk enumeration, unchanged."""
    if operation == "all":
        # Merged partitions; pagination was never supported for this shape.
        items: list[dict] = []
        scanned = 0
        for op in _ALL_OPERATION_PKS:
            op_items, op_scanned, _ = query_by_pk(
                table, f"OBSERVATORY#{op}", start_iso, end_iso, filter_expr,
                limit=limit, exclusive_start_key=None,
            )
            items.extend(op_items)
            scanned += op_scanned
        return items, scanned, None

    return query_by_pk(
        table, f"OBSERVATORY#{operation}", start_iso, end_iso, filter_expr,
        limit=limit, exclusive_start_key=exclusive_start_key,
    )


def _list_items(
    table,
    operation: str,
    start_iso: Optional[str],
    end_iso: Optional[str],
    filter_expr,
    limit: int,
    exclusive_start_key: Optional[dict],
) -> tuple[list[dict], int, Optional[dict]]:
    """List-mode fetch: the index, falling back to pk only when it is absent."""
    if exclusive_start_key and not exclusive_start_key.get(SPAN_TIMELINE_PARTITION_KEY):
        # A cursor is only meaningful against the index that minted it.  One
        # without span_date came from the legacy path, so resume it there
        # rather than replaying a base-table key against the GSI.
        return _list_via_pk(
            table, operation, start_iso, end_iso, filter_expr, limit, exclusive_start_key
        )

    try:
        return _list_via_index(
            table, operation, start_iso, end_iso, filter_expr, limit, exclusive_start_key
        )
    except Exception as exc:  # noqa: BLE001 -- re-raised unless it is the index
        if not _is_index_missing_error(exc):
            raise
        _warn_index_missing_once(exc)
        return _list_via_pk(
            table, operation, start_iso, end_iso, filter_expr, limit, exclusive_start_key
        )


def aggregate_items(items: list[dict], mode: str) -> list[dict]:
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
    table = get_table()
    if table is None:
        return json_response(500, {"error": "OBSERVATORY_METRICS_TABLE environment variable not set"})

    params: dict[str, str] = event.get("queryStringParameters") or {}

    # --- Parameter parsing & validation ---
    operation = params.get("operation", "all").lower()
    if operation not in _VALID_OPERATIONS:
        return json_response(400, {"error": f"operation must be one of {sorted(_VALID_OPERATIONS)}"})

    aggregate = params.get("aggregate", "none").lower()
    if aggregate not in _VALID_AGGREGATES:
        return json_response(400, {"error": f"aggregate must be one of {sorted(_VALID_AGGREGATES)}"})

    sort_by = params.get("sort_by", "timestamp").lower()
    if sort_by not in _VALID_SORT_BY:
        return json_response(400, {"error": f"sort_by must be one of {sorted(_VALID_SORT_BY)}"})

    sort_order = params.get("sort_order", "desc").lower()
    if sort_order not in {"asc", "desc"}:
        return json_response(400, {"error": "sort_order must be 'asc' or 'desc'"})

    try:
        limit = int(params.get("limit", _DEFAULT_LIMIT))
    except ValueError:
        return json_response(400, {"error": "limit must be an integer"})
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
            return json_response(400, {"error": "start must be a Unix epoch or ISO 8601 timestamp"})
    if params.get("end"):
        try:
            end_iso = _parse_timestamp(params["end"])
        except Exception:
            return json_response(400, {"error": "end must be a Unix epoch or ISO 8601 timestamp"})

    filter_expr = _build_filter_expression(
        model_id, decision, risk_tier, policy_decision,
        composite_risk_level, hallucination_risk_level,
        is_shadow, gate_blocked, fallback_used,
    )

    # --- Aggregate mode: fetch all, group, return ---
    if aggregate != "none":
        try:
            items, scanned = fetch_all_for_aggregate(
                table, operation, agent_id, start_iso, end_iso, filter_expr
            )
        except Exception as exc:
            log.error("agent_metrics_aggregate_error", extra={"err": str(exc)})
            return json_response(500, {"error": "Failed to query metrics"})

        groups = aggregate_items(items, aggregate)
        return json_response(200, {
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
        else:
            items, scanned, last_key = _list_items(
                table, operation, start_iso, end_iso, filter_expr,
                limit=limit, exclusive_start_key=exclusive_start_key,
            )
            items = _dedupe([_normalize_item(item) for item in items])
    except Exception as exc:
        log.error("agent_metrics_query_error", extra={"err": str(exc)})
        return json_response(500, {"error": "Failed to query metrics"})

    # Sort
    items = _sort_items(items, sort_by, sort_order)[:limit]

    response_body: dict = {
        "items": items,
        "count": len(items),
        "scanned_count": scanned,
    }
    if last_key:
        response_body["next_token"] = _encode_next_token(last_key)

    return json_response(200, response_body)
