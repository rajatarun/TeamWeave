"""Unified observability Lambda handler — one read for the whole platform.

Serves ``GET /observability``, composing the three places platform health
currently lives into a single response:

1. ``observatoryMetrics``  -- the OBSERVATORY_METRICS DynamoDB table, aggregated
   by operation (the same data ``GET /observability/agent-metrics?aggregate=by_operation``
   returns, produced by the very same helpers in ``agent_metrics_handler``).
2. ``routingGraph``        -- the ``routingGraph`` block of ContextWeave's
   ``GET /health``: per question type, which retrieval strategy leads and
   whether the router is ``learning`` / ``converged`` / ``starved``.
3. ``routingDecisions``    -- ContextWeave's ``GET /routing-decisions?mode=summary``
   rollup: per (question type, strategy) counts, average self-confidence,
   average human rating, and the mean gap between the two.

Degradation
-----------
Each source degrades **independently**; the endpoint answers 200 with whatever
it could gather.  This follows ``contextweave_client``'s stated philosophy —
"a knowledge layer being down must never fail a pipeline run" — applied to a
dashboard: partial data beats a 500, because a console that shows two of three
panels is still useful, while a 500 shows none of them.

Concretely:

* ContextWeave not configured (``CONTEXTWEAVE_URL`` unset) -> ``routingGraph``
  and ``routingDecisions`` are ``null`` and no HTTP call is attempted.  A
  deployment that never opted into ContextWeave is a supported deployment, not
  an error.
* ContextWeave configured but unreachable / erroring -> that key is an
  ``{"error": "..."}`` object.
* ``OBSERVATORY_METRICS_TABLE`` unset, or the DynamoDB query failing ->
  ``observatoryMetrics`` is an ``{"error": "..."}`` object.  This deliberately
  differs from ``agent_metrics_handler``, which 500s when the table env var is
  missing: that endpoint has nothing left to say without the table, whereas
  this one still has two other sources to report.

Query parameters
----------------
questionType  (optional) -- forwarded to ContextWeave's routing-decisions call
since         (optional) -- forwarded to ContextWeave's routing-decisions call

Responses
---------
200  {"generatedAt": "...", "observatoryMetrics": {...}, "routingGraph": {...} | null,
      "routingDecisions": {...} | null}
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from . import contextweave_client
from .agent_metrics_handler import (
    aggregate_items,
    fetch_all_for_aggregate,
    get_table,
    json_response,
)
from .logger import get_logger

log = get_logger("unified_observability_handler")

# The aggregation the console's top-level view needs: one row per operation.
_AGGREGATE_MODE = "by_operation"


def _observatory_metrics_section() -> dict:
    """Aggregate OBSERVATORY_METRICS by operation, or describe why we could not."""
    table = get_table()
    if table is None:
        log.warning("unified_observability_metrics_unconfigured")
        return {"error": "OBSERVATORY_METRICS_TABLE environment variable not set"}

    try:
        items, scanned = fetch_all_for_aggregate(
            table,
            operation="all",
            agent_id=None,
            start_iso=None,
            end_iso=None,
            filter_expr=None,
        )
    except Exception as exc:
        log.error("unified_observability_metrics_error", extra={"err": str(exc)})
        return {"error": "Failed to query metrics"}

    return {
        "aggregate": _AGGREGATE_MODE,
        "groups": aggregate_items(items, _AGGREGATE_MODE),
        "total_count": len(items),
        "scanned_count": scanned,
    }


def _routing_graph_section() -> Optional[dict]:
    """The routingGraph block of ContextWeave's /health, or an error object."""
    payload = contextweave_client.get_health()
    if payload is None:
        return {"error": "ContextWeave /health unavailable"}
    return payload.get("routingGraph")


def _routing_decisions_section(
    question_type: Optional[str],
    since: Optional[str],
) -> Optional[dict]:
    """ContextWeave's routing-decisions summary, or an error object."""
    payload = contextweave_client.get_routing_decisions_summary(
        question_type=question_type, since=since
    )
    if payload is None:
        return {"error": "ContextWeave /routing-decisions unavailable"}
    return payload


def handler(event: dict, context: object) -> dict:
    params: dict[str, str] = event.get("queryStringParameters") or {}
    question_type = params.get("questionType") or None
    since = params.get("since") or None

    body: dict[str, Any] = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "observatoryMetrics": _observatory_metrics_section(),
    }

    if contextweave_client.is_configured():
        body["routingGraph"] = _routing_graph_section()
        body["routingDecisions"] = _routing_decisions_section(question_type, since)
    else:
        # Not an error: ContextWeave is optional for this platform.
        log.info("unified_observability_contextweave_not_configured")
        body["routingGraph"] = None
        body["routingDecisions"] = None

    return json_response(200, body)
