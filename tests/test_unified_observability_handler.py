"""Unit tests for src/orchestrator/unified_observability_handler.py.

DynamoDB is mocked the same way tests/test_agent_metrics_handler.py mocks it
(patch boto3.resource, hand back a MagicMock table with .query() stubbed);
the ContextWeave half is mocked at the client-function boundary so the HTTP
layer stays the concern of tests/test_contextweave_client.py.
"""
import importlib
import json
import os
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

# Pre-import boto3 so module-level stubs installed by other test files do not
# shadow the real package before this module imports the handler.
import boto3  # noqa: F401

_URL = "https://contextweave.example.com"

_HEALTH = {
    "status": "healthy",
    "knowledgeBaseId": "kb-123",
    "neptuneGraphId": "g-123",
    "environment": "dev",
    "routingGraph": {
        "documentTypeDistribution": [{"documentType": "technical_spec", "count": 12}],
        "health": {
            "exploration": 0.21,
            "priorStrength": 2.0,
            "questionTypes": {
                "skill_depth": {
                    "verdict": "converged",
                    "leader": "graph_first",
                    "leaderPBest": 0.97,
                    "starvedArms": [],
                    "arms": {"graph_first": {"alpha": 40.0, "beta": 5.0}},
                }
            },
        },
    },
}

_DECISIONS = {
    "groups": [
        {
            "questionType": "skill_depth",
            "strategy": "graph_first",
            "count": 120,
            "ratedCount": 34,
            "avgConfidence": 0.81,
            "avgRating": 0.74,
            "meanAbsDiff": 0.19,
        },
        {
            "questionType": "project",
            "strategy": "keyword_boosted",
            "count": 12,
            "ratedCount": 0,
            "avgConfidence": None,
            "avgRating": None,
            "meanAbsDiff": None,
        },
    ],
    "totalCount": 512,
    "totalRated": 140,
}

_ITEMS = [
    {
        "pk": "OBSERVATORY#invoke_agent",
        "sk": "2026-09-13T10:00:00.000000#t1",
        "operation": "invoke_agent",
        "agent_id": "AGENT1",
        "model_id": "nova-lite",
        "timestamp": "2026-09-13T10:00:00.000000",
        "cost_usd": Decimal("0.010"),
        "prompt_tokens": Decimal("100"),
    },
    {
        "pk": "OBSERVATORY#invoke_agent",
        "sk": "2026-09-13T11:00:00.000000#t2",
        "operation": "invoke_agent",
        "agent_id": "AGENT2",
        "model_id": "nova-lite",
        "timestamp": "2026-09-13T11:00:00.000000",
        "cost_usd": Decimal("0.030"),
        "prompt_tokens": Decimal("300"),
    },
]


def _load():
    return importlib.import_module("src.orchestrator.unified_observability_handler")


def _metrics_module():
    return importlib.import_module("src.orchestrator.agent_metrics_handler")


def _event(params=None):
    return {"queryStringParameters": params or {}}


def _mock_table(items=None, query_side_effect=None):
    """A DynamoDB table whose invoke_agent partition holds ``items``.

    An aggregate over operation "all" queries every known operation PK in turn
    (invoke_agent, invoke_model, classify_question, synthesize_answer), so only
    the first query returns rows; the rest are empty partitions.  Returning
    ``items`` for every call would count them four times.
    """
    tbl = MagicMock()
    if query_side_effect is not None:
        tbl.query.side_effect = query_side_effect
        return tbl

    responses = [{"Items": list(items or []), "ScannedCount": len(items or [])}]

    def _query(**_kwargs):
        if responses:
            return responses.pop(0)
        return {"Items": [], "ScannedCount": 0}

    tbl.query.side_effect = _query
    return tbl


def _run(params=None, table=None, table_name="test-table", cw_url=_URL,
         health=_HEALTH, decisions=_DECISIONS):
    """Invoke the handler with DynamoDB and the ContextWeave client both mocked.

    ``health`` / ``decisions`` of None simulate a failing ContextWeave call
    (that is exactly what the client returns when it gives up).
    """
    mod = _load()
    _metrics_module()._ddb_table = None  # reset the shared lazy table cache

    env = {"CONTEXTWEAVE_URL": cw_url or ""}
    if table_name:
        env["OBSERVATORY_METRICS_TABLE"] = table_name

    if table is None:
        table = _mock_table(_ITEMS)

    with patch.dict(os.environ, env, clear=False):
        if not table_name:
            os.environ.pop("OBSERVATORY_METRICS_TABLE", None)
        with patch("boto3.resource") as mock_res, \
                patch.object(mod.contextweave_client, "get_health",
                             return_value=health) as mock_health, \
                patch.object(mod.contextweave_client, "get_routing_decisions_summary",
                             return_value=decisions) as mock_decisions:
            mock_res.return_value.Table.return_value = table
            resp = mod.handler(_event(params), None)
    return resp, mock_health, mock_decisions


def _body(resp):
    return json.loads(resp["body"])


class TestAllThreeSourcesHealthy(unittest.TestCase):
    def test_composes_all_three_sections(self):
        resp, _, _ = _run()
        self.assertEqual(resp["statusCode"], 200)
        body = _body(resp)

        self.assertEqual(
            set(body),
            {"generatedAt", "observatoryMetrics", "routingGraph", "routingDecisions"},
        )
        self.assertEqual(body["observatoryMetrics"]["aggregate"], "by_operation")
        self.assertEqual(body["routingGraph"], _HEALTH["routingGraph"])
        self.assertEqual(body["routingDecisions"], _DECISIONS)

    def test_observatory_metrics_are_aggregated_by_operation(self):
        resp, _, _ = _run()
        groups = _body(resp)["observatoryMetrics"]["groups"]

        # Both items share operation invoke_agent -> exactly one group.
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["key"], {"operation": "invoke_agent"})
        self.assertEqual(groups[0]["count"], 2)
        self.assertAlmostEqual(groups[0]["sum_cost_usd"], 0.04)
        self.assertAlmostEqual(groups[0]["avg_prompt_tokens"], 200.0)

    def test_decimals_are_serialised_as_numbers(self):
        resp, _, _ = _run()
        # json.loads would raise if Decimal had leaked into the body unserialised.
        self.assertIsInstance(
            _body(resp)["observatoryMetrics"]["groups"][0]["sum_cost_usd"], float
        )

    def test_generated_at_is_utc_iso8601(self):
        from datetime import datetime
        resp, _, _ = _run()
        parsed = datetime.fromisoformat(_body(resp)["generatedAt"])
        self.assertIsNotNone(parsed.tzinfo)

    def test_cors_headers_match_agent_metrics_endpoint(self):
        resp, _, _ = _run()
        self.assertEqual(resp["headers"], _metrics_module().CORS_HEADERS)

    def test_query_params_are_forwarded_to_routing_decisions(self):
        _, _, mock_decisions = _run({"questionType": "skill_depth", "since": "2026-09-01"})
        mock_decisions.assert_called_once_with(
            question_type="skill_depth", since="2026-09-01"
        )

    def test_absent_query_params_are_forwarded_as_none(self):
        _, _, mock_decisions = _run({})
        mock_decisions.assert_called_once_with(question_type=None, since=None)

    def test_missing_query_string_parameters_key_is_tolerated(self):
        mod = _load()
        _metrics_module()._ddb_table = None
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": "", "OBSERVATORY_METRICS_TABLE": "t"}):
            with patch("boto3.resource") as mock_res:
                mock_res.return_value.Table.return_value = _mock_table(_ITEMS)
                resp = mod.handler({}, None)
        self.assertEqual(resp["statusCode"], 200)


class TestContextWeaveNotConfigured(unittest.TestCase):
    def test_both_contextweave_sections_are_null(self):
        resp, _, _ = _run(cw_url="")
        body = _body(resp)

        self.assertEqual(resp["statusCode"], 200)
        self.assertIsNone(body["routingGraph"])
        self.assertIsNone(body["routingDecisions"])

    def test_no_http_call_is_attempted(self):
        _, mock_health, mock_decisions = _run(cw_url="")
        mock_health.assert_not_called()
        mock_decisions.assert_not_called()

    def test_observatory_metrics_still_populated(self):
        resp, _, _ = _run(cw_url="")
        self.assertEqual(_body(resp)["observatoryMetrics"]["total_count"], 2)


class TestContextWeaveDegrades(unittest.TestCase):
    def test_health_failure_reports_error_object_and_keeps_metrics(self):
        resp, _, _ = _run(health=None)
        body = _body(resp)

        self.assertEqual(resp["statusCode"], 200)
        self.assertIn("error", body["routingGraph"])
        self.assertEqual(body["routingDecisions"], _DECISIONS)
        self.assertEqual(body["observatoryMetrics"]["total_count"], 2)

    def test_routing_decisions_failure_reports_error_object(self):
        resp, _, _ = _run(decisions=None)
        body = _body(resp)

        self.assertIn("error", body["routingDecisions"])
        self.assertEqual(body["routingGraph"], _HEALTH["routingGraph"])

    def test_both_contextweave_calls_failing_still_returns_200(self):
        resp, _, _ = _run(health=None, decisions=None)
        body = _body(resp)

        self.assertEqual(resp["statusCode"], 200)
        self.assertIn("error", body["routingGraph"])
        self.assertIn("error", body["routingDecisions"])
        self.assertEqual(body["observatoryMetrics"]["aggregate"], "by_operation")

    def test_health_without_routing_graph_key_yields_null(self):
        resp, _, _ = _run(health={"status": "healthy"})
        self.assertIsNone(_body(resp)["routingGraph"])


class TestObservatoryMetricsDegrades(unittest.TestCase):
    def test_missing_table_env_degrades_instead_of_500(self):
        """Unlike /observability/agent-metrics (which 500s), this endpoint
        reports the missing table as an error object and still serves the
        ContextWeave sections."""
        resp, _, _ = _run(table_name=None)
        body = _body(resp)

        self.assertEqual(resp["statusCode"], 200)
        self.assertIn("OBSERVATORY_METRICS_TABLE", body["observatoryMetrics"]["error"])
        self.assertEqual(body["routingGraph"], _HEALTH["routingGraph"])
        self.assertEqual(body["routingDecisions"], _DECISIONS)

    def test_dynamodb_failure_degrades_to_error_object(self):
        table = _mock_table(query_side_effect=RuntimeError("throttled"))
        resp, _, _ = _run(table=table)
        body = _body(resp)

        self.assertEqual(resp["statusCode"], 200)
        self.assertIn("error", body["observatoryMetrics"])
        self.assertEqual(body["routingDecisions"], _DECISIONS)

    def test_empty_table_yields_empty_groups(self):
        resp, _, _ = _run(table=_mock_table([]))
        section = _body(resp)["observatoryMetrics"]

        self.assertEqual(section["groups"], [])
        self.assertEqual(section["total_count"], 0)


class TestSharedMetricsLogic(unittest.TestCase):
    def test_handler_reuses_agent_metrics_helpers(self):
        """The DynamoDB/aggregation logic has one source of truth."""
        mod = _load()
        metrics = _metrics_module()

        self.assertIs(mod.fetch_all_for_aggregate, metrics.fetch_all_for_aggregate)
        self.assertIs(mod.aggregate_items, metrics.aggregate_items)
        self.assertIs(mod.get_table, metrics.get_table)
        self.assertIs(mod.json_response, metrics.json_response)


if __name__ == "__main__":
    unittest.main()
