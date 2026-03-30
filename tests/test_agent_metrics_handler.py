import base64
import importlib
import json
import os
import sys
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, call, patch

# Pre-import boto3 so that module-level stubs installed by other test files
# (e.g. test_enrich.py uses sys.modules.setdefault) do not shadow the real
# boto3 package before our tests attempt to import agent_metrics_handler.
import boto3  # noqa: F401


def _load():
    return importlib.import_module("src.orchestrator.agent_metrics_handler")


def _event(params=None):
    return {"queryStringParameters": params or {}}


def _condition_values(condition) -> list:
    """Extract all attribute values from a boto3 DynamoDB condition by rendering it."""
    from boto3.dynamodb.conditions import ConditionExpressionBuilder
    builder = ConditionExpressionBuilder()
    expr = builder.build_expression(condition)
    return list(expr.attribute_value_placeholders.values())


def _mock_table(items=None, scanned=None, last_key=None):
    """Return a mock DynamoDB Table with .query() pre-configured."""
    tbl = MagicMock()
    resp = {
        "Items": items or [],
        "ScannedCount": scanned if scanned is not None else len(items or []),
    }
    if last_key:
        resp["LastEvaluatedKey"] = last_key
    tbl.query.return_value = resp
    return tbl


def _run(params, table=None, table_name="test-table"):
    m = _load()
    # Reset module-level cache between tests
    m._ddb_table = None
    env = {"OBSERVATORY_METRICS_TABLE": table_name} if table_name else {}
    if table is None:
        table = _mock_table()
    with patch.dict(os.environ, env, clear=False):
        with patch("boto3.resource") as mock_res:
            mock_res.return_value.Table.return_value = table
            return m.handler(_event(params), None)


class TestMissingTableEnv(unittest.TestCase):
    def test_missing_env_returns_500(self):
        m = _load()
        m._ddb_table = None
        with patch.dict(os.environ, {}, clear=True):
            # Remove key if present
            os.environ.pop("OBSERVATORY_METRICS_TABLE", None)
            resp = m.handler(_event({}), None)
        self.assertEqual(resp["statusCode"], 500)
        body = json.loads(resp["body"])
        self.assertIn("error", body)


class TestInputValidation(unittest.TestCase):
    def test_invalid_operation_returns_400(self):
        resp = _run({"operation": "invoke_something_bad"})
        self.assertEqual(resp["statusCode"], 400)

    def test_invalid_aggregate_returns_400(self):
        resp = _run({"aggregate": "by_unicorn"})
        self.assertEqual(resp["statusCode"], 400)

    def test_invalid_sort_by_returns_400(self):
        resp = _run({"sort_by": "not_a_field"})
        self.assertEqual(resp["statusCode"], 400)

    def test_invalid_sort_order_returns_400(self):
        resp = _run({"sort_order": "sideways"})
        self.assertEqual(resp["statusCode"], 400)

    def test_invalid_limit_returns_400(self):
        resp = _run({"limit": "abc"})
        self.assertEqual(resp["statusCode"], 400)

    def test_limit_clamped_to_1000(self):
        tbl = _mock_table(items=[{"trace_id": "x", "operation": "invoke_agent",
                                  "timestamp": "2024-01-01T00:00:00", "pk": "OBSERVATORY#invoke_agent",
                                  "sk": "2024-01-01T00:00:00#x"}])
        resp = _run({"limit": "9999", "operation": "invoke_agent"}, table=tbl)
        self.assertEqual(resp["statusCode"], 200)
        # Verify DynamoDB was called with Limit=1000
        call_kwargs = tbl.query.call_args[1]
        self.assertEqual(call_kwargs["Limit"], 1000)


class TestListModeQueryRouting(unittest.TestCase):
    def _item(self, op, ts="2024-01-15T10:30:00.000000", trace="abc"):
        return {
            "pk": f"OBSERVATORY#{op}",
            "sk": f"{ts}#{trace}",
            "trace_id": trace,
            "operation": op,
            "timestamp": ts,
            "prompt_tokens": Decimal("100"),
            "completion_tokens": Decimal("50"),
            "cost_usd": Decimal("0.001"),
            "decision": "allow",
            "decision_reason": "none",
        }

    def test_invoke_agent_queries_correct_pk(self):
        tbl = _mock_table(items=[self._item("invoke_agent")])
        _run({"operation": "invoke_agent"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        values = _condition_values(call_kwargs["KeyConditionExpression"])
        self.assertIn("OBSERVATORY#invoke_agent", values)

    def test_invoke_model_queries_correct_pk(self):
        tbl = _mock_table(items=[self._item("invoke_model")])
        _run({"operation": "invoke_model"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        values = _condition_values(call_kwargs["KeyConditionExpression"])
        self.assertIn("OBSERVATORY#invoke_model", values)

    def test_all_operations_queries_both_pks(self):
        tbl = _mock_table(items=[])
        _run({"operation": "all"}, table=tbl)
        # Should have been called once per known operation PK (invoke_agent,
        # invoke_model, classify_question, synthesize_answer)
        self.assertEqual(tbl.query.call_count, 4)
        all_values = []
        for c in tbl.query.call_args_list:
            all_values.extend(_condition_values(c[1]["KeyConditionExpression"]))
        self.assertIn("OBSERVATORY#invoke_agent", all_values)
        self.assertIn("OBSERVATORY#invoke_model", all_values)
        self.assertIn("OBSERVATORY#classify_question", all_values)
        self.assertIn("OBSERVATORY#synthesize_answer", all_values)

    def test_agent_id_filter_uses_gsi(self):
        tbl = _mock_table(items=[self._item("invoke_agent")])
        _run({"agent_id": "AGENT-123"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        self.assertEqual(call_kwargs.get("IndexName"), "AgentIdTimestampIndex")
        values = _condition_values(call_kwargs["KeyConditionExpression"])
        self.assertIn("AGENT-123", values)

    def test_agent_id_operation_filter_falls_back_to_pk_for_legacy_items(self):
        legacy_item = self._item("invoke_agent")
        legacy_item.pop("operation")
        tbl = _mock_table(items=[legacy_item])
        resp = _run({"agent_id": "AGENT-123", "operation": "invoke_agent"}, table=tbl)
        body = json.loads(resp["body"])
        self.assertEqual(body["count"], 1)

    def test_time_range_filter_applied_to_sk(self):
        tbl = _mock_table(items=[self._item("invoke_agent")])
        _run({"operation": "invoke_agent", "start": "2024-01-01T00:00:00",
              "end": "2024-12-31T23:59:59"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        values = _condition_values(call_kwargs["KeyConditionExpression"])
        # Both start and end timestamps should appear as values
        self.assertTrue(any("2024-01-01" in v for v in values))
        self.assertTrue(any("2024-12-31" in v for v in values))

    def test_unix_epoch_start_normalised_to_iso(self):
        tbl = _mock_table(items=[self._item("invoke_agent")])
        _run({"operation": "invoke_agent", "start": "1704067200"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        values = _condition_values(call_kwargs["KeyConditionExpression"])
        # 1704067200 = 2024-01-01T00:00:00 UTC
        self.assertTrue(any("2024" in v for v in values))

    def test_decision_filter_expression_applied(self):
        tbl = _mock_table(items=[self._item("invoke_agent")])
        _run({"operation": "invoke_agent", "decision": "block"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        self.assertIn("FilterExpression", call_kwargs)
        values = _condition_values(call_kwargs["FilterExpression"])
        self.assertIn("block", values)

    def test_model_id_filter_expression_applied(self):
        tbl = _mock_table(items=[self._item("invoke_model")])
        _run({"operation": "invoke_model", "model_id": "nova-micro"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        self.assertIn("FilterExpression", call_kwargs)
        values = _condition_values(call_kwargs["FilterExpression"])
        self.assertIn("nova-micro", values)

    def test_no_filter_expression_when_no_filters(self):
        tbl = _mock_table(items=[self._item("invoke_agent")])
        _run({"operation": "invoke_agent"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        self.assertNotIn("FilterExpression", call_kwargs)


class TestListModeResponse(unittest.TestCase):
    def _item(self, op="invoke_agent", ts="2024-01-15T10:30:00.000000",
              trace="t1", cost="0.005"):
        return {
            "pk": f"OBSERVATORY#{op}", "sk": f"{ts}#{trace}",
            "trace_id": trace, "operation": op, "timestamp": ts,
            "prompt_tokens": Decimal("100"), "completion_tokens": Decimal("50"),
            "cost_usd": Decimal(cost), "decision": "allow",
        }

    def test_returns_200_with_items(self):
        tbl = _mock_table(items=[self._item()])
        resp = _run({"operation": "invoke_agent"}, table=tbl)
        self.assertEqual(resp["statusCode"], 200)
        body = json.loads(resp["body"])
        self.assertIn("items", body)
        self.assertEqual(body["count"], 1)

    def test_empty_result_returns_empty_items(self):
        tbl = _mock_table(items=[])
        resp = _run({"operation": "invoke_agent"}, table=tbl)
        self.assertEqual(resp["statusCode"], 200)
        body = json.loads(resp["body"])
        self.assertEqual(body["items"], [])
        self.assertEqual(body["count"], 0)

    def test_decimal_serialised_to_float(self):
        tbl = _mock_table(items=[self._item()])
        resp = _run({"operation": "invoke_agent"}, table=tbl)
        self.assertEqual(resp["statusCode"], 200)
        # Should not raise — Decimal -> float
        body = json.loads(resp["body"])
        item = body["items"][0]
        self.assertIsInstance(item["cost_usd"], float)
        self.assertIsInstance(item["prompt_tokens"], float)

    def test_cors_headers_present(self):
        tbl = _mock_table(items=[])
        resp = _run({"operation": "invoke_agent"}, table=tbl)
        self.assertIn("Access-Control-Allow-Origin", resp["headers"])
        self.assertIn("Content-Type", resp["headers"])

    def test_cors_headers_present_on_400(self):
        resp = _run({"operation": "bad"})
        self.assertIn("Access-Control-Allow-Origin", resp["headers"])

    def test_no_next_token_when_no_last_key(self):
        tbl = _mock_table(items=[self._item()])
        resp = _run({"operation": "invoke_agent"}, table=tbl)
        body = json.loads(resp["body"])
        self.assertNotIn("next_token", body)

    def test_next_token_present_when_last_key(self):
        tbl = _mock_table(
            items=[self._item()],
            last_key={"pk": "OBSERVATORY#invoke_agent", "sk": "2024-01-15T10:30:00#t1"}
        )
        resp = _run({"operation": "invoke_agent"}, table=tbl)
        body = json.loads(resp["body"])
        self.assertIn("next_token", body)
        # Should be valid base64 JSON
        decoded = json.loads(base64.b64decode(body["next_token"]))
        self.assertIn("pk", decoded)

    def test_next_token_decoded_as_exclusive_start_key(self):
        last_key = {"pk": "OBSERVATORY#invoke_agent", "sk": "2024-01-15T10:30:00#t1"}
        token = base64.b64encode(json.dumps(last_key).encode()).decode()
        tbl = _mock_table(items=[self._item()])
        _run({"operation": "invoke_agent", "next_token": token}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        self.assertIn("ExclusiveStartKey", call_kwargs)
        self.assertEqual(call_kwargs["ExclusiveStartKey"]["pk"], "OBSERVATORY#invoke_agent")

    def test_sort_by_cost_usd_desc(self):
        items = [
            self._item(trace="cheap", cost="0.001"),
            self._item(trace="expensive", cost="0.009"),
            self._item(trace="medium", cost="0.005"),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "sort_by": "cost_usd", "sort_order": "desc"},
                    table=tbl)
        body = json.loads(resp["body"])
        costs = [i["cost_usd"] for i in body["items"]]
        self.assertEqual(costs, sorted(costs, reverse=True))

    def test_sort_by_cost_usd_asc(self):
        items = [
            self._item(trace="cheap", cost="0.001"),
            self._item(trace="expensive", cost="0.009"),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "sort_by": "cost_usd", "sort_order": "asc"},
                    table=tbl)
        body = json.loads(resp["body"])
        costs = [i["cost_usd"] for i in body["items"]]
        self.assertEqual(costs, sorted(costs))


class TestAggregateMode(unittest.TestCase):
    def _agent_item(self, agent_id, trace, cost="0.001", prompt=100, completion=50,
                    ts="2024-01-15T10:30:00.000000"):
        return {
            "pk": "OBSERVATORY#invoke_agent", "sk": f"{ts}#{trace}",
            "trace_id": trace, "operation": "invoke_agent", "timestamp": ts,
            "agent_id": agent_id,
            "prompt_tokens": Decimal(str(prompt)),
            "completion_tokens": Decimal(str(completion)),
            "cost_usd": Decimal(cost),
            "decision": "allow",
        }

    def _model_item(self, model_id, trace, cost="0.002"):
        ts = "2024-01-15T11:00:00.000000"
        return {
            "pk": "OBSERVATORY#invoke_model", "sk": f"{ts}#{trace}",
            "trace_id": trace, "operation": "invoke_model", "timestamp": ts,
            "model_id": model_id,
            "prompt_tokens": Decimal("200"),
            "completion_tokens": Decimal("100"),
            "cost_usd": Decimal(cost),
            "decision": "allow",
        }

    def test_aggregate_by_agent_groups_correctly(self):
        items = [
            self._agent_item("A1", "t1", cost="0.001"),
            self._agent_item("A1", "t2", cost="0.003"),
            self._agent_item("A2", "t3", cost="0.002"),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_agent"}, table=tbl)
        self.assertEqual(resp["statusCode"], 200)
        body = json.loads(resp["body"])
        self.assertEqual(body["aggregate"], "by_agent")
        groups = {g["key"]["agent_id"]: g for g in body["groups"]}
        self.assertIn("A1", groups)
        self.assertIn("A2", groups)
        self.assertEqual(groups["A1"]["count"], 2)
        self.assertAlmostEqual(groups["A1"]["sum_cost_usd"], 0.004, places=5)
        self.assertAlmostEqual(groups["A1"]["avg_cost_usd"], 0.002, places=5)
        self.assertAlmostEqual(groups["A1"]["min_cost_usd"], 0.001, places=5)
        self.assertAlmostEqual(groups["A1"]["max_cost_usd"], 0.003, places=5)
        self.assertEqual(groups["A2"]["count"], 1)

    def test_aggregate_by_model_groups_correctly(self):
        items = [
            self._model_item("nova-micro", "t1", cost="0.001"),
            self._model_item("nova-micro", "t2", cost="0.001"),
            self._model_item("claude-haiku", "t3", cost="0.005"),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_model", "aggregate": "by_model"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["model_id"]: g for g in body["groups"]}
        self.assertEqual(groups["nova-micro"]["count"], 2)
        self.assertEqual(groups["claude-haiku"]["count"], 1)

    def test_aggregate_by_operation(self):
        items = [
            self._agent_item("A1", "t1"),
            self._agent_item("A2", "t2"),
            self._model_item("M1", "t3"),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"aggregate": "by_operation"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["operation"]: g for g in body["groups"]}
        self.assertIn("invoke_agent", groups)

    def test_aggregate_by_operation_uses_pk_when_operation_missing(self):
        item = self._agent_item("A1", "t1")
        item.pop("operation")
        tbl = _mock_table(items=[item])
        resp = _run({"aggregate": "by_operation"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["operation"]: g for g in body["groups"]}
        self.assertIn("invoke_agent", groups)

    def test_aggregate_by_decision(self):
        items = [
            {**self._agent_item("A1", "t1"), "decision": "allow"},
            {**self._agent_item("A1", "t2"), "decision": "allow"},
            {**self._agent_item("A2", "t3"), "decision": "block"},
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_decision"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["decision"]: g for g in body["groups"]}
        self.assertEqual(groups["allow"]["count"], 2)
        self.assertEqual(groups["block"]["count"], 1)

    def test_aggregate_by_hour(self):
        items = [
            self._agent_item("A1", "t1", ts="2024-01-15T10:05:00.000000"),
            self._agent_item("A1", "t2", ts="2024-01-15T10:55:00.000000"),
            self._agent_item("A1", "t3", ts="2024-01-15T11:05:00.000000"),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_hour"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["hour"]: g for g in body["groups"]}
        self.assertEqual(groups["2024-01-15T10"]["count"], 2)
        self.assertEqual(groups["2024-01-15T11"]["count"], 1)

    def test_aggregate_by_day(self):
        items = [
            self._agent_item("A1", "t1", ts="2024-01-15T10:00:00.000000"),
            self._agent_item("A1", "t2", ts="2024-01-15T22:00:00.000000"),
            self._agent_item("A1", "t3", ts="2024-01-16T08:00:00.000000"),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_day"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["day"]: g for g in body["groups"]}
        self.assertEqual(groups["2024-01-15"]["count"], 2)
        self.assertEqual(groups["2024-01-16"]["count"], 1)

    def test_aggregate_returns_total_count(self):
        items = [self._agent_item("A1", f"t{i}") for i in range(5)]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_agent"}, table=tbl)
        body = json.loads(resp["body"])
        self.assertEqual(body["total_count"], 5)

    def test_aggregate_no_next_token(self):
        tbl = _mock_table(items=[self._agent_item("A1", "t1")])
        resp = _run({"operation": "invoke_agent", "aggregate": "by_agent"}, table=tbl)
        body = json.loads(resp["body"])
        self.assertNotIn("next_token", body)

    def test_aggregate_sum_prompt_tokens(self):
        items = [
            self._agent_item("A1", "t1", prompt=100),
            self._agent_item("A1", "t2", prompt=200),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_agent"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["agent_id"]: g for g in body["groups"]}
        self.assertAlmostEqual(groups["A1"]["sum_prompt_tokens"], 300.0)
        self.assertAlmostEqual(groups["A1"]["avg_prompt_tokens"], 150.0)


class TestNewFilterParams(unittest.TestCase):
    """Tests for new filter parameters: risk_tier, policy_decision, etc."""

    def _item(self, op="invoke_agent", ts="2024-01-15T10:30:00.000000", trace="t1"):
        return {
            "pk": f"OBSERVATORY#{op}", "sk": f"{ts}#{trace}",
            "trace_id": trace, "operation": op, "timestamp": ts,
            "prompt_tokens": Decimal("100"), "completion_tokens": Decimal("50"),
            "cost_usd": Decimal("0.001"), "decision": "allow",
            "risk_tier": "high", "policy_decision": "block",
            "composite_risk_level": "critical", "hallucination_risk_level": "medium",
            "is_shadow": False, "gate_blocked": True, "fallback_used": False,
        }

    def test_risk_tier_filter_expression_applied(self):
        tbl = _mock_table(items=[self._item()])
        _run({"operation": "invoke_agent", "risk_tier": "high"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        self.assertIn("FilterExpression", call_kwargs)
        values = _condition_values(call_kwargs["FilterExpression"])
        self.assertIn("high", values)

    def test_policy_decision_filter_expression_applied(self):
        tbl = _mock_table(items=[self._item()])
        _run({"operation": "invoke_agent", "policy_decision": "block"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        values = _condition_values(call_kwargs["FilterExpression"])
        self.assertIn("block", values)

    def test_composite_risk_level_filter_applied(self):
        tbl = _mock_table(items=[self._item()])
        _run({"operation": "invoke_agent", "composite_risk_level": "critical"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        values = _condition_values(call_kwargs["FilterExpression"])
        self.assertIn("critical", values)

    def test_hallucination_risk_level_filter_applied(self):
        tbl = _mock_table(items=[self._item()])
        _run({"operation": "invoke_agent", "hallucination_risk_level": "medium"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        values = _condition_values(call_kwargs["FilterExpression"])
        self.assertIn("medium", values)

    def test_is_shadow_true_filter_applied(self):
        tbl = _mock_table(items=[self._item()])
        _run({"operation": "invoke_agent", "is_shadow": "true"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        values = _condition_values(call_kwargs["FilterExpression"])
        self.assertIn(True, values)

    def test_is_shadow_false_filter_applied(self):
        tbl = _mock_table(items=[self._item()])
        _run({"operation": "invoke_agent", "is_shadow": "false"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        values = _condition_values(call_kwargs["FilterExpression"])
        self.assertIn(False, values)

    def test_gate_blocked_filter_applied(self):
        tbl = _mock_table(items=[self._item()])
        _run({"operation": "invoke_agent", "gate_blocked": "true"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        values = _condition_values(call_kwargs["FilterExpression"])
        self.assertIn(True, values)

    def test_fallback_used_filter_applied(self):
        tbl = _mock_table(items=[self._item()])
        _run({"operation": "invoke_agent", "fallback_used": "false"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        values = _condition_values(call_kwargs["FilterExpression"])
        self.assertIn(False, values)

    def test_multiple_filters_combined(self):
        tbl = _mock_table(items=[self._item()])
        _run({"operation": "invoke_agent", "risk_tier": "high", "policy_decision": "block"},
             table=tbl)
        call_kwargs = tbl.query.call_args[1]
        values = _condition_values(call_kwargs["FilterExpression"])
        self.assertIn("high", values)
        self.assertIn("block", values)


class TestNewAggregationModes(unittest.TestCase):
    """Tests for by_risk_tier, by_composite_risk_level, by_hallucination_risk_level, by_policy_decision."""

    def _item(self, trace, risk_tier="low", composite_risk_level="low",
              hallucination_risk_level="low", policy_decision="allow",
              cost="0.001", ts="2024-01-15T10:30:00.000000"):
        return {
            "pk": "OBSERVATORY#invoke_agent", "sk": f"{ts}#{trace}",
            "trace_id": trace, "operation": "invoke_agent", "timestamp": ts,
            "prompt_tokens": Decimal("100"), "completion_tokens": Decimal("50"),
            "cost_usd": Decimal(cost), "decision": "allow",
            "risk_tier": risk_tier,
            "composite_risk_level": composite_risk_level,
            "hallucination_risk_level": hallucination_risk_level,
            "policy_decision": policy_decision,
        }

    def test_aggregate_by_risk_tier(self):
        items = [
            self._item("t1", risk_tier="high"),
            self._item("t2", risk_tier="high"),
            self._item("t3", risk_tier="low"),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_risk_tier"}, table=tbl)
        self.assertEqual(resp["statusCode"], 200)
        body = json.loads(resp["body"])
        self.assertEqual(body["aggregate"], "by_risk_tier")
        groups = {g["key"]["risk_tier"]: g for g in body["groups"]}
        self.assertEqual(groups["high"]["count"], 2)
        self.assertEqual(groups["low"]["count"], 1)

    def test_aggregate_by_composite_risk_level(self):
        items = [
            self._item("t1", composite_risk_level="critical"),
            self._item("t2", composite_risk_level="moderate"),
            self._item("t3", composite_risk_level="critical"),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_composite_risk_level"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["composite_risk_level"]: g for g in body["groups"]}
        self.assertEqual(groups["critical"]["count"], 2)
        self.assertEqual(groups["moderate"]["count"], 1)

    def test_aggregate_by_hallucination_risk_level(self):
        items = [
            self._item("t1", hallucination_risk_level="high"),
            self._item("t2", hallucination_risk_level="low"),
            self._item("t3", hallucination_risk_level="high"),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_hallucination_risk_level"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["hallucination_risk_level"]: g for g in body["groups"]}
        self.assertEqual(groups["high"]["count"], 2)
        self.assertEqual(groups["low"]["count"], 1)

    def test_aggregate_by_policy_decision(self):
        items = [
            self._item("t1", policy_decision="allow"),
            self._item("t2", policy_decision="block"),
            self._item("t3", policy_decision="allow"),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_policy_decision"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["policy_decision"]: g for g in body["groups"]}
        self.assertEqual(groups["allow"]["count"], 2)
        self.assertEqual(groups["block"]["count"], 1)

    def test_invalid_aggregate_returns_400(self):
        resp = _run({"aggregate": "by_something_unknown"})
        self.assertEqual(resp["statusCode"], 400)


class TestAggregateLowLevelDynamoShape(unittest.TestCase):
    def test_by_operation_handles_attributevalue_items(self):
        items = [
            {
                "pk": {"S": "OBSERVATORY#invoke_agent"},
                "sk": {"S": "2026-03-30T19:52:24.997872#c79d1034-cb6c-45a7-af79-41996a73aad9"},
                "operation": {"S": "invoke_agent"},
                "agent_id": {"S": "04PEAR5QXH"},
                "prompt_tokens": {"N": "3698"},
                "completion_tokens": {"N": "155"},
                "cost_usd": {"N": "0.008016"},
                "decision": {"S": "allow"},
                "timestamp": {"S": "2026-03-30T19:52:24.997872"},
            },
            {
                "pk": {"S": "OBSERVATORY#invoke_model"},
                "sk": {"S": "2026-03-30T19:39:30.005896#15be86c9-dccc-4bbd-9a8c-e5c8cbf90d99"},
                "operation": {"S": "invoke_model"},
                "model_id": {"S": "amazon.nova-micro-v1:0"},
                "prompt_tokens": {"N": "23"},
                "completion_tokens": {"N": "125"},
                "cost_usd": {"N": "0.000546"},
                "decision": {"S": "allow"},
                "timestamp": {"S": "2026-03-30T19:39:30.005896"},
            },
        ]
        tbl = _mock_table(items=items)
        resp = _run({"aggregate": "by_operation"}, table=tbl)
        self.assertEqual(resp["statusCode"], 200)
        body = json.loads(resp["body"])
        groups = {g["key"]["operation"]: g for g in body["groups"]}
        self.assertIn("invoke_agent", groups)
        self.assertEqual(groups["invoke_agent"]["count"], 1)


class TestNewSortByFields(unittest.TestCase):
    """Tests for new sort_by options: composite_risk_score, hallucination_risk_score, etc."""

    def _item(self, trace, composite_risk=None, hallucination_risk=None,
              retries=None, grounding=None):
        item = {
            "pk": "OBSERVATORY#invoke_agent", "sk": f"2024-01-15T10:30:00#{trace}",
            "trace_id": trace, "operation": "invoke_agent",
            "timestamp": "2024-01-15T10:30:00.000000",
            "prompt_tokens": Decimal("100"), "completion_tokens": Decimal("50"),
            "cost_usd": Decimal("0.001"), "decision": "allow",
        }
        if composite_risk is not None:
            item["composite_risk_score"] = Decimal(str(composite_risk))
        if hallucination_risk is not None:
            item["hallucination_risk_score"] = Decimal(str(hallucination_risk))
        if retries is not None:
            item["retries"] = Decimal(str(retries))
        if grounding is not None:
            item["grounding_score"] = Decimal(str(grounding))
        return item

    def test_sort_by_composite_risk_score_desc(self):
        items = [
            self._item("low", composite_risk=0.1),
            self._item("high", composite_risk=0.9),
            self._item("mid", composite_risk=0.5),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"sort_by": "composite_risk_score", "sort_order": "desc", "operation": "invoke_agent"},
                    table=tbl)
        body = json.loads(resp["body"])
        scores = [i.get("composite_risk_score") for i in body["items"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_sort_by_hallucination_risk_score_asc(self):
        items = [
            self._item("a", hallucination_risk=0.8),
            self._item("b", hallucination_risk=0.2),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"sort_by": "hallucination_risk_score", "sort_order": "asc", "operation": "invoke_agent"},
                    table=tbl)
        body = json.loads(resp["body"])
        scores = [i.get("hallucination_risk_score") for i in body["items"]]
        self.assertEqual(scores, sorted(scores))

    def test_sort_by_retries_desc(self):
        items = [
            self._item("a", retries=0),
            self._item("b", retries=3),
            self._item("c", retries=1),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"sort_by": "retries", "sort_order": "desc", "operation": "invoke_agent"},
                    table=tbl)
        body = json.loads(resp["body"])
        retries_vals = [i.get("retries") for i in body["items"]]
        self.assertEqual(retries_vals, sorted(retries_vals, reverse=True))

    def test_invalid_sort_by_returns_400(self):
        resp = _run({"sort_by": "unknown_field"})
        self.assertEqual(resp["statusCode"], 400)


class TestNewNumericFieldsInAggregation(unittest.TestCase):
    """New TraceContext numeric fields are included in aggregation sums/avgs."""

    def _item(self, trace, composite_risk=0.5, retries=1, grounding=0.8):
        return {
            "pk": "OBSERVATORY#invoke_agent", "sk": f"2024-01-15T10:30:00#{trace}",
            "trace_id": trace, "operation": "invoke_agent",
            "timestamp": "2024-01-15T10:30:00.000000",
            "prompt_tokens": Decimal("100"), "completion_tokens": Decimal("50"),
            "cost_usd": Decimal("0.001"), "decision": "allow",
            "agent_id": "A1",
            "composite_risk_score": Decimal(str(composite_risk)),
            "retries": Decimal(str(retries)),
            "grounding_score": Decimal(str(grounding)),
        }

    def test_aggregate_includes_composite_risk_score(self):
        items = [
            self._item("t1", composite_risk=0.4),
            self._item("t2", composite_risk=0.6),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_agent"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["agent_id"]: g for g in body["groups"]}
        self.assertAlmostEqual(groups["A1"]["sum_composite_risk_score"], 1.0, places=5)
        self.assertAlmostEqual(groups["A1"]["avg_composite_risk_score"], 0.5, places=5)

    def test_aggregate_includes_retries(self):
        items = [
            self._item("t1", retries=2),
            self._item("t2", retries=4),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_agent"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["agent_id"]: g for g in body["groups"]}
        self.assertAlmostEqual(groups["A1"]["sum_retries"], 6.0, places=5)
        self.assertAlmostEqual(groups["A1"]["avg_retries"], 3.0, places=5)

    def test_aggregate_includes_grounding_score(self):
        items = [
            self._item("t1", grounding=0.9),
            self._item("t2", grounding=0.7),
        ]
        tbl = _mock_table(items=items)
        resp = _run({"operation": "invoke_agent", "aggregate": "by_agent"}, table=tbl)
        body = json.loads(resp["body"])
        groups = {g["key"]["agent_id"]: g for g in body["groups"]}
        self.assertIn("sum_grounding_score", groups["A1"])
        self.assertAlmostEqual(groups["A1"]["sum_grounding_score"], 1.6, places=5)


class TestTimestampParsing(unittest.TestCase):
    def test_iso_timestamp_passed_through(self):
        m = _load()
        result = m._parse_timestamp("2024-06-15T12:00:00")
        self.assertIn("2024-06-15", result)

    def test_unix_epoch_converted_to_iso(self):
        m = _load()
        result = m._parse_timestamp("1718445600")  # ~2024-06-15
        self.assertIn("2024", result)
        self.assertIn("T", result)


class TestTokenRoundtrip(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        m = _load()
        key = {"pk": "OBSERVATORY#invoke_agent", "sk": "2024-01-01T00:00:00#abc"}
        encoded = m._encode_next_token(key)
        decoded = m._decode_next_token(encoded)
        self.assertEqual(decoded["pk"], key["pk"])
        self.assertEqual(decoded["sk"], key["sk"])

    def test_invalid_token_returns_none(self):
        m = _load()
        result = m._decode_next_token("not-valid-base64!!!")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
