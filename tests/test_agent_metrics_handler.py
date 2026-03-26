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
        # Should have been called twice (once per pk)
        self.assertEqual(tbl.query.call_count, 2)
        all_values = []
        for c in tbl.query.call_args_list:
            all_values.extend(_condition_values(c[1]["KeyConditionExpression"]))
        self.assertIn("OBSERVATORY#invoke_agent", all_values)
        self.assertIn("OBSERVATORY#invoke_model", all_values)

    def test_agent_id_filter_uses_gsi(self):
        tbl = _mock_table(items=[self._item("invoke_agent")])
        _run({"agent_id": "AGENT-123"}, table=tbl)
        call_kwargs = tbl.query.call_args[1]
        self.assertEqual(call_kwargs.get("IndexName"), "AgentIdTimestampIndex")
        values = _condition_values(call_kwargs["KeyConditionExpression"])
        self.assertIn("AGENT-123", values)

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
