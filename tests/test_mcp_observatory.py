import importlib
import os
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch


def _make_wrapper_result(output, *, action="allow", reason="within_budget"):
    """Build a minimal WrapperResult-like object for use in mocks."""
    span = MagicMock()
    span.trace_id = "trace-abc"
    span.prompt_tokens = 10
    span.completion_tokens = 5
    span.cost_usd = 0.0

    decision = MagicMock()
    decision.action = action
    decision.reason = reason

    result = MagicMock()
    result.output = output
    result.span = span
    result.decision = decision
    return result


class McpObservatoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observatory = importlib.import_module("src.orchestrator.mcp_observatory")

    def setUp(self):
        # Reset cached DynamoDB table between tests.
        self.observatory._ddb_table = None

    # ------------------------------------------------------------------
    # observe_agent_request – core behaviour
    # ------------------------------------------------------------------

    def test_observe_agent_request_returns_wrapper_output(self):
        expected = {"completion": []}
        fake_result = _make_wrapper_result(expected)

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_push_metric"):
                output = self.observatory.observe_agent_request(
                    MagicMock(),
                    agent_id="agent-1",
                    alias_id="alias-1",
                    session_id="sess-1",
                    input_text="hello",
                )

        self.assertEqual(output, expected)

    def test_observe_agent_request_invokes_wrapper_with_correct_source_and_model(self):
        fake_result = _make_wrapper_result({"completion": []})
        mock_invoke = AsyncMock(return_value=fake_result)

        with patch.object(self.observatory._wrapper, "invoke", new=mock_invoke):
            with patch.object(self.observatory, "_push_metric"):
                self.observatory.observe_agent_request(
                    MagicMock(),
                    agent_id="agent-1",
                    alias_id="alias-1",
                    session_id="sess-1",
                    input_text="hello",
                )

        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args.kwargs
        self.assertEqual(call_kwargs["source"], "agent")
        self.assertEqual(call_kwargs["model"], "bedrock-agent")
        self.assertEqual(call_kwargs["prompt"], "hello")

    def test_observe_agent_request_emits_log_with_span_fields(self):
        fake_result = _make_wrapper_result({"completion": []})

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_push_metric"):
                with patch.object(self.observatory.log, "info") as mock_log:
                    self.observatory.observe_agent_request(
                        MagicMock(),
                        agent_id="agent-1",
                        alias_id="alias-1",
                        session_id="sess-1",
                        input_text="hello",
                    )

        mock_log.assert_called_once()
        extra = mock_log.call_args.kwargs["extra"]
        self.assertEqual(extra["operation"], "invoke_agent")
        self.assertEqual(extra["trace_id"], "trace-abc")
        self.assertEqual(extra["decision"], "allow")
        self.assertIn("cost_usd", extra)

    def test_observe_agent_request_propagates_exceptions(self):
        with patch.object(
            self.observatory._wrapper, "invoke", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            with self.assertRaises(RuntimeError, msg="boom"):
                self.observatory.observe_agent_request(
                    MagicMock(),
                    agent_id="agent-1",
                    alias_id="alias-1",
                    session_id="sess-1",
                    input_text="hello",
                )

    # ------------------------------------------------------------------
    # observe_agent_request – DynamoDB writes
    # ------------------------------------------------------------------

    def test_observe_agent_request_writes_metric_to_dynamodb(self):
        fake_result = _make_wrapper_result({"completion": []})
        mock_table = MagicMock()

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_get_ddb_table", return_value=mock_table):
                self.observatory.observe_agent_request(
                    MagicMock(),
                    agent_id="agent-1",
                    alias_id="alias-1",
                    session_id="sess-1",
                    input_text="hello",
                )

        mock_table.put_item.assert_called_once()
        item = mock_table.put_item.call_args.kwargs["Item"]
        self.assertEqual(item["pk"], "OBSERVATORY#invoke_agent")
        self.assertIn("trace-abc", item["sk"])
        self.assertEqual(item["trace_id"], "trace-abc")
        self.assertEqual(item["operation"], "invoke_agent")
        self.assertEqual(item["decision"], "allow")
        self.assertEqual(item["agent_id"], "agent-1")
        self.assertIn("ttl", item)
        self.assertIsInstance(item["ttl"], Decimal)
        self.assertIsInstance(item["cost_usd"], Decimal)

    def test_observe_agent_request_skips_dynamodb_when_table_not_configured(self):
        fake_result = _make_wrapper_result({"completion": []})

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("OBSERVATORY_METRICS_TABLE", None)
                self.observatory._ddb_table = None
                # Should complete without error even with no table configured.
                output = self.observatory.observe_agent_request(
                    MagicMock(),
                    agent_id="agent-1",
                    alias_id="alias-1",
                    session_id="sess-1",
                    input_text="hello",
                )
        self.assertEqual(output, {"completion": []})

    def test_observe_agent_request_swallows_dynamodb_write_errors(self):
        """A DynamoDB failure must not propagate to the caller."""
        fake_result = _make_wrapper_result({"completion": []})
        bad_table = MagicMock()
        bad_table.put_item.side_effect = Exception("network error")

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_get_ddb_table", return_value=bad_table):
                output = self.observatory.observe_agent_request(
                    MagicMock(),
                    agent_id="agent-1",
                    alias_id="alias-1",
                    session_id="sess-1",
                    input_text="hello",
                )

        self.assertEqual(output, {"completion": []})

    # ------------------------------------------------------------------
    # observe_model_request – core behaviour
    # ------------------------------------------------------------------

    def test_observe_model_request_returns_wrapper_output(self):
        expected = {"body": b"{}"}
        fake_result = _make_wrapper_result(expected)

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_push_metric"):
                output = self.observatory.observe_model_request(
                    MagicMock(),
                    model_id="amazon.nova-micro-v1:0",
                    body='{"prompt":"hi"}',
                )

        self.assertEqual(output, expected)

    def test_observe_model_request_invokes_wrapper_with_model_id(self):
        fake_result = _make_wrapper_result({"body": b"{}"})
        mock_invoke = AsyncMock(return_value=fake_result)

        with patch.object(self.observatory._wrapper, "invoke", new=mock_invoke):
            with patch.object(self.observatory, "_push_metric"):
                self.observatory.observe_model_request(
                    MagicMock(),
                    model_id="amazon.nova-micro-v1:0",
                    body='{"prompt":"hi"}',
                )

        call_kwargs = mock_invoke.call_args.kwargs
        self.assertEqual(call_kwargs["source"], "model")
        self.assertEqual(call_kwargs["model"], "amazon.nova-micro-v1:0")

    def test_observe_model_request_emits_log_with_span_fields(self):
        fake_result = _make_wrapper_result({"body": b"{}"})

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_push_metric"):
                with patch.object(self.observatory.log, "info") as mock_log:
                    self.observatory.observe_model_request(
                        MagicMock(),
                        model_id="amazon.nova-micro-v1:0",
                        body='{"prompt":"hi"}',
                    )

        extra = mock_log.call_args.kwargs["extra"]
        self.assertEqual(extra["operation"], "invoke_model")
        self.assertEqual(extra["model_id"], "amazon.nova-micro-v1:0")
        self.assertEqual(extra["trace_id"], "trace-abc")
        self.assertEqual(extra["decision"], "allow")

    def test_observe_model_request_propagates_exceptions(self):
        with patch.object(
            self.observatory._wrapper, "invoke", new=AsyncMock(side_effect=RuntimeError("model error"))
        ):
            with self.assertRaises(RuntimeError, msg="model error"):
                self.observatory.observe_model_request(
                    MagicMock(),
                    model_id="amazon.nova-micro-v1:0",
                    body="{}",
                )

    # ------------------------------------------------------------------
    # observe_model_request – DynamoDB writes
    # ------------------------------------------------------------------

    def test_observe_model_request_writes_metric_to_dynamodb(self):
        fake_result = _make_wrapper_result({"body": b"{}"})
        mock_table = MagicMock()

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_get_ddb_table", return_value=mock_table):
                self.observatory.observe_model_request(
                    MagicMock(),
                    model_id="amazon.nova-micro-v1:0",
                    body='{"prompt":"hi"}',
                )

        mock_table.put_item.assert_called_once()
        item = mock_table.put_item.call_args.kwargs["Item"]
        self.assertEqual(item["pk"], "OBSERVATORY#invoke_model")
        self.assertIn("trace-abc", item["sk"])
        self.assertEqual(item["operation"], "invoke_model")
        self.assertEqual(item["model_id"], "amazon.nova-micro-v1:0")
        self.assertIsInstance(item["ttl"], Decimal)
        self.assertIsInstance(item["cost_usd"], Decimal)

    def test_observe_model_request_swallows_dynamodb_write_errors(self):
        fake_result = _make_wrapper_result({"body": b"{}"})
        bad_table = MagicMock()
        bad_table.put_item.side_effect = Exception("throttled")

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_get_ddb_table", return_value=bad_table):
                output = self.observatory.observe_model_request(
                    MagicMock(),
                    model_id="amazon.nova-micro-v1:0",
                    body="{}",
                )

        self.assertEqual(output, {"body": b"{}"})

    # ------------------------------------------------------------------
    # _push_metric helpers
    # ------------------------------------------------------------------

    def test_push_metric_skips_when_no_env_var(self):
        os.environ.pop("OBSERVATORY_METRICS_TABLE", None)
        self.observatory._ddb_table = None
        # Should not raise even with no env var set.
        span = MagicMock(trace_id="t1", prompt_tokens=1, completion_tokens=1, cost_usd=0.0)
        decision = MagicMock(action="allow", reason="ok")
        self.observatory._push_metric("invoke_agent", span, decision, {})

    def test_push_metric_ttl_is_90_days_from_now(self):
        import time

        mock_table = MagicMock()
        os.environ["OBSERVATORY_METRICS_TABLE"] = "test-table"
        self.observatory._ddb_table = None

        with patch("boto3.resource") as mock_boto:
            mock_boto.return_value.Table.return_value = mock_table
            span = MagicMock(trace_id="t1", prompt_tokens=1, completion_tokens=1, cost_usd=0.001)
            decision = MagicMock(action="allow", reason="ok")
            before = int(time.time())
            self.observatory._push_metric("invoke_agent", span, decision, {})
            after = int(time.time())

        item = mock_table.put_item.call_args.kwargs["Item"]
        ttl_val = int(item["ttl"])
        expected_min = before + 90 * 24 * 60 * 60
        expected_max = after + 90 * 24 * 60 * 60
        self.assertGreaterEqual(ttl_val, expected_min)
        self.assertLessEqual(ttl_val, expected_max)

        os.environ.pop("OBSERVATORY_METRICS_TABLE", None)
        self.observatory._ddb_table = None


if __name__ == "__main__":
    unittest.main()
