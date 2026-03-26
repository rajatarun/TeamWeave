import importlib
import os
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch


def _make_wrapper_result(output, *, action="allow", reason="within_budget",
                         shadow_disagreement=None, shadow_variance=None):
    """Build a minimal WrapperResult-like object for use in mocks."""
    span = MagicMock()
    span.trace_id = "trace-abc"
    span.prompt_tokens = 10
    span.completion_tokens = 5
    span.cost_usd = 0.0
    span.shadow_disagreement_score = shadow_disagreement
    span.shadow_numeric_variance = shadow_variance

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
        self.observatory._ddb_table = None

    # ------------------------------------------------------------------
    # observe_agent_request — core behaviour (no shadow)
    # ------------------------------------------------------------------

    def test_observe_agent_request_returns_wrapper_output(self):
        expected = {"completion": []}
        fake_result = _make_wrapper_result(expected)

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_push_metric"):
                output = self.observatory.observe_agent_request(
                    MagicMock(), agent_id="a1", alias_id="al1",
                    session_id="s1", input_text="hello",
                )
        self.assertEqual(output, expected)

    def test_observe_agent_request_no_shadow_uses_dual_invoke_false(self):
        fake_result = _make_wrapper_result({"completion": []})
        mock_invoke = AsyncMock(return_value=fake_result)

        with patch.object(self.observatory._wrapper, "invoke", new=mock_invoke):
            with patch.object(self.observatory, "_push_metric"):
                self.observatory.observe_agent_request(
                    MagicMock(), agent_id="a1", alias_id="al1",
                    session_id="s1", input_text="hello",
                )

        kw = mock_invoke.call_args.kwargs
        self.assertEqual(kw["source"], "agent")
        self.assertEqual(kw["model"], "bedrock-agent")
        self.assertFalse(kw["dual_invoke"])
        self.assertIsNone(kw["shadow_call"])

    def test_observe_agent_request_emits_log_with_span_fields(self):
        fake_result = _make_wrapper_result({"completion": []})

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_push_metric"):
                with patch.object(self.observatory.log, "info") as mock_log:
                    self.observatory.observe_agent_request(
                        MagicMock(), agent_id="a1", alias_id="al1",
                        session_id="s1", input_text="hello",
                    )

        extra = mock_log.call_args.kwargs["extra"]
        self.assertEqual(extra["operation"], "invoke_agent")
        self.assertEqual(extra["trace_id"], "trace-abc")
        self.assertEqual(extra["decision"], "allow")
        self.assertIn("cost_usd", extra)

    def test_observe_agent_request_propagates_exceptions(self):
        with patch.object(
            self.observatory._wrapper, "invoke", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            with self.assertRaises(RuntimeError):
                self.observatory.observe_agent_request(
                    MagicMock(), agent_id="a1", alias_id="al1",
                    session_id="s1", input_text="hello",
                )

    # ------------------------------------------------------------------
    # observe_agent_request — shadow alias / dual_invoke
    # ------------------------------------------------------------------

    def test_observe_agent_request_shadow_enables_dual_invoke(self):
        fake_result = _make_wrapper_result({"completion": []}, shadow_disagreement=0.12)
        mock_invoke = AsyncMock(return_value=fake_result)

        with patch.object(self.observatory._wrapper, "invoke", new=mock_invoke):
            with patch.object(self.observatory, "_push_metric"):
                self.observatory.observe_agent_request(
                    MagicMock(), agent_id="a1", alias_id="al1",
                    session_id="s1", input_text="hello",
                    shadow_alias_id="shadow-al",
                )

        kw = mock_invoke.call_args.kwargs
        self.assertTrue(kw["dual_invoke"])
        self.assertEqual(kw["shadow_model"], "bedrock-agent/shadow-al")
        self.assertIsNotNone(kw["shadow_call"])

    def test_observe_agent_request_shadow_writes_disagreement_to_dynamodb(self):
        fake_result = _make_wrapper_result(
            {"completion": []}, shadow_disagreement=0.35, shadow_variance=1.5
        )
        mock_table = MagicMock()

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_get_ddb_table", return_value=mock_table):
                self.observatory.observe_agent_request(
                    MagicMock(), agent_id="a1", alias_id="al1",
                    session_id="s1", input_text="hello",
                    shadow_alias_id="shadow-al",
                )

        item = mock_table.put_item.call_args.kwargs["Item"]
        self.assertIn("shadow_disagreement_score", item)
        self.assertAlmostEqual(float(item["shadow_disagreement_score"]), 0.35, places=4)
        self.assertIn("shadow_numeric_variance", item)
        self.assertEqual(item["shadow_alias_id"], "shadow-al")

    def test_observe_agent_request_shadow_logs_disagreement_score(self):
        fake_result = _make_wrapper_result({"completion": []}, shadow_disagreement=0.42)

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_push_metric"):
                with patch.object(self.observatory.log, "info") as mock_log:
                    self.observatory.observe_agent_request(
                        MagicMock(), agent_id="a1", alias_id="al1",
                        session_id="s1", input_text="hello",
                        shadow_alias_id="shadow-al",
                    )

        extra = mock_log.call_args.kwargs["extra"]
        self.assertEqual(extra["shadow_alias_id"], "shadow-al")
        self.assertEqual(extra["shadow_disagreement_score"], 0.42)

    def test_observe_agent_request_no_shadow_disagreement_skipped_from_dynamodb(self):
        """When no shadow, shadow fields must NOT appear in the DynamoDB item."""
        fake_result = _make_wrapper_result({"completion": []})  # shadow fields = None
        mock_table = MagicMock()

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_get_ddb_table", return_value=mock_table):
                self.observatory.observe_agent_request(
                    MagicMock(), agent_id="a1", alias_id="al1",
                    session_id="s1", input_text="hello",
                )

        item = mock_table.put_item.call_args.kwargs["Item"]
        self.assertNotIn("shadow_disagreement_score", item)
        self.assertNotIn("shadow_alias_id", item)

    # ------------------------------------------------------------------
    # observe_agent_request — DynamoDB writes
    # ------------------------------------------------------------------

    def test_observe_agent_request_writes_metric_to_dynamodb(self):
        fake_result = _make_wrapper_result({"completion": []})
        mock_table = MagicMock()

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_get_ddb_table", return_value=mock_table):
                self.observatory.observe_agent_request(
                    MagicMock(), agent_id="a1", alias_id="al1",
                    session_id="s1", input_text="hello",
                )

        item = mock_table.put_item.call_args.kwargs["Item"]
        self.assertEqual(item["pk"], "OBSERVATORY#invoke_agent")
        self.assertIn("trace-abc", item["sk"])
        self.assertEqual(item["trace_id"], "trace-abc")
        self.assertEqual(item["decision"], "allow")
        self.assertEqual(item["agent_id"], "a1")
        self.assertIsInstance(item["ttl"], Decimal)
        self.assertIsInstance(item["cost_usd"], Decimal)

    def test_observe_agent_request_skips_dynamodb_when_table_not_configured(self):
        fake_result = _make_wrapper_result({"completion": []})

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            os.environ.pop("OBSERVATORY_METRICS_TABLE", None)
            self.observatory._ddb_table = None
            output = self.observatory.observe_agent_request(
                MagicMock(), agent_id="a1", alias_id="al1",
                session_id="s1", input_text="hello",
            )
        self.assertEqual(output, {"completion": []})

    def test_observe_agent_request_swallows_dynamodb_write_errors(self):
        fake_result = _make_wrapper_result({"completion": []})
        bad_table = MagicMock()
        bad_table.put_item.side_effect = Exception("network error")

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_get_ddb_table", return_value=bad_table):
                output = self.observatory.observe_agent_request(
                    MagicMock(), agent_id="a1", alias_id="al1",
                    session_id="s1", input_text="hello",
                )
        self.assertEqual(output, {"completion": []})

    # ------------------------------------------------------------------
    # observe_model_request — core behaviour
    # ------------------------------------------------------------------

    def test_observe_model_request_returns_wrapper_output(self):
        expected = {"body": b"{}"}
        fake_result = _make_wrapper_result(expected)

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_push_metric"):
                output = self.observatory.observe_model_request(
                    MagicMock(), model_id="amazon.nova-micro-v1:0", body='{"prompt":"hi"}',
                )
        self.assertEqual(output, expected)

    def test_observe_model_request_invokes_wrapper_with_model_id(self):
        fake_result = _make_wrapper_result({"body": b"{}"})
        mock_invoke = AsyncMock(return_value=fake_result)

        with patch.object(self.observatory._wrapper, "invoke", new=mock_invoke):
            with patch.object(self.observatory, "_push_metric"):
                self.observatory.observe_model_request(
                    MagicMock(), model_id="amazon.nova-micro-v1:0", body='{"prompt":"hi"}',
                )

        kw = mock_invoke.call_args.kwargs
        self.assertEqual(kw["source"], "model")
        self.assertEqual(kw["model"], "amazon.nova-micro-v1:0")

    def test_observe_model_request_emits_log_with_span_fields(self):
        fake_result = _make_wrapper_result({"body": b"{}"})

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_push_metric"):
                with patch.object(self.observatory.log, "info") as mock_log:
                    self.observatory.observe_model_request(
                        MagicMock(), model_id="amazon.nova-micro-v1:0", body='{"prompt":"hi"}',
                    )

        extra = mock_log.call_args.kwargs["extra"]
        self.assertEqual(extra["operation"], "invoke_model")
        self.assertEqual(extra["model_id"], "amazon.nova-micro-v1:0")
        self.assertEqual(extra["trace_id"], "trace-abc")

    def test_observe_model_request_propagates_exceptions(self):
        with patch.object(
            self.observatory._wrapper, "invoke", new=AsyncMock(side_effect=RuntimeError("model error"))
        ):
            with self.assertRaises(RuntimeError):
                self.observatory.observe_model_request(
                    MagicMock(), model_id="amazon.nova-micro-v1:0", body="{}",
                )

    def test_observe_model_request_writes_metric_to_dynamodb(self):
        fake_result = _make_wrapper_result({"body": b"{}"})
        mock_table = MagicMock()

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_get_ddb_table", return_value=mock_table):
                self.observatory.observe_model_request(
                    MagicMock(), model_id="amazon.nova-micro-v1:0", body='{"prompt":"hi"}',
                )

        item = mock_table.put_item.call_args.kwargs["Item"]
        self.assertEqual(item["pk"], "OBSERVATORY#invoke_model")
        self.assertIn("trace-abc", item["sk"])
        self.assertEqual(item["model_id"], "amazon.nova-micro-v1:0")
        self.assertIsInstance(item["ttl"], Decimal)

    def test_observe_model_request_swallows_dynamodb_write_errors(self):
        fake_result = _make_wrapper_result({"body": b"{}"})
        bad_table = MagicMock()
        bad_table.put_item.side_effect = Exception("throttled")

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
            with patch.object(self.observatory, "_get_ddb_table", return_value=bad_table):
                output = self.observatory.observe_model_request(
                    MagicMock(), model_id="amazon.nova-micro-v1:0", body="{}",
                )
        self.assertEqual(output, {"body": b"{}"})

    # ------------------------------------------------------------------
    # _push_metric helpers
    # ------------------------------------------------------------------

    def test_push_metric_skips_when_no_env_var(self):
        os.environ.pop("OBSERVATORY_METRICS_TABLE", None)
        self.observatory._ddb_table = None
        span = MagicMock(
            trace_id="t1", prompt_tokens=1, completion_tokens=1, cost_usd=0.0,
            shadow_disagreement_score=None, shadow_numeric_variance=None,
        )
        decision = MagicMock(action="allow", reason="ok")
        self.observatory._push_metric("invoke_agent", span, decision, {})

    def test_push_metric_ttl_is_90_days_from_now(self):
        import time as _time
        mock_table = MagicMock()
        os.environ["OBSERVATORY_METRICS_TABLE"] = "test-table"
        self.observatory._ddb_table = None

        with patch("boto3.resource") as mock_boto:
            mock_boto.return_value.Table.return_value = mock_table
            span = MagicMock(
                trace_id="t1", prompt_tokens=1, completion_tokens=1, cost_usd=0.001,
                shadow_disagreement_score=None, shadow_numeric_variance=None,
            )
            decision = MagicMock(action="allow", reason="ok")
            before = int(_time.time())
            self.observatory._push_metric("invoke_agent", span, decision, {})
            after = int(_time.time())

        item = mock_table.put_item.call_args.kwargs["Item"]
        ttl_val = int(item["ttl"])
        self.assertGreaterEqual(ttl_val, before + 90 * 24 * 60 * 60)
        self.assertLessEqual(ttl_val, after + 90 * 24 * 60 * 60)

        os.environ.pop("OBSERVATORY_METRICS_TABLE", None)
        self.observatory._ddb_table = None

    def test_push_metric_includes_shadow_fields_when_present(self):
        mock_table = MagicMock()
        os.environ["OBSERVATORY_METRICS_TABLE"] = "test-table"
        self.observatory._ddb_table = None

        with patch("boto3.resource") as mock_boto:
            mock_boto.return_value.Table.return_value = mock_table
            span = MagicMock(
                trace_id="t1", prompt_tokens=1, completion_tokens=1, cost_usd=0.0,
                shadow_disagreement_score=0.25,
                shadow_numeric_variance=0.5,
            )
            decision = MagicMock(action="allow", reason="ok")
            self.observatory._push_metric("invoke_agent", span, decision, {
                "shadow_alias_id": "shadow-al",
            })

        item = mock_table.put_item.call_args.kwargs["Item"]
        self.assertIn("shadow_disagreement_score", item)
        self.assertAlmostEqual(float(item["shadow_disagreement_score"]), 0.25, places=4)
        self.assertIn("shadow_numeric_variance", item)
        self.assertEqual(item["shadow_alias_id"], "shadow-al")

        os.environ.pop("OBSERVATORY_METRICS_TABLE", None)
        self.observatory._ddb_table = None


class ExtractSpanFieldsTests(unittest.TestCase):
    """Tests for _extract_span_fields — TraceContext field extraction."""

    @classmethod
    def setUpClass(cls):
        cls.observatory = importlib.import_module("src.orchestrator.mcp_observatory")

    def _make_span(self, **kwargs):
        span = MagicMock()
        # Default: all optional TraceContext fields absent (sentinel)
        for field in (
            "confidence", "grounding_score", "verifier_score", "self_consistency_score",
            "numeric_variance_score", "hallucination_risk_score",
            "grounding_risk", "self_consistency_risk", "numeric_instability_risk",
            "tool_mismatch_risk", "drift_risk", "composite_risk_score",
            "retries", "prompt_size_chars", "exec_token_ttl_ms",
            "fallback_used", "is_shadow", "gate_blocked", "tool_claim_mismatch", "exec_token_verified",
            "span_id", "parent_span_id", "risk_tier", "prompt_template_id", "prompt_hash",
            "normalized_prompt_hash", "answer_hash", "hallucination_risk_level",
            "shadow_parent_trace_id", "fallback_type", "fallback_reason",
            "request_id", "method", "tool_name", "tool_args_hash", "tool_criticality",
            "policy_decision", "policy_id", "policy_version",
            "composite_risk_level", "exec_token_id", "exec_token_hash",
            "start_time", "end_time",
        ):
            setattr(span, field, None)
        for k, v in kwargs.items():
            setattr(span, k, v)
        return span

    def test_float_field_stored_as_decimal(self):
        span = self._make_span(composite_risk_score=0.75)
        result = self.observatory._extract_span_fields(span)
        from decimal import Decimal
        self.assertIn("composite_risk_score", result)
        self.assertAlmostEqual(float(result["composite_risk_score"]), 0.75, places=5)
        self.assertIsInstance(result["composite_risk_score"], Decimal)

    def test_int_field_stored_as_decimal(self):
        span = self._make_span(retries=3, prompt_size_chars=512)
        result = self.observatory._extract_span_fields(span)
        self.assertIn("retries", result)
        self.assertEqual(int(result["retries"]), 3)
        self.assertIn("prompt_size_chars", result)
        self.assertEqual(int(result["prompt_size_chars"]), 512)

    def test_int_zero_stored(self):
        span = self._make_span(retries=0)
        result = self.observatory._extract_span_fields(span)
        self.assertIn("retries", result)
        self.assertEqual(int(result["retries"]), 0)

    def test_bool_fields_stored(self):
        span = self._make_span(fallback_used=True, is_shadow=False, gate_blocked=True)
        result = self.observatory._extract_span_fields(span)
        self.assertIs(result["fallback_used"], True)
        self.assertIs(result["is_shadow"], False)
        self.assertIs(result["gate_blocked"], True)

    def test_string_fields_stored(self):
        span = self._make_span(
            risk_tier="high",
            policy_decision="block",
            composite_risk_level="critical",
            hallucination_risk_level="medium",
            span_id="span-xyz",
        )
        result = self.observatory._extract_span_fields(span)
        self.assertEqual(result["risk_tier"], "high")
        self.assertEqual(result["policy_decision"], "block")
        self.assertEqual(result["composite_risk_level"], "critical")
        self.assertEqual(result["hallucination_risk_level"], "medium")
        self.assertEqual(result["span_id"], "span-xyz")

    def test_none_optional_fields_skipped(self):
        span = self._make_span()  # all optional fields = None
        result = self.observatory._extract_span_fields(span)
        self.assertEqual(result, {})

    def test_datetime_field_serialised_to_iso(self):
        from datetime import datetime, timezone
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        span = self._make_span(start_time=dt)
        result = self.observatory._extract_span_fields(span)
        self.assertIn("start_time", result)
        self.assertIn("2024-06-15", result["start_time"])

    def test_push_metric_writes_trace_context_fields(self):
        """_push_metric persists new TraceContext fields via _extract_span_fields."""
        mock_table = MagicMock()
        os.environ["OBSERVATORY_METRICS_TABLE"] = "test-table"
        observatory = importlib.import_module("src.orchestrator.mcp_observatory")
        observatory._ddb_table = None

        with patch("boto3.resource") as mock_boto:
            mock_boto.return_value.Table.return_value = mock_table
            span = MagicMock(
                trace_id="t1", prompt_tokens=10, completion_tokens=5, cost_usd=0.001,
                shadow_disagreement_score=None, shadow_numeric_variance=None,
                composite_risk_score=0.8, risk_tier="high", retries=2,
                fallback_used=True, is_shadow=False, gate_blocked=False,
                policy_decision="block", hallucination_risk_level="high",
                # set remaining optional fields to None
                confidence=None, grounding_score=None, verifier_score=None,
                self_consistency_score=None, numeric_variance_score=None,
                hallucination_risk_score=None, grounding_risk=None,
                self_consistency_risk=None, numeric_instability_risk=None,
                tool_mismatch_risk=None, drift_risk=None,
                prompt_size_chars=200, exec_token_ttl_ms=None,
                tool_claim_mismatch=None, exec_token_verified=None,
                span_id="s1", parent_span_id=None, prompt_template_id=None,
                prompt_hash=None, normalized_prompt_hash=None, answer_hash=None,
                shadow_parent_trace_id=None, fallback_type=None, fallback_reason=None,
                request_id=None, method=None, tool_name=None, tool_args_hash=None,
                tool_criticality=None, policy_id=None, policy_version=None,
                composite_risk_level="critical", exec_token_id=None, exec_token_hash=None,
                start_time=None, end_time=None,
            )
            decision = MagicMock(action="block", reason="risk_exceeded")
            observatory._push_metric("invoke_agent", span, decision, {})

        item = mock_table.put_item.call_args.kwargs["Item"]
        self.assertAlmostEqual(float(item["composite_risk_score"]), 0.8, places=4)
        self.assertEqual(item["risk_tier"], "high")
        self.assertEqual(int(item["retries"]), 2)
        self.assertIs(item["fallback_used"], True)
        self.assertEqual(item["policy_decision"], "block")
        self.assertEqual(item["composite_risk_level"], "critical")
        self.assertEqual(item["span_id"], "s1")
        self.assertNotIn("confidence", item)  # None fields skipped

        os.environ.pop("OBSERVATORY_METRICS_TABLE", None)
        observatory._ddb_table = None

    def test_push_metric_does_not_overwrite_explicit_fields(self):
        """_extract_span_fields results never overwrite explicitly set fields."""
        mock_table = MagicMock()
        os.environ["OBSERVATORY_METRICS_TABLE"] = "test-table"
        observatory = importlib.import_module("src.orchestrator.mcp_observatory")
        observatory._ddb_table = None

        with patch("boto3.resource") as mock_boto:
            mock_boto.return_value.Table.return_value = mock_table
            # Simulate a span that also exposes trace_id, prompt_tokens etc.
            span = MagicMock(
                trace_id="explicit-trace", prompt_tokens=99, completion_tokens=99,
                cost_usd=99.0, shadow_disagreement_score=None, shadow_numeric_variance=None,
                retries=0, prompt_size_chars=0, is_shadow=False,
                fallback_used=False, gate_blocked=False,
                tool_claim_mismatch=None, exec_token_verified=None,
                # set all remaining optional fields to None
                confidence=None, grounding_score=None, verifier_score=None,
                self_consistency_score=None, numeric_variance_score=None,
                hallucination_risk_score=None, grounding_risk=None,
                self_consistency_risk=None, numeric_instability_risk=None,
                tool_mismatch_risk=None, drift_risk=None, composite_risk_score=None,
                exec_token_ttl_ms=None, span_id=None, parent_span_id=None, risk_tier=None,
                prompt_template_id=None, prompt_hash=None, normalized_prompt_hash=None,
                answer_hash=None, hallucination_risk_level=None, shadow_parent_trace_id=None,
                fallback_type=None, fallback_reason=None, request_id=None, method=None,
                tool_name=None, tool_args_hash=None, tool_criticality=None,
                policy_decision=None, policy_id=None, policy_version=None,
                composite_risk_level=None, exec_token_id=None, exec_token_hash=None,
                start_time=None, end_time=None,
            )
            decision = MagicMock(action="allow", reason="ok")
            observatory._push_metric("invoke_agent", span, decision, {})

        item = mock_table.put_item.call_args.kwargs["Item"]
        # prompt_tokens, completion_tokens, cost_usd are set explicitly from span attrs
        # They must match the explicit values, not any _extract_span_fields result
        self.assertEqual(item["trace_id"], "explicit-trace")
        self.assertEqual(int(item["prompt_tokens"]), 99)

        os.environ.pop("OBSERVATORY_METRICS_TABLE", None)
        observatory._ddb_table = None


if __name__ == "__main__":
    unittest.main()
