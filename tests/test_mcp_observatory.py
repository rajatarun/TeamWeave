import importlib
import unittest
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

    # ------------------------------------------------------------------
    # observe_agent_request
    # ------------------------------------------------------------------

    def test_observe_agent_request_returns_wrapper_output(self):
        expected = {"completion": []}
        fake_result = _make_wrapper_result(expected)

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
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
    # observe_model_request
    # ------------------------------------------------------------------

    def test_observe_model_request_returns_wrapper_output(self):
        expected = {"body": b"{}"}
        fake_result = _make_wrapper_result(expected)

        with patch.object(self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)):
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


if __name__ == "__main__":
    unittest.main()
