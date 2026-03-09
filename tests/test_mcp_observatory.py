import importlib
import unittest
from unittest.mock import patch, MagicMock


class McpObservatoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observatory = importlib.import_module("src.orchestrator.mcp_observatory")

    # ------------------------------------------------------------------
    # observe_agent_request
    # ------------------------------------------------------------------

    def test_observe_agent_request_passes_through_to_bedrock_wrapper(self):
        fake_client = MagicMock()
        fake_client.invoke_agent.return_value = {"completion": []}

        with patch.object(
            self.observatory, "invoke_agent_request", return_value={"completion": []}
        ) as mock_invoke:
            result = self.observatory.observe_agent_request(
                fake_client,
                agent_id="agent-1",
                alias_id="alias-1",
                session_id="sess-1",
                input_text="hello",
            )

        mock_invoke.assert_called_once_with(
            fake_client,
            agent_id="agent-1",
            alias_id="alias-1",
            session_id="sess-1",
            input_text="hello",
        )
        self.assertEqual(result, {"completion": []})

    def test_observe_agent_request_emits_log_on_success(self):
        with patch.object(
            self.observatory, "invoke_agent_request", return_value={"completion": []}
        ):
            with patch.object(self.observatory.log, "info") as mock_log:
                self.observatory.observe_agent_request(
                    MagicMock(),
                    agent_id="agent-1",
                    alias_id="alias-1",
                    session_id="sess-1",
                    input_text="hello",
                )

        mock_log.assert_called_once()
        _msg, kwargs = mock_log.call_args[0][0], mock_log.call_args[1]
        extra = kwargs.get("extra", {})
        self.assertEqual(extra["operation"], "invoke_agent")
        self.assertTrue(extra["success"])
        self.assertNotIn("error", extra)
        self.assertIn("duration_ms", extra)

    def test_observe_agent_request_emits_log_and_reraises_on_failure(self):
        boom = RuntimeError("bedrock unavailable")

        with patch.object(self.observatory, "invoke_agent_request", side_effect=boom):
            with patch.object(self.observatory.log, "info") as mock_log:
                with self.assertRaises(RuntimeError):
                    self.observatory.observe_agent_request(
                        MagicMock(),
                        agent_id="agent-1",
                        alias_id="alias-1",
                        session_id="sess-1",
                        input_text="hello",
                    )

        mock_log.assert_called_once()
        extra = mock_log.call_args[1].get("extra", {})
        self.assertFalse(extra["success"])
        self.assertIn("error", extra)
        self.assertIn("bedrock unavailable", extra["error"])

    # ------------------------------------------------------------------
    # observe_model_request
    # ------------------------------------------------------------------

    def test_observe_model_request_passes_through_to_bedrock_wrapper(self):
        with patch.object(
            self.observatory, "invoke_model_request", return_value={"body": b"{}"}
        ) as mock_invoke:
            result = self.observatory.observe_model_request(
                MagicMock(),
                model_id="model-1",
                body='{"prompt":"hi"}',
                content_type="application/json",
                accept="application/json",
            )

        mock_invoke.assert_called_once_with(
            unittest.mock.ANY,
            model_id="model-1",
            body='{"prompt":"hi"}',
            content_type="application/json",
            accept="application/json",
        )
        self.assertEqual(result, {"body": b"{}"})

    def test_observe_model_request_emits_log_on_success(self):
        with patch.object(
            self.observatory, "invoke_model_request", return_value={"body": b"{}"}
        ):
            with patch.object(self.observatory.log, "info") as mock_log:
                self.observatory.observe_model_request(
                    MagicMock(),
                    model_id="model-1",
                    body='{"prompt":"hi"}',
                )

        extra = mock_log.call_args[1].get("extra", {})
        self.assertEqual(extra["operation"], "invoke_model")
        self.assertTrue(extra["success"])
        self.assertIn("duration_ms", extra)

    def test_observe_model_request_emits_log_and_reraises_on_failure(self):
        boom = RuntimeError("model error")

        with patch.object(self.observatory, "invoke_model_request", side_effect=boom):
            with patch.object(self.observatory.log, "info") as mock_log:
                with self.assertRaises(RuntimeError):
                    self.observatory.observe_model_request(
                        MagicMock(),
                        model_id="model-1",
                        body="{}",
                    )

        extra = mock_log.call_args[1].get("extra", {})
        self.assertFalse(extra["success"])
        self.assertIn("model error", extra["error"])


if __name__ == "__main__":
    unittest.main()
