import unittest

from src.orchestrator.bedrock_wrappers import invoke_agent_request, invoke_model_request


class _FakeClient:
    def __init__(self):
        self.kwargs = None

    def invoke_agent(self, **kwargs):
        self.kwargs = kwargs
        return {"ok": True}

    def invoke_model(self, **kwargs):
        self.kwargs = kwargs
        return {"ok": True}


class BedrockWrappersTests(unittest.TestCase):
    def test_invoke_agent_request(self):
        client = _FakeClient()

        response = invoke_agent_request(
            client,
            agent_id="agent-123",
            alias_id="alias-123",
            session_id="session-123",
            input_text="hello",
        )

        self.assertEqual(response, {"ok": True})
        self.assertEqual(
            client.kwargs,
            {
                "agentId": "agent-123",
                "agentAliasId": "alias-123",
                "sessionId": "session-123",
                "inputText": "hello",
            },
        )

    def test_invoke_model_request(self):
        client = _FakeClient()

        response = invoke_model_request(
            client,
            model_id="model-123",
            body='{"input":"hello"}',
            content_type="application/json",
            accept="application/json",
        )

        self.assertEqual(response, {"ok": True})
        self.assertEqual(
            client.kwargs,
            {
                "modelId": "model-123",
                "body": '{"input":"hello"}',
                "contentType": "application/json",
                "accept": "application/json",
            },
        )


if __name__ == "__main__":
    unittest.main()
