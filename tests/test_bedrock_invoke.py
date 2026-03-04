import importlib
import sys
import types
import unittest
from unittest.mock import patch


class _FakeClientError(Exception):
    def __init__(self, error_response, operation_name):
        super().__init__(error_response.get("Error", {}).get("Message", ""))
        self.response = error_response
        self.operation_name = operation_name


class InvokeAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "boto3" not in sys.modules:
            fake_boto3 = types.ModuleType("boto3")

            class _FakeRuntimeClient:
                def invoke_agent(self, **kwargs):
                    return {"completion": []}

            fake_boto3.client = lambda *_args, **_kwargs: _FakeRuntimeClient()
            sys.modules["boto3"] = fake_boto3

        if "botocore.config" not in sys.modules:
            fake_botocore_config = types.ModuleType("botocore.config")

            class _FakeConfig:
                def __init__(self, **_kwargs):
                    pass

            fake_botocore_config.Config = _FakeConfig
            sys.modules["botocore.config"] = fake_botocore_config

        if "botocore.exceptions" not in sys.modules:
            fake_botocore_exceptions = types.ModuleType("botocore.exceptions")
            fake_botocore_exceptions.ClientError = _FakeClientError
            sys.modules["botocore.exceptions"] = fake_botocore_exceptions

        cls.bedrock_invoke = importlib.import_module("src.orchestrator.bedrock_invoke")
        cls.StepFailed = importlib.import_module("src.orchestrator.models").StepFailed

    def test_access_denied_fails_fast_without_retries(self):
        err = _FakeClientError(
            error_response={"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            operation_name="InvokeAgent",
        )

        with patch.object(self.bedrock_invoke.brt, "invoke_agent", side_effect=err) as invoke_mock:
            with self.assertRaises(self.StepFailed) as exc:
                self.bedrock_invoke.invoke_agent("agent-123", "alias-123", "session-123", "prompt", max_retries=3)

        self.assertIn("permission/auth failure", str(exc.exception))
        self.assertIn("bedrock:InvokeAgent", str(exc.exception))
        self.assertEqual(invoke_mock.call_count, 1)

    def test_non_auth_failures_still_retry(self):
        err = RuntimeError("transient boom")

        with patch.object(self.bedrock_invoke.brt, "invoke_agent", side_effect=err) as invoke_mock:
            with self.assertRaises(self.StepFailed):
                self.bedrock_invoke.invoke_agent("agent-123", "alias-123", "session-123", "prompt", max_retries=2)

        self.assertEqual(invoke_mock.call_count, 3)


if __name__ == "__main__":
    unittest.main()
