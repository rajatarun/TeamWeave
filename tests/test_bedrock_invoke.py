import importlib
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError as _RealClientError


class _FakeClientError(_RealClientError):
    def __init__(self, error_response, operation_name):
        super().__init__(error_response, operation_name)


class InvokeAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure a region is set so boto3 can create clients without AWS creds.
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")

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
