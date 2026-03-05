import importlib
import os
import sys
import types
import unittest


class _FakeSfnClient:
    def __init__(self):
        self.last_execution_arn = None

    def describe_execution(self, **_kwargs):
        self.last_execution_arn = _kwargs.get("executionArn")
        return {"status": "RUNNING"}


class StatusHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "boto3" not in sys.modules:
            fake_boto3 = types.ModuleType("boto3")
            fake_boto3.client = lambda *_args, **_kwargs: _FakeSfnClient()
            sys.modules["boto3"] = fake_boto3

        if "botocore.exceptions" not in sys.modules:
            fake_botocore_exceptions = types.ModuleType("botocore.exceptions")

            class _FakeClientError(Exception):
                pass

            fake_botocore_exceptions.ClientError = _FakeClientError
            sys.modules["botocore.exceptions"] = fake_botocore_exceptions

        cls.status_handler = importlib.import_module("src.orchestrator.status_handler")

    def test_options_request_returns_cors_without_run_id(self):
        response = self.status_handler.handler({"httpMethod": "OPTIONS"}, None)

        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(response["body"], "")
        self.assertEqual(response["headers"]["access-control-allow-methods"], "POST,GET,OPTIONS")

    def test_get_status_accepts_execution_arn_directly(self):
        response = self.status_handler.handler(
            {"httpMethod": "GET", "pathParameters": {"run_id": "arn:aws:states:us-east-1:123456789012:execution:my-state-machine:abc123"}},
            None,
        )

        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(self.status_handler.sfn.last_execution_arn, "arn:aws:states:us-east-1:123456789012:execution:my-state-machine:abc123")

    def test_get_status_builds_execution_arn_from_execution_id(self):
        os.environ["STATE_MACHINE_ARN"] = "arn:aws:states:us-east-1:123456789012:stateMachine:my-state-machine"
        response = self.status_handler.handler(
            {"httpMethod": "GET", "pathParameters": {"run_id": "abc123"}},
            None,
        )

        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(
            self.status_handler.sfn.last_execution_arn,
            "arn:aws:states:us-east-1:123456789012:execution:my-state-machine:abc123",
        )

    def test_get_status_returns_400_for_execution_id_without_state_machine_arn(self):
        os.environ.pop("STATE_MACHINE_ARN", None)
        response = self.status_handler.handler(
            {"httpMethod": "GET", "pathParameters": {"run_id": "abc123"}},
            None,
        )

        self.assertEqual(response["statusCode"], 400)


if __name__ == "__main__":
    unittest.main()
