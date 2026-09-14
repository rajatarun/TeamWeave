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
        # These stubs used to be installed only `if "boto3" not in sys.modules`,
        # which made the whole class order-dependent: several other test modules
        # put their own boto3 stand-in in sys.modules first, status_handler
        # imported *that* at module scope, and `sfn` came out as None. Running
        # this file alone passed; running the suite failed with
        # "'NoneType' object has no attribute 'describe_execution'", which reads
        # like a handler bug and is not one.
        #
        # So the module-level client is replaced after the import rather than
        # the import being steered by a stub. Whatever boto3 the rest of the
        # suite has installed, these tests now exercise the same fake.
        if "botocore.exceptions" not in sys.modules:
            fake_botocore_exceptions = types.ModuleType("botocore.exceptions")

            class _FakeClientError(Exception):
                pass

            fake_botocore_exceptions.ClientError = _FakeClientError
            sys.modules["botocore.exceptions"] = fake_botocore_exceptions

        if "boto3" not in sys.modules:
            fake_boto3 = types.ModuleType("boto3")
            fake_boto3.client = lambda *_args, **_kwargs: _FakeSfnClient()
            sys.modules["boto3"] = fake_boto3

        cls.status_handler = importlib.import_module("src.orchestrator.status_handler")
        cls._real_sfn = cls.status_handler.sfn

    @classmethod
    def tearDownClass(cls):
        cls.status_handler.sfn = cls._real_sfn

    def setUp(self):
        # A fresh fake per test, so last_execution_arn cannot leak between them.
        self.status_handler.sfn = _FakeSfnClient()

    def test_options_request_returns_cors_without_run_id(self):
        response = self.status_handler.handler({"httpMethod": "OPTIONS"}, None)

        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(response["body"], "")
        self.assertEqual(response["headers"]["access-control-allow-methods"], "OPTIONS,GET,POST,PUT,DELETE")

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
