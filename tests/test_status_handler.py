import importlib
import sys
import types
import unittest


class _FakeSfnClient:
    def describe_execution(self, **_kwargs):
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


if __name__ == "__main__":
    unittest.main()
