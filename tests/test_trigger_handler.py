import importlib
import io
import json
import os
import sys
import types
import unittest


class _FakeSfnClient:
    def __init__(self):
        self.async_calls = []
        self.last_execution_arn = None

    def start_execution(self, **kwargs):
        self.async_calls.append(kwargs)
        return {"executionArn": "arn:aws:states:region:acct:execution:sm:id"}

    def describe_execution(self, **kwargs):
        self.last_execution_arn = kwargs.get("executionArn")
        return {"status": "RUNNING"}


class _FakeLambdaClient:
    """Simulates a successful synchronous Lambda invoke returning an empty list."""

    def invoke(self, **kwargs):
        payload = json.dumps({"statusCode": 200, "body": json.dumps({"items": []})})
        return {"StatusCode": 200, "Payload": io.BytesIO(payload.encode())}


class TriggerHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake_sfn = _FakeSfnClient()
        cls.fake_lambda = _FakeLambdaClient()

        if "boto3" not in sys.modules:
            fake_boto3 = types.ModuleType("boto3")

            def _client(service, **_kw):
                if service == "lambda":
                    return cls.fake_lambda
                return cls.fake_sfn

            fake_boto3.client = _client
            fake_boto3.resource = lambda *_args, **_kwargs: object()
            sys.modules["boto3"] = fake_boto3

            fake_boto3_dynamodb = types.ModuleType("boto3.dynamodb")
            fake_boto3_dynamodb_conditions = types.ModuleType("boto3.dynamodb.conditions")

            class _FakeKey:
                def __init__(self, *_args, **_kwargs):
                    pass

                def eq(self, *_args, **_kwargs):
                    return None

            fake_boto3_dynamodb_conditions.Key = _FakeKey
            sys.modules["boto3.dynamodb"] = fake_boto3_dynamodb
            sys.modules["boto3.dynamodb.conditions"] = fake_boto3_dynamodb_conditions

        if "botocore.exceptions" not in sys.modules:
            fake_botocore_exceptions = types.ModuleType("botocore.exceptions")

            class _FakeClientError(Exception):
                def __init__(self, response, operation_name):
                    super().__init__(operation_name)
                    self.response = response

            fake_botocore_exceptions.ClientError = _FakeClientError
            sys.modules["botocore.exceptions"] = fake_botocore_exceptions

        os.environ["STATE_MACHINE_ARN"] = "arn:aws:states:region:acct:stateMachine:sm"
        os.environ["PROVISION_FUNCTION_NAME"] = "fake-provision-fn"
        cls.trigger_handler = importlib.import_module("src.orchestrator.trigger_handler")
        cls.trigger_handler.sfn = cls.fake_sfn
        cls.trigger_handler.lambda_client = cls.fake_lambda

    def setUp(self):
        self.fake_sfn.async_calls.clear()

    def _assert_has_cors(self, response):
        headers = response.get("headers", {})
        self.assertIn("access-control-allow-origin", headers, "CORS origin header missing")
        self.assertEqual(headers["access-control-allow-origin"], "*")

    # ── GET routes (synchronous proxy to ProvisionTeamFunction) ──────────────

    def test_get_agents_returns_200_with_cors(self):
        event = {"httpMethod": "GET", "path": "/agents", "queryStringParameters": {}}
        response = self.trigger_handler.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self._assert_has_cors(response)
        self.assertEqual(len(self.fake_sfn.async_calls), 0)

    def test_get_teams_returns_200_with_cors(self):
        event = {"httpMethod": "GET", "path": "/teams", "queryStringParameters": {}}
        response = self.trigger_handler.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self._assert_has_cors(response)

    def test_get_roles_returns_200_with_cors(self):
        event = {"httpMethod": "GET", "path": "/roles", "queryStringParameters": {}}
        response = self.trigger_handler.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self._assert_has_cors(response)

    def test_get_departments_returns_200_with_cors(self):
        event = {"httpMethod": "GET", "path": "/departments", "queryStringParameters": {}}
        response = self.trigger_handler.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self._assert_has_cors(response)

    # ── POST/PUT/DELETE routes (async via Step Functions) ────────────────────

    def test_post_agents_is_async_with_cors(self):
        event = {
            "httpMethod": "POST",
            "path": "/agents",
            "body": json.dumps({"name": "agent"}),
        }
        response = self.trigger_handler.handler(event, None)
        self.assertEqual(response["statusCode"], 202)
        body = json.loads(response["body"])
        self.assertIn("run_id", body)
        self._assert_has_cors(response)
        self.assertEqual(len(self.fake_sfn.async_calls), 1)
        payload = json.loads(self.fake_sfn.async_calls[0]["input"])
        self.assertEqual(payload["run_id"], body["run_id"])

    def test_post_teams_is_async_with_cors(self):
        event = {"httpMethod": "POST", "path": "/teams", "body": json.dumps({"name": "my-team"})}
        response = self.trigger_handler.handler(event, None)
        self.assertEqual(response["statusCode"], 202)
        self._assert_has_cors(response)
        self.assertEqual(len(self.fake_sfn.async_calls), 1)

    def test_post_roles_is_async_with_cors(self):
        event = {"httpMethod": "POST", "path": "/roles", "body": json.dumps({"role_id": "r1"})}
        response = self.trigger_handler.handler(event, None)
        self.assertEqual(response["statusCode"], 202)
        self._assert_has_cors(response)

    def test_post_departments_is_async_with_cors(self):
        event = {"httpMethod": "POST", "path": "/departments", "body": json.dumps({"department_id": "d1"})}
        response = self.trigger_handler.handler(event, None)
        self.assertEqual(response["statusCode"], 202)
        self._assert_has_cors(response)

    # ── OPTIONS preflight ────────────────────────────────────────────────────

    def test_options_returns_cors_for_agents(self):
        event = {"httpMethod": "OPTIONS", "path": "/agents"}
        response = self.trigger_handler.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self._assert_has_cors(response)

    def test_options_returns_cors_for_teams(self):
        event = {"httpMethod": "OPTIONS", "path": "/teams"}
        response = self.trigger_handler.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self._assert_has_cors(response)

    def test_options_returns_cors_for_roles(self):
        event = {"httpMethod": "OPTIONS", "path": "/roles"}
        response = self.trigger_handler.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self._assert_has_cors(response)

    def test_options_returns_cors_for_departments(self):
        event = {"httpMethod": "OPTIONS", "path": "/departments"}
        response = self.trigger_handler.handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self._assert_has_cors(response)


if __name__ == "__main__":
    unittest.main()
