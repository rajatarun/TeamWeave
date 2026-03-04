import importlib
import json
import os
import sys
import types
import unittest


class _FakeSfnClient:
    def __init__(self):
        self.sync_calls = []
        self.async_calls = []
        self.sync_response = {
            "status": "SUCCEEDED",
            "output": json.dumps({"statusCode": 200, "body": json.dumps({"items": ["a"]})}),
        }

    def start_sync_execution(self, **kwargs):
        self.sync_calls.append(kwargs)
        return self.sync_response

    def start_execution(self, **kwargs):
        self.async_calls.append(kwargs)
        return {"executionArn": "arn:aws:states:region:acct:execution:sm:id"}


class TriggerHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake_sfn = _FakeSfnClient()

        if "boto3" not in sys.modules:
            fake_boto3 = types.ModuleType("boto3")
            fake_boto3.client = lambda *_args, **_kwargs: cls.fake_sfn
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
        cls.trigger_handler = importlib.import_module("src.orchestrator.trigger_handler")
        cls.trigger_handler.sfn = cls.fake_sfn

    def setUp(self):
        self.fake_sfn.sync_calls.clear()
        self.fake_sfn.async_calls.clear()
        self.fake_sfn.sync_response = {
            "status": "SUCCEEDED",
            "output": json.dumps({"statusCode": 200, "body": json.dumps({"items": ["a"]})}),
        }

    def test_get_agent_routes_wait_for_sync_step_function_response(self):
        event = {
            "httpMethod": "GET",
            "path": "/agents",
            "queryStringParameters": {"team": "core"},
        }

        response = self.trigger_handler.handler(event, None)

        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(response["body"]), {"items": ["a"]})
        self.assertEqual(len(self.fake_sfn.sync_calls), 1)
        self.assertEqual(len(self.fake_sfn.async_calls), 0)

    def test_get_agent_routes_return_error_when_sync_execution_fails(self):
        self.fake_sfn.sync_response = {
            "status": "FAILED",
            "error": "BadRequest",
            "cause": "validation failed",
        }
        event = {"httpMethod": "GET", "path": "/agents"}

        response = self.trigger_handler.handler(event, None)

        self.assertEqual(response["statusCode"], 500)
        body = json.loads(response["body"])
        self.assertEqual(body["status"], "FAILED")
        self.assertEqual(body["error"], "BadRequest")
        self.assertEqual(body["cause"], "validation failed")
        self.assertEqual(len(self.fake_sfn.sync_calls), 1)
        self.assertEqual(len(self.fake_sfn.async_calls), 0)

    def test_get_agent_routes_wrap_non_json_sync_output(self):
        self.fake_sfn.sync_response = {"status": "SUCCEEDED", "output": "not-json"}
        event = {"httpMethod": "GET", "path": "/agents"}

        response = self.trigger_handler.handler(event, None)

        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(response["body"]), {"result": "not-json"})

    def test_post_agent_routes_remain_async(self):
        event = {
            "httpMethod": "POST",
            "path": "/agents",
            "body": json.dumps({"name": "agent"}),
        }

        response = self.trigger_handler.handler(event, None)

        self.assertEqual(response["statusCode"], 202)
        self.assertIn("run_id", json.loads(response["body"]))
        self.assertEqual(len(self.fake_sfn.async_calls), 1)
        self.assertEqual(len(self.fake_sfn.sync_calls), 0)


if __name__ == "__main__":
    unittest.main()
