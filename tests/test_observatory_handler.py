import importlib
import json
import os
import unittest
from unittest.mock import MagicMock, patch


def _load():
    return importlib.import_module("src.orchestrator.observatory_handler")


def _event(params=None):
    return {"queryStringParameters": params or {}}


class ObservatoryHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_missing_query_param_returns_400(self):
        response = self.m.handler(_event(), None)
        self.assertEqual(response["statusCode"], 400)
        body = json.loads(response["body"])
        self.assertIn("error", body)

    def test_empty_query_string_parameters_returns_400(self):
        response = self.m.handler({"queryStringParameters": None}, None)
        self.assertEqual(response["statusCode"], 400)

    # ------------------------------------------------------------------
    # Instant query URL construction
    # ------------------------------------------------------------------

    def _mock_amp_get(self, status=200, body='{"status":"success"}'):
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.text = body
        return mock_resp

    def _run_handler(self, params, mock_get_return=None):
        env = {"AMP_WORKSPACE_ID": "ws-test", "AMP_REGION": "eu-west-1"}
        if mock_get_return is None:
            mock_get_return = self._mock_amp_get()
        with patch.dict(os.environ, env):
            with patch("requests.get", return_value=mock_get_return) as mock_get:
                with patch("boto3.session.Session"):
                    with patch("botocore.auth.SigV4Auth") as mock_signer:
                        mock_signer.return_value.add_auth.return_value = None
                        response = self.m.handler(_event(params), None)
                        return response, mock_get

    def test_instant_query_calls_query_endpoint(self):
        _, mock_get = self._run_handler({"query": "up"})
        call_url = mock_get.call_args[0][0]
        self.assertIn("/api/v1/query", call_url)
        self.assertNotIn("query_range", call_url)

    def test_instant_query_includes_promql_in_url(self):
        _, mock_get = self._run_handler({"query": "teamweave_bedrock_requests_total"})
        call_url = mock_get.call_args[0][0]
        self.assertIn("teamweave_bedrock_requests_total", call_url)

    def test_instant_query_includes_optional_time(self):
        _, mock_get = self._run_handler({"query": "up", "time": "1700000000"})
        call_url = mock_get.call_args[0][0]
        self.assertIn("1700000000", call_url)

    def test_instant_query_url_contains_workspace_and_region(self):
        _, mock_get = self._run_handler({"query": "up"})
        call_url = mock_get.call_args[0][0]
        self.assertIn("ws-test", call_url)
        self.assertIn("eu-west-1", call_url)

    # ------------------------------------------------------------------
    # Range query URL construction
    # ------------------------------------------------------------------

    def test_range_query_calls_query_range_endpoint(self):
        _, mock_get = self._run_handler(
            {"query": "up", "start": "1700000000", "end": "1700003600"}
        )
        call_url = mock_get.call_args[0][0]
        self.assertIn("/api/v1/query_range", call_url)

    def test_range_query_includes_start_end_step(self):
        _, mock_get = self._run_handler(
            {"query": "up", "start": "1700000000", "end": "1700003600", "step": "30s"}
        )
        call_url = mock_get.call_args[0][0]
        self.assertIn("1700000000", call_url)
        self.assertIn("1700003600", call_url)
        self.assertIn("30s", call_url)

    def test_range_query_defaults_step_to_60s(self):
        _, mock_get = self._run_handler(
            {"query": "up", "start": "1700000000", "end": "1700003600"}
        )
        call_url = mock_get.call_args[0][0]
        self.assertIn("60s", call_url)

    # ------------------------------------------------------------------
    # Response propagation
    # ------------------------------------------------------------------

    def test_returns_amp_status_code(self):
        response, _ = self._run_handler(
            {"query": "up"}, self._mock_amp_get(status=200, body='{"status":"success"}')
        )
        self.assertEqual(response["statusCode"], 200)

    def test_propagates_amp_error_status(self):
        response, _ = self._run_handler(
            {"query": "bad{query"},
            self._mock_amp_get(status=400, body='{"status":"error","error":"parse error"}'),
        )
        self.assertEqual(response["statusCode"], 400)

    def test_returns_amp_body_verbatim(self):
        amp_body = '{"status":"success","data":{"resultType":"vector","result":[]}}'
        response, _ = self._run_handler(
            {"query": "up"}, self._mock_amp_get(body=amp_body)
        )
        self.assertEqual(response["body"], amp_body)

    def test_content_type_header_is_json(self):
        response, _ = self._run_handler({"query": "up"})
        self.assertEqual(response["headers"]["Content-Type"], "application/json")

    # ------------------------------------------------------------------
    # SigV4 signing
    # ------------------------------------------------------------------

    def test_signs_request_with_aps_service(self):
        env = {"AMP_WORKSPACE_ID": "ws-test", "AMP_REGION": "us-east-1"}
        with patch.dict(os.environ, env):
            with patch("requests.get", return_value=self._mock_amp_get()):
                with patch("boto3.session.Session"):
                    with patch("botocore.auth.SigV4Auth") as mock_signer:
                        mock_signer.return_value.add_auth.return_value = None
                        self.m.handler(_event({"query": "up"}), None)

                args = mock_signer.call_args[0]
                self.assertEqual(args[1], "aps")

    def test_signed_headers_passed_to_requests_get(self):
        env = {"AMP_WORKSPACE_ID": "ws-test", "AMP_REGION": "us-east-1"}
        with patch.dict(os.environ, env):
            with patch("requests.get", return_value=self._mock_amp_get()) as mock_get:
                with patch("boto3.session.Session"):
                    with patch("botocore.auth.SigV4Auth") as mock_signer:
                        def add_auth(req):
                            req.headers["Authorization"] = "AWS4-signed"

                        mock_signer.return_value.add_auth.side_effect = add_auth
                        self.m.handler(_event({"query": "up"}), None)

                call_kwargs = mock_get.call_args[1]
                self.assertIn("headers", call_kwargs)
                self.assertIn("Authorization", call_kwargs["headers"])


if __name__ == "__main__":
    unittest.main()
