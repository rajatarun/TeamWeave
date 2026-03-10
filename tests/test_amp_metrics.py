import importlib
import os
import struct
import unittest
from unittest.mock import MagicMock, patch


def _load():
    return importlib.import_module("src.orchestrator.amp_metrics")


# ---------------------------------------------------------------------------
# Protobuf encoder unit tests
# ---------------------------------------------------------------------------


class EncodeVarintTests(unittest.TestCase):
    def setUp(self):
        self.m = _load()

    def test_zero(self):
        self.assertEqual(self.m._encode_varint(0), b"\x00")

    def test_one(self):
        self.assertEqual(self.m._encode_varint(1), b"\x01")

    def test_127(self):
        self.assertEqual(self.m._encode_varint(127), b"\x7f")

    def test_128_uses_two_bytes(self):
        result = self.m._encode_varint(128)
        self.assertEqual(len(result), 2)
        self.assertEqual(result, b"\x80\x01")

    def test_300(self):
        result = self.m._encode_varint(300)
        self.assertEqual(len(result), 2)
        # 300 = 0b100101100 → varint: 0xAC 0x02
        self.assertEqual(result, bytes([0xAC, 0x02]))

    def test_large_value(self):
        result = self.m._encode_varint(2**21)
        self.assertGreater(len(result), 2)


class EncodeLabelTests(unittest.TestCase):
    def setUp(self):
        self.m = _load()

    def test_returns_bytes(self):
        result = self.m._encode_label("__name__", "my_metric")
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    def test_starts_with_name_tag(self):
        result = self.m._encode_label("a", "b")
        # field 1 string tag = 0x0A
        self.assertEqual(result[0:1], b"\x0a")

    def test_contains_name_bytes(self):
        result = self.m._encode_label("myname", "myvalue")
        self.assertIn(b"myname", result)
        self.assertIn(b"myvalue", result)


class EncodeSampleTests(unittest.TestCase):
    def setUp(self):
        self.m = _load()

    def test_returns_bytes(self):
        result = self.m._encode_sample(1.5, 1000000)
        self.assertIsInstance(result, bytes)

    def test_starts_with_double_tag(self):
        result = self.m._encode_sample(1.0, 1)
        # field 1 double tag = 0x09
        self.assertEqual(result[0:1], b"\x09")

    def test_double_encoded_correctly(self):
        result = self.m._encode_sample(3.14, 0)
        # bytes 1-8 are the little-endian double
        val = struct.unpack("<d", result[1:9])[0]
        self.assertAlmostEqual(val, 3.14)

    def test_timestamp_tag_present(self):
        result = self.m._encode_sample(0.0, 12345)
        # After 0x09 + 8 bytes of double, field 2 varint tag = 0x10
        self.assertIn(b"\x10", result)


class EncodeTimeseriesTests(unittest.TestCase):
    def setUp(self):
        self.m = _load()

    def test_returns_bytes(self):
        labels = [("__name__", "foo"), ("op", "bar")]
        result = self.m._encode_timeseries(labels, 1.0, 1000)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    def test_labels_sorted_by_name(self):
        # Produce two series with labels in different order; bytes should be identical
        labels_a = [("z_label", "val1"), ("a_label", "val2")]
        labels_b = [("a_label", "val2"), ("z_label", "val1")]
        result_a = self.m._encode_timeseries(labels_a, 1.0, 1000)
        result_b = self.m._encode_timeseries(labels_b, 1.0, 1000)
        self.assertEqual(result_a, result_b)


class EncodeWriteRequestTests(unittest.TestCase):
    def setUp(self):
        self.m = _load()

    def test_empty_series_returns_empty_bytes(self):
        result = self.m._encode_write_request([])
        self.assertEqual(result, b"")

    def test_single_series_returns_bytes(self):
        series = [([("__name__", "foo"), ("op", "agent")], 42.0, 9999)]
        result = self.m._encode_write_request(series)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    def test_multiple_series_longer_than_single(self):
        series1 = [([("__name__", "m1")], 1.0, 1000)]
        series2 = [([("__name__", "m1")], 1.0, 1000), ([("__name__", "m2")], 2.0, 2000)]
        self.assertGreater(
            len(self.m._encode_write_request(series2)),
            len(self.m._encode_write_request(series1)),
        )

    def test_metric_name_bytes_present_in_output(self):
        series = [([("__name__", "my_special_metric")], 1.0, 1000)]
        result = self.m._encode_write_request(series)
        self.assertIn(b"my_special_metric", result)


# ---------------------------------------------------------------------------
# push_to_amp tests
# ---------------------------------------------------------------------------


class PushToAmpTests(unittest.TestCase):
    def setUp(self):
        self.m = _load()

    def test_skips_when_no_workspace_id(self):
        env = {k: v for k, v in os.environ.items() if k != "AMP_WORKSPACE_ID"}
        with patch.dict(os.environ, env, clear=True):
            with patch("requests.post") as mock_post:
                self.m.push_to_amp([([("__name__", "x")], 1.0, 1000)])
                mock_post.assert_not_called()

    def test_posts_to_correct_url(self):
        env = {"AMP_WORKSPACE_ID": "ws-abc123", "AMP_REGION": "us-west-2"}
        with patch.dict(os.environ, env):
            with patch("requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.raise_for_status.return_value = None
                mock_post.return_value = mock_resp
                with patch("boto3.session.Session") as mock_session:
                    mock_session.return_value.get_credentials.return_value.get_frozen_credentials.return_value = MagicMock()
                    with patch("botocore.auth.SigV4Auth") as mock_signer:
                        mock_signer.return_value.add_auth.return_value = None
                        with patch("snappy.compress", return_value=b"compressed"):
                            self.m.push_to_amp([([("__name__", "m")], 1.0, 1000)])

                call_url = mock_post.call_args[0][0]
                self.assertIn("us-west-2", call_url)
                self.assertIn("ws-abc123", call_url)
                self.assertIn("/api/v1/remote_write", call_url)

    def test_signs_with_aps_service(self):
        env = {"AMP_WORKSPACE_ID": "ws-xyz", "AMP_REGION": "us-east-1"}
        with patch.dict(os.environ, env):
            with patch("requests.post") as mock_post:
                mock_post.return_value.raise_for_status.return_value = None
                with patch("boto3.session.Session") as mock_session:
                    mock_creds = MagicMock()
                    mock_session.return_value.get_credentials.return_value.get_frozen_credentials.return_value = mock_creds
                    with patch("botocore.auth.SigV4Auth") as mock_signer:
                        mock_signer.return_value.add_auth.return_value = None
                        with patch("snappy.compress", return_value=b"c"):
                            self.m.push_to_amp([([("__name__", "m")], 1.0, 1)])

                # SigV4Auth constructed with service="aps"
                args = mock_signer.call_args
                self.assertEqual(args[0][1], "aps")

    def test_sets_correct_headers(self):
        env = {"AMP_WORKSPACE_ID": "ws-xyz", "AMP_REGION": "us-east-1"}
        with patch.dict(os.environ, env):
            with patch("requests.post") as mock_post:
                mock_post.return_value.raise_for_status.return_value = None
                with patch("boto3.session.Session"):
                    with patch("botocore.auth.SigV4Auth") as mock_signer:
                        # Simulate add_auth adding a header to the AWSRequest
                        captured_req = {}

                        def fake_add_auth(req):
                            captured_req["req"] = req
                            req.headers["Authorization"] = "AWS4-HMAC-SHA256 ..."

                        mock_signer.return_value.add_auth.side_effect = fake_add_auth
                        with patch("snappy.compress", return_value=b"c"):
                            self.m.push_to_amp([([("__name__", "m")], 1.0, 1)])

                # Content-Type header must be set on the call
                call_headers = mock_post.call_args[1]["headers"]
                self.assertEqual(call_headers.get("Content-Type"), "application/x-protobuf")
                self.assertEqual(call_headers.get("Content-Encoding"), "snappy")


# ---------------------------------------------------------------------------
# record_agent_span tests
# ---------------------------------------------------------------------------


def _make_span(
    trace_id="trace-1",
    prompt=10,
    completion=5,
    cost=0.001,
    shadow_disagreement=None,
    shadow_variance=None,
):
    span = MagicMock()
    span.trace_id = trace_id
    span.prompt_tokens = prompt
    span.completion_tokens = completion
    span.cost_usd = cost
    span.shadow_disagreement_score = shadow_disagreement
    span.shadow_numeric_variance = shadow_variance
    return span


def _make_decision(action="allow", reason="within_budget"):
    d = MagicMock()
    d.action = action
    d.reason = reason
    return d


class RecordAgentSpanTests(unittest.TestCase):
    def setUp(self):
        self.m = _load()

    def test_skips_when_no_workspace_id(self):
        env = {k: v for k, v in os.environ.items() if k != "AMP_WORKSPACE_ID"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(self.m, "push_to_amp") as mock_push:
                self.m.record_agent_span(_make_span(), _make_decision(), {})
                mock_push.assert_not_called()

    def test_calls_push_with_series(self):
        with patch.dict(os.environ, {"AMP_WORKSPACE_ID": "ws-1"}):
            with patch.object(self.m, "push_to_amp") as mock_push:
                self.m.record_agent_span(
                    _make_span(), _make_decision(),
                    {"agent_id": "agent-1", "alias_id": "alias-1", "input_len": 42},
                )
                mock_push.assert_called_once()
                series = mock_push.call_args[0][0]
                self.assertIsInstance(series, list)
                self.assertGreater(len(series), 0)

    def test_all_base_metrics_present(self):
        with patch.dict(os.environ, {"AMP_WORKSPACE_ID": "ws-1"}):
            with patch.object(self.m, "push_to_amp") as mock_push:
                self.m.record_agent_span(
                    _make_span(), _make_decision(),
                    {"agent_id": "a1", "alias_id": "al1", "input_len": 10},
                )
                series = mock_push.call_args[0][0]
                metric_names = {
                    next(v for k, v in labels if k == "__name__")
                    for labels, _, _ in series
                }
                self.assertIn("teamweave_bedrock_prompt_tokens_total", metric_names)
                self.assertIn("teamweave_bedrock_completion_tokens_total", metric_names)
                self.assertIn("teamweave_bedrock_cost_usd_total", metric_names)
                self.assertIn("teamweave_bedrock_input_length_chars", metric_names)
                self.assertIn("teamweave_bedrock_requests_total", metric_names)

    def test_agent_id_and_trace_id_in_labels(self):
        with patch.dict(os.environ, {"AMP_WORKSPACE_ID": "ws-1"}):
            with patch.object(self.m, "push_to_amp") as mock_push:
                self.m.record_agent_span(
                    _make_span(trace_id="t-999"), _make_decision(),
                    {"agent_id": "agent-X", "alias_id": "alias-X"},
                )
                series = mock_push.call_args[0][0]
                # Every series that is NOT the requests_total should carry trace_id
                for labels, _, _ in series:
                    label_dict = dict(labels)
                    if label_dict.get("__name__") != "teamweave_bedrock_requests_total":
                        self.assertEqual(label_dict.get("agent_id"), "agent-X")
                        self.assertEqual(label_dict.get("trace_id"), "t-999")

    def test_decision_label_on_requests_total(self):
        with patch.dict(os.environ, {"AMP_WORKSPACE_ID": "ws-1"}):
            with patch.object(self.m, "push_to_amp") as mock_push:
                self.m.record_agent_span(
                    _make_span(), _make_decision(action="block"),
                    {"agent_id": "a1", "alias_id": "al1"},
                )
                series = mock_push.call_args[0][0]
                req_series = [
                    (labels, v, ts) for labels, v, ts in series
                    if dict(labels).get("__name__") == "teamweave_bedrock_requests_total"
                ]
                self.assertEqual(len(req_series), 1)
                label_dict = dict(req_series[0][0])
                self.assertEqual(label_dict["decision"], "block")

    def test_shadow_metrics_included_when_present(self):
        with patch.dict(os.environ, {"AMP_WORKSPACE_ID": "ws-1"}):
            with patch.object(self.m, "push_to_amp") as mock_push:
                self.m.record_agent_span(
                    _make_span(shadow_disagreement=0.42, shadow_variance=1.5),
                    _make_decision(),
                    {"agent_id": "a1", "alias_id": "al1"},
                )
                series = mock_push.call_args[0][0]
                metric_names = {dict(labels).get("__name__") for labels, _, _ in series}
                self.assertIn("teamweave_bedrock_shadow_disagreement_score", metric_names)
                self.assertIn("teamweave_bedrock_shadow_numeric_variance", metric_names)

    def test_shadow_metrics_omitted_when_none(self):
        with patch.dict(os.environ, {"AMP_WORKSPACE_ID": "ws-1"}):
            with patch.object(self.m, "push_to_amp") as mock_push:
                self.m.record_agent_span(
                    _make_span(shadow_disagreement=None, shadow_variance=None),
                    _make_decision(),
                    {"agent_id": "a1", "alias_id": "al1"},
                )
                series = mock_push.call_args[0][0]
                metric_names = {dict(labels).get("__name__") for labels, _, _ in series}
                self.assertNotIn("teamweave_bedrock_shadow_disagreement_score", metric_names)
                self.assertNotIn("teamweave_bedrock_shadow_numeric_variance", metric_names)

    def test_push_failure_is_swallowed(self):
        with patch.dict(os.environ, {"AMP_WORKSPACE_ID": "ws-1"}):
            with patch.object(self.m, "push_to_amp", side_effect=RuntimeError("boom")):
                # Must not raise
                self.m.record_agent_span(_make_span(), _make_decision(), {})


# ---------------------------------------------------------------------------
# record_model_span tests
# ---------------------------------------------------------------------------


class RecordModelSpanTests(unittest.TestCase):
    def setUp(self):
        self.m = _load()

    def test_skips_when_no_workspace_id(self):
        env = {k: v for k, v in os.environ.items() if k != "AMP_WORKSPACE_ID"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(self.m, "push_to_amp") as mock_push:
                self.m.record_model_span(_make_span(), _make_decision(), {})
                mock_push.assert_not_called()

    def test_uses_model_id_label(self):
        with patch.dict(os.environ, {"AMP_WORKSPACE_ID": "ws-1"}):
            with patch.object(self.m, "push_to_amp") as mock_push:
                self.m.record_model_span(
                    _make_span(trace_id="t-m1"), _make_decision(),
                    {"model_id": "amazon.titan-v2", "body_len": 512},
                )
                series = mock_push.call_args[0][0]
                for labels, _, _ in series:
                    label_dict = dict(labels)
                    if label_dict.get("__name__") != "teamweave_bedrock_requests_total":
                        self.assertEqual(label_dict.get("model_id"), "amazon.titan-v2")

    def test_operation_label_is_invoke_model(self):
        with patch.dict(os.environ, {"AMP_WORKSPACE_ID": "ws-1"}):
            with patch.object(self.m, "push_to_amp") as mock_push:
                self.m.record_model_span(_make_span(), _make_decision(), {"model_id": "m1"})
                series = mock_push.call_args[0][0]
                for labels, _, _ in series:
                    self.assertEqual(dict(labels).get("operation"), "invoke_model")

    def test_push_failure_is_swallowed(self):
        with patch.dict(os.environ, {"AMP_WORKSPACE_ID": "ws-1"}):
            with patch.object(self.m, "push_to_amp", side_effect=ValueError("oops")):
                self.m.record_model_span(_make_span(), _make_decision(), {})


if __name__ == "__main__":
    unittest.main()
