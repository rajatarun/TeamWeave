"""Unit tests for src/orchestrator/dpo_collector.py"""
import importlib
import json
import os
import unittest
from unittest.mock import MagicMock, patch


def _setenv(**kwargs):
    """Context manager helper: patch.dict(os.environ, ...) with clear safety."""
    return patch.dict(os.environ, kwargs)


def _clearenv(*keys):
    """Remove keys from os.environ safely (for tests that need them absent)."""
    for k in keys:
        os.environ.pop(k, None)


class DpoCollectorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
        cls.dpo = importlib.import_module("src.orchestrator.dpo_collector")

    def setUp(self):
        # Reset module-level S3 client singleton between tests
        self.dpo._s3_client = None
        # Clear DPO env vars so each test starts clean
        _clearenv("DPO_TRAINING_BUCKET", "DPO_DELTA_THRESHOLD")

    # ── dpo_bucket / dpo_delta_threshold ─────────────────────────────────────

    def test_dpo_bucket_empty_when_env_not_set(self):
        self.assertEqual(self.dpo.dpo_bucket(), "")

    def test_dpo_bucket_returns_env_value(self):
        with _setenv(DPO_TRAINING_BUCKET="my-dpo-bucket"):
            self.assertEqual(self.dpo.dpo_bucket(), "my-dpo-bucket")

    def test_dpo_delta_threshold_default(self):
        self.assertAlmostEqual(self.dpo.dpo_delta_threshold(), 0.4)

    def test_dpo_delta_threshold_from_env(self):
        with _setenv(DPO_DELTA_THRESHOLD="0.25"):
            self.assertAlmostEqual(self.dpo.dpo_delta_threshold(), 0.25)

    def test_dpo_delta_threshold_invalid_env_returns_default(self):
        with _setenv(DPO_DELTA_THRESHOLD="not-a-float"):
            self.assertAlmostEqual(self.dpo.dpo_delta_threshold(), 0.4)

    # ── collect_dpo_step — best response selection ────────────────────────────

    def _run_collect(self, score_a, score_b, bucket="test-bucket", threshold="0.4"):
        """Helper: run collect_dpo_step with two invocations returning given scores."""
        def invoke(session_id):
            if "dpo-a" in session_id:
                score = score_a
            else:
                score = score_b
            metrics = {"composite_risk_score": score} if score is not None else {}
            return f"response-{'A' if 'dpo-a' in session_id else 'B'}", metrics

        mock_s3 = MagicMock()
        env = {"DPO_DELTA_THRESHOLD": threshold}
        if bucket:
            env["DPO_TRAINING_BUCKET"] = bucket

        with patch.dict(os.environ, env):
            with patch.object(self.dpo, "_get_s3", return_value=mock_s3):
                result = self.dpo.collect_dpo_step(
                    invoke,
                    team="myteam", step_id="step1", run_id="run-xyz",
                    prompt="prompt", context={},
                    session_id_a="run-xyz-step1-dpo-a",
                    session_id_b="run-xyz-step1-dpo-b",
                )
        return result, mock_s3

    def test_returns_a_when_score_a_lower(self):
        result, _ = self._run_collect(score_a=0.2, score_b=0.8)
        self.assertEqual(result, "response-A")

    def test_returns_b_when_score_b_lower(self):
        result, _ = self._run_collect(score_a=0.9, score_b=0.1)
        self.assertEqual(result, "response-B")

    def test_returns_a_when_scores_equal(self):
        result, _ = self._run_collect(score_a=0.5, score_b=0.5)
        self.assertEqual(result, "response-A")

    def test_returns_a_when_both_scores_none(self):
        result, mock_s3 = self._run_collect(score_a=None, score_b=None)
        self.assertEqual(result, "response-A")
        mock_s3.put_object.assert_not_called()

    def test_none_score_loses_to_real_score_b(self):
        """None composite_risk_score is treated as infinity; real score wins."""
        result, _ = self._run_collect(score_a=None, score_b=0.9)
        self.assertEqual(result, "response-B")

    def test_none_score_loses_to_real_score_a(self):
        result, _ = self._run_collect(score_a=0.3, score_b=None)
        self.assertEqual(result, "response-A")

    # ── collect_dpo_step — upload gating ─────────────────────────────────────

    def test_no_upload_when_delta_below_threshold(self):
        # delta = 0.3 < 0.4 threshold → no upload
        _, mock_s3 = self._run_collect(score_a=0.2, score_b=0.5)
        mock_s3.put_object.assert_not_called()

    def test_no_upload_when_delta_equals_threshold(self):
        # delta == threshold: the condition is >=, so this IS uploaded
        # (boundary: "if delta >= threshold")
        _, mock_s3 = self._run_collect(score_a=0.0, score_b=0.4)
        mock_s3.put_object.assert_called_once()

    def test_upload_when_delta_exceeds_threshold(self):
        _, mock_s3 = self._run_collect(score_a=0.1, score_b=0.8)
        mock_s3.put_object.assert_called_once()

    def test_no_upload_when_bucket_not_set(self):
        _, mock_s3 = self._run_collect(score_a=0.1, score_b=0.9, bucket="")
        mock_s3.put_object.assert_not_called()

    # ── collect_dpo_step — S3 record content ─────────────────────────────────

    def test_upload_record_content(self):
        def invoke(session_id):
            if "dpo-a" in session_id:
                return "chosen-text", {"composite_risk_score": 0.1, "trace_id": "t1"}
            return "rejected-text", {"composite_risk_score": 0.8, "trace_id": "t2"}

        mock_s3 = MagicMock()
        with _setenv(DPO_TRAINING_BUCKET="mybucket", DPO_DELTA_THRESHOLD="0.4"):
            with patch.object(self.dpo, "_get_s3", return_value=mock_s3):
                self.dpo.collect_dpo_step(
                    invoke,
                    team="myteam", step_id="mystep", run_id="myrun",
                    prompt="test prompt", context={"k": "v"},
                    session_id_a="myrun-mystep-dpo-a",
                    session_id_b="myrun-mystep-dpo-b",
                )

        call_kwargs = mock_s3.put_object.call_args.kwargs
        self.assertEqual(call_kwargs["Bucket"], "mybucket")
        self.assertIn("myteam/mystep/myrun/dpo_", call_kwargs["Key"])
        self.assertEqual(call_kwargs["ContentType"], "application/json")

        record = json.loads(call_kwargs["Body"].decode("utf-8"))
        self.assertEqual(record["schema_version"], "dpo-v1")
        self.assertEqual(record["team"], "myteam")
        self.assertEqual(record["step_id"], "mystep")
        self.assertEqual(record["run_id"], "myrun")
        self.assertEqual(record["prompt"], "test prompt")
        self.assertEqual(record["context"], {"k": "v"})
        self.assertEqual(record["chosen"], "chosen-text")
        self.assertEqual(record["rejected"], "rejected-text")
        self.assertAlmostEqual(record["chosen_composite_score"], 0.1)
        self.assertAlmostEqual(record["rejected_composite_score"], 0.8)
        self.assertAlmostEqual(record["delta"], 0.7, places=5)
        self.assertIn("timestamp", record)
        self.assertIn("metrics_a", record)
        self.assertIn("metrics_b", record)

    def test_upload_key_uses_team_step_run(self):
        def invoke(session_id):
            score = 0.1 if "dpo-a" in session_id else 0.9
            return "text", {"composite_risk_score": score}

        mock_s3 = MagicMock()
        with _setenv(DPO_TRAINING_BUCKET="b", DPO_DELTA_THRESHOLD="0.4"):
            with patch.object(self.dpo, "_get_s3", return_value=mock_s3):
                self.dpo.collect_dpo_step(
                    invoke,
                    team="teamA", step_id="stepX", run_id="runZ",
                    prompt="p", context={},
                    session_id_a="runZ-stepX-dpo-a",
                    session_id_b="runZ-stepX-dpo-b",
                )

        key = mock_s3.put_object.call_args.kwargs["Key"]
        self.assertTrue(key.startswith("teamA/stepX/runZ/dpo_"))

    # ── collect_dpo_step — error handling ────────────────────────────────────

    def test_propagates_invocation_a_failure(self):
        def invoke(session_id):
            if "dpo-a" in session_id:
                raise RuntimeError("primary invocation failed")
            return "response-B", {"composite_risk_score": 0.3}

        with _setenv(DPO_TRAINING_BUCKET="bucket"):
            with patch.object(self.dpo, "_get_s3", return_value=MagicMock()):
                with self.assertRaises(RuntimeError):
                    self.dpo.collect_dpo_step(
                        invoke,
                        team="t", step_id="s", run_id="r",
                        prompt="p", context={},
                        session_id_a="r-s-dpo-a", session_id_b="r-s-dpo-b",
                    )

    def test_falls_back_to_a_when_invocation_b_fails(self):
        def invoke(session_id):
            if "dpo-a" in session_id:
                return "response-A", {"composite_risk_score": 0.5}
            raise RuntimeError("secondary invocation failed")

        with _setenv(DPO_TRAINING_BUCKET="bucket"):
            with patch.object(self.dpo, "_get_s3", return_value=MagicMock()):
                result = self.dpo.collect_dpo_step(
                    invoke,
                    team="t", step_id="s", run_id="r",
                    prompt="p", context={},
                    session_id_a="r-s-dpo-a", session_id_b="r-s-dpo-b",
                )
        self.assertEqual(result, "response-A")

    def test_upload_failure_does_not_propagate(self):
        def invoke(session_id):
            score = 0.1 if "dpo-a" in session_id else 0.9
            return "text", {"composite_risk_score": score}

        bad_s3 = MagicMock()
        bad_s3.put_object.side_effect = Exception("S3 network error")

        with _setenv(DPO_TRAINING_BUCKET="bucket", DPO_DELTA_THRESHOLD="0.4"):
            with patch.object(self.dpo, "_get_s3", return_value=bad_s3):
                result = self.dpo.collect_dpo_step(
                    invoke,
                    team="t", step_id="s", run_id="r",
                    prompt="p", context={},
                    session_id_a="r-s-dpo-a", session_id_b="r-s-dpo-b",
                )
        # Pipeline should not be disrupted by upload failure
        self.assertEqual(result, "text")

    # ── _get_plain_span_metrics integration (via mcp_observatory) ────────────

    def test_collect_handles_missing_composite_score_gracefully(self):
        """When metrics_a has no composite_risk_score, use B if it has one."""
        def invoke(session_id):
            if "dpo-a" in session_id:
                return "response-A", {}  # no composite_risk_score
            return "response-B", {"composite_risk_score": 0.7}

        mock_s3 = MagicMock()
        with _setenv(DPO_TRAINING_BUCKET="bucket", DPO_DELTA_THRESHOLD="0.4"):
            with patch.object(self.dpo, "_get_s3", return_value=mock_s3):
                result = self.dpo.collect_dpo_step(
                    invoke,
                    team="t", step_id="s", run_id="r",
                    prompt="p", context={},
                    session_id_a="r-s-dpo-a", session_id_b="r-s-dpo-b",
                )
        # B has real score (0.7), A has inf — B wins
        self.assertEqual(result, "response-B")


if __name__ == "__main__":
    unittest.main()
