"""Unit tests for src/orchestrator/contextweave_client.py (HTTP stubbed)."""
import io
import json
import os
import unittest
import urllib.error
import urllib.parse
from unittest.mock import patch

from src.orchestrator import contextweave_client as cw

_URL = "https://contextweave.example.com"


def _setenv(**kwargs):
    return patch.dict(os.environ, kwargs)


class _FakeResponse(io.BytesIO):
    """Minimal stand-in for the object urlopen() yields as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _ok(payload):
    return _FakeResponse(json.dumps(payload).encode("utf-8"))


def _http_error(code, body=b"boom"):
    return urllib.error.HTTPError(_URL, code, "err", {}, io.BytesIO(body))


_SAMPLE = {
    "queryId": "q-123",
    "answer": "Tarun has built serverless RAG systems on AWS.",
    "sources": [
        {"file": "architecture.md", "excerpt": "Step Functions orchestrate ingest.", "weight": 1.0},
        {"file": "repo-signals.yaml", "excerpt": "pgvector, Neptune, Bedrock.", "weight": 0.7},
    ],
    "confidence": 0.91,
    "questionType": "architecture",
    "cacheHit": False,
}


class ConfigurationTests(unittest.TestCase):
    def test_not_configured_without_url(self):
        with _setenv(CONTEXTWEAVE_URL=""):
            self.assertFalse(cw.is_configured())

    def test_base_url_strips_trailing_slash(self):
        with _setenv(CONTEXTWEAVE_URL=_URL + "/"):
            self.assertEqual(cw.base_url(), _URL)

    def test_api_key_header_only_sent_when_set(self):
        with _setenv(CONTEXTWEAVE_API_KEY=""):
            self.assertNotIn("x-api-key", cw._headers())
        with _setenv(CONTEXTWEAVE_API_KEY="secret"):
            self.assertEqual(cw._headers()["x-api-key"], "secret")


class QueryExpertiseTests(unittest.TestCase):
    def test_posts_question_and_top_k_to_query_endpoint(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok(_SAMPLE)
        ) as mock_open:
            payload = cw.query_expertise("what has he built?", top_k=5)

        self.assertEqual(payload["queryId"], "q-123")
        req = mock_open.call_args[0][0]
        self.assertEqual(req.full_url, _URL + "/query-expertise")
        self.assertEqual(json.loads(req.data.decode("utf-8")), {"question": "what has he built?", "topK": 5})

    def test_returns_none_without_url_configured(self):
        with _setenv(CONTEXTWEAVE_URL=""), patch.object(cw.urllib.request, "urlopen") as mock_open:
            self.assertIsNone(cw.query_expertise("anything"))
        mock_open.assert_not_called()

    def test_returns_none_for_blank_question(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(cw.urllib.request, "urlopen") as mock_open:
            self.assertIsNone(cw.query_expertise("   "))
        mock_open.assert_not_called()

    def test_retries_transport_failure_then_succeeds(self):
        responses = [TimeoutError("read timed out"), _ok(_SAMPLE)]

        def _side_effect(*_a, **_kw):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", side_effect=_side_effect
        ) as mock_open:
            payload = cw.query_expertise("q")

        self.assertEqual(payload["queryId"], "q-123")
        self.assertEqual(mock_open.call_count, 2)

    def test_gives_up_after_retry_budget_and_degrades_to_none(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", side_effect=OSError("connection reset")
        ) as mock_open:
            self.assertIsNone(cw.query_expertise("q"))
        self.assertEqual(mock_open.call_count, cw._MAX_RETRIES + 1)

    def test_does_not_retry_client_error(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", side_effect=_http_error(400)
        ) as mock_open:
            self.assertIsNone(cw.query_expertise("q"))
        self.assertEqual(mock_open.call_count, 1)

    def test_retries_server_error(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", side_effect=_http_error(503)
        ) as mock_open:
            self.assertIsNone(cw.query_expertise("q"))
        self.assertEqual(mock_open.call_count, cw._MAX_RETRIES + 1)

    def test_error_body_is_treated_as_failure(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok({"error": "Internal server error"})
        ):
            self.assertIsNone(cw.query_expertise("q"))


class FeedbackTests(unittest.TestCase):
    def test_send_feedback_posts_query_id_and_rating(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok({"applied": True})
        ) as mock_open:
            self.assertTrue(cw.send_feedback("q-123", "up"))

        req = mock_open.call_args[0][0]
        self.assertEqual(req.full_url, _URL + "/feedback")
        self.assertEqual(json.loads(req.data.decode("utf-8")), {"queryId": "q-123", "rating": "up"})

    def test_send_feedback_without_query_id_is_a_noop(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(cw.urllib.request, "urlopen") as mock_open:
            self.assertFalse(cw.send_feedback(""))
        mock_open.assert_not_called()

    def test_valid_output_feedback_is_off_by_default(self):
        with _setenv(CONTEXTWEAVE_URL=_URL, CONTEXTWEAVE_FEEDBACK_ON_VALID_OUTPUT=""), patch.object(
            cw.urllib.request, "urlopen"
        ) as mock_open:
            self.assertFalse(cw.maybe_send_valid_output_feedback("q-123"))
        mock_open.assert_not_called()

    def test_valid_output_feedback_sent_when_opted_in(self):
        with _setenv(CONTEXTWEAVE_URL=_URL, CONTEXTWEAVE_FEEDBACK_ON_VALID_OUTPUT="1"), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok({"applied": True})
        ) as mock_open:
            self.assertTrue(cw.maybe_send_valid_output_feedback("q-123"))

        body = json.loads(mock_open.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(body, {"queryId": "q-123", "rating": "up"})

    def test_feedback_failure_is_not_fatal(self):
        with _setenv(CONTEXTWEAVE_URL=_URL, CONTEXTWEAVE_FEEDBACK_ON_VALID_OUTPUT="1"), patch.object(
            cw.urllib.request, "urlopen", side_effect=OSError("down")
        ):
            self.assertFalse(cw.maybe_send_valid_output_feedback("q-123"))


_HEALTH = {
    "status": "healthy",
    "knowledgeBaseId": "kb-1",
    "neptuneGraphId": "g-1",
    "environment": "dev",
    "routingGraph": {
        "documentTypeDistribution": [],
        "health": {
            "exploration": 0.2,
            "priorStrength": 2.0,
            "questionTypes": {
                "skill_depth": {
                    "verdict": "converged",
                    "leader": "graph_first",
                    "leaderPBest": 0.97,
                    "starvedArms": [],
                    "arms": {},
                }
            },
        },
    },
}

_DECISIONS = {
    "groups": [
        {
            "questionType": "skill_depth",
            "strategy": "graph_first",
            "count": 120,
            "ratedCount": 34,
            "avgConfidence": 0.81,
            "avgRating": 0.74,
            "meanAbsDiff": 0.19,
        }
    ],
    "totalCount": 512,
    "totalRated": 140,
}


class GetHealthTests(unittest.TestCase):
    def test_gets_health_endpoint_and_returns_body(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok(_HEALTH)
        ) as mock_open:
            payload = cw.get_health()

        self.assertEqual(payload["routingGraph"], _HEALTH["routingGraph"])
        req = mock_open.call_args[0][0]
        self.assertEqual(req.full_url, _URL + "/health")
        self.assertEqual(req.get_method(), "GET")
        self.assertIsNone(req.data)

    def test_returns_none_without_url_configured(self):
        with _setenv(CONTEXTWEAVE_URL=""), patch.object(cw.urllib.request, "urlopen") as mock_open:
            self.assertIsNone(cw.get_health())
        mock_open.assert_not_called()

    def test_retries_server_error_then_degrades_to_none(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", side_effect=_http_error(503)
        ) as mock_open:
            self.assertIsNone(cw.get_health())
        self.assertEqual(mock_open.call_count, cw._MAX_RETRIES + 1)

    def test_does_not_retry_client_error(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", side_effect=_http_error(404)
        ) as mock_open:
            self.assertIsNone(cw.get_health())
        self.assertEqual(mock_open.call_count, 1)

    def test_transport_failure_degrades_to_none(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", side_effect=OSError("connection reset")
        ):
            self.assertIsNone(cw.get_health())

    def test_error_body_is_treated_as_failure(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok({"error": "boom"})
        ):
            self.assertIsNone(cw.get_health())

    def test_api_key_header_is_sent_on_get(self):
        with _setenv(CONTEXTWEAVE_URL=_URL, CONTEXTWEAVE_API_KEY="secret"), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok(_HEALTH)
        ) as mock_open:
            cw.get_health()

        # urllib capitalises header names on the Request object.
        self.assertEqual(mock_open.call_args[0][0].get_header("X-api-key"), "secret")


class GetRoutingDecisionsSummaryTests(unittest.TestCase):
    def test_requests_summary_mode_by_default(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok(_DECISIONS)
        ) as mock_open:
            payload = cw.get_routing_decisions_summary()

        self.assertEqual(payload["totalCount"], 512)
        self.assertEqual(mock_open.call_args[0][0].full_url, _URL + "/routing-decisions?mode=summary")

    def test_passes_question_type_and_since_as_query_params(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok(_DECISIONS)
        ) as mock_open:
            cw.get_routing_decisions_summary(question_type="skill_depth", since="2026-09-01")

        url = mock_open.call_args[0][0].full_url
        base, _, query = url.partition("?")
        self.assertEqual(base, _URL + "/routing-decisions")
        self.assertEqual(
            dict(urllib.parse.parse_qsl(query)),
            {"mode": "summary", "questionType": "skill_depth", "since": "2026-09-01"},
        )

    def test_blank_filters_are_omitted_from_the_query_string(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok(_DECISIONS)
        ) as mock_open:
            cw.get_routing_decisions_summary(question_type="", since=None)

        self.assertEqual(mock_open.call_args[0][0].full_url, _URL + "/routing-decisions?mode=summary")

    def test_null_averages_in_unrated_group_are_preserved(self):
        body = {
            "groups": [{"questionType": "project", "strategy": "keyword_boosted",
                        "count": 4, "ratedCount": 0, "avgConfidence": None,
                        "avgRating": None, "meanAbsDiff": None}],
            "totalCount": 4,
            "totalRated": 0,
        }
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok(body)
        ):
            payload = cw.get_routing_decisions_summary()

        self.assertIsNone(payload["groups"][0]["avgRating"])

    def test_returns_none_without_url_configured(self):
        with _setenv(CONTEXTWEAVE_URL=""), patch.object(cw.urllib.request, "urlopen") as mock_open:
            self.assertIsNone(cw.get_routing_decisions_summary())
        mock_open.assert_not_called()

    def test_failure_degrades_to_none(self):
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", side_effect=OSError("down")
        ):
            self.assertIsNone(cw.get_routing_decisions_summary())


class FormatRagContextTests(unittest.TestCase):
    def test_maps_answer_and_sources_into_explicit_mode_block_shape(self):
        ctx = cw.format_rag_context(_SAMPLE)

        self.assertEqual(
            ctx.splitlines(),
            [
                "[RAG #1] SOURCE: contextweave:answer (confidence 0.91)",
                "Tarun has built serverless RAG systems on AWS.",
                "---",
                "[RAG #2] SOURCE: architecture.md (weight 1.00)",
                "Step Functions orchestrate ingest.",
                "---",
                "[RAG #3] SOURCE: repo-signals.yaml (weight 0.70)",
                "pgvector, Neptune, Bedrock.",
                "---",
            ],
        )

    def test_falls_back_to_raw_chunk_source_shape(self):
        payload = {
            "answer": "",
            "sources": [{"sourceUri": "s3://bucket/raw/cw/CLAUDE.md", "content": "chunk text", "sourceWeight": 0.6}],
        }
        ctx = cw.format_rag_context(payload)

        self.assertEqual(
            ctx.splitlines(),
            [
                "[RAG #1] SOURCE: s3://bucket/raw/cw/CLAUDE.md (weight 0.60)",
                "chunk text",
                "---",
            ],
        )

    def test_skips_empty_and_malformed_sources(self):
        payload = {"answer": "A", "sources": [{"file": "x.md", "excerpt": "  "}, "not-a-dict"]}
        self.assertEqual(
            cw.format_rag_context(payload).splitlines(),
            ["[RAG #1] SOURCE: contextweave:answer", "A", "---"],
        )

    def test_empty_payload_maps_to_empty_context(self):
        self.assertEqual(cw.format_rag_context({}), "")


if __name__ == "__main__":
    unittest.main()
