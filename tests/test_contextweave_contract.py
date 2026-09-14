"""ContextWeave HTTP consumer conformance — fixtures come from the contract file.

ContextWeave produces the endpoints in ``contracts/contextweave_http_api.json``
(canonical home: the ContextWeave repository); TeamWeave consumes them in
``src/orchestrator/contextweave_client.py`` and composes two of them into
``GET /observability``.  Before that file existed, each side tested against its
own private idea of the other — this repository hard-coded sample response dicts
in its test bodies — so a field rename on either side left both suites green and
broke production.

Every payload in this module is therefore **read out of the contract JSON**, and
so are the key names the assertions look up: the endpoint's ``sample``,
``required_top_level_keys``, ``required_source_keys``, ``request_keys``,
``required_group_keys`` and ``nullable_group_keys``.  Nothing is retyped into the
test body, so renaming a field in the shared contract fails these tests rather
than passing them against a stale copy of TeamWeave's assumptions.

HTTP is stubbed at ``urllib.request.urlopen`` (the style
tests/test_contextweave_client.py uses) so the real client code — headers, query
string, retry policy, JSON decoding — runs against the contract samples.
"""
import copy
import importlib
import io
import json
import os
import sys
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3  # noqa: F401  -- imported before the handlers, as sibling tests do

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.orchestrator import contextweave_client as cw  # noqa: E402

CONTRACT_PATH = _REPO_ROOT / "contracts" / "contextweave_http_api.json"
with open(CONTRACT_PATH, encoding="utf-8") as _fh:
    CONTRACT = json.load(_fh)

ENDPOINTS = CONTRACT["endpoints"]

# Endpoint keys are the HTTP routes themselves, and the test below checks the
# client's own path constants against them.
QUERY_EP = "POST /query-expertise"
FEEDBACK_EP = "POST /feedback"
HEALTH_EP = "GET /health"
SUMMARY_EP = "GET /routing-decisions?mode=summary"

_URL = "https://contextweave.example.com"


def sample(endpoint: str) -> dict:
    """A fresh copy of the contract's sample response for an endpoint."""
    return copy.deepcopy(ENDPOINTS[endpoint]["sample"])


def _link_key() -> str:
    """The field that makes a later POST /feedback able to reach an answer.

    Derived, not typed: it is the one key the query response and the feedback
    request have in common.  If ContextWeave renames it, this resolves to a
    different name (or to nothing, failing loudly here) instead of silently
    letting a client that dropped the field keep passing.
    """
    shared = set(ENDPOINTS[QUERY_EP]["required_top_level_keys"]) & set(
        ENDPOINTS[FEEDBACK_EP]["request_keys"]
    )
    assert len(shared) == 1, f"expected exactly one linking field, got {sorted(shared)}"
    return shared.pop()


LINK_KEY = _link_key()


# ---------------------------------------------------------------------------
# HTTP stubs
# ---------------------------------------------------------------------------


class _FakeResponse(io.BytesIO):
    """Minimal stand-in for the object urlopen() yields as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _ok(payload) -> _FakeResponse:
    return _FakeResponse(json.dumps(payload).encode("utf-8"))


def _route_by_path(mapping):
    """A urlopen side effect that answers each request from ``mapping`` by path."""

    def _open(req, *_args, **_kwargs):
        path = urllib.parse.urlparse(req.full_url).path
        if path not in mapping:
            raise AssertionError(f"unexpected request to {req.full_url}")
        return _ok(mapping[path])

    return _open


def _setenv(**kwargs):
    return patch.dict(os.environ, kwargs)


# ---------------------------------------------------------------------------
# the contract file and the client agree on what endpoints exist
# ---------------------------------------------------------------------------


class ContractWiringTests(unittest.TestCase):
    def test_client_has_a_path_constant_for_every_contract_endpoint(self):
        """A contract endpoint this client cannot address is an unconsumed
        producer change; a client path absent from the contract is an
        unverified call."""
        contract_paths = {
            urllib.parse.urlparse(key.split(" ", 1)[1]).path for key in ENDPOINTS
        }
        client_paths = {
            cw.QUERY_PATH, cw.FEEDBACK_PATH, cw.HEALTH_PATH, cw.ROUTING_DECISIONS_PATH,
        }
        self.assertEqual(contract_paths, client_paths)

    def test_every_endpoint_carries_a_sample(self):
        for name, spec in ENDPOINTS.items():
            self.assertIn("sample", spec, f"{name} has no sample to test against")


# ---------------------------------------------------------------------------
# POST /query-expertise
# ---------------------------------------------------------------------------


class QueryExpertiseSampleTests(unittest.TestCase):
    """query_expertise() must parse the producer's real response shape."""

    def _query(self, payload=None, question="what has he built?", top_k=5):
        body = sample(QUERY_EP) if payload is None else payload
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok(body)
        ) as mock_open:
            result = cw.query_expertise(question, top_k=top_k)
        return result, mock_open

    def test_request_uses_the_contracts_request_keys(self):
        _, mock_open = self._query()
        req = mock_open.call_args[0][0]
        self.assertEqual(
            urllib.parse.urlparse(req.full_url).path,
            urllib.parse.urlparse(QUERY_EP.split(" ", 1)[1]).path,
        )
        self.assertEqual(
            set(json.loads(req.data.decode("utf-8"))),
            set(ENDPOINTS[QUERY_EP]["request_keys"]),
        )

    def test_every_required_key_of_the_sample_survives_parsing(self):
        expected = sample(QUERY_EP)
        result, _ = self._query()
        self.assertIsNotNone(result)
        for key in ENDPOINTS[QUERY_EP]["required_top_level_keys"]:
            self.assertIn(key, result, f"client dropped required key {key!r}")
            self.assertEqual(result[key], expected[key])

    def test_sources_keep_the_contracts_source_keys(self):
        expected = sample(QUERY_EP)
        result, _ = self._query()
        self.assertTrue(expected["sources"], "sample has no sources to check")
        for got, want in zip(result["sources"], expected["sources"]):
            for key in ENDPOINTS[QUERY_EP]["required_source_keys"]:
                self.assertIn(key, got, f"client dropped source key {key!r}")
                self.assertEqual(got[key], want[key])

    def test_preserves_the_field_a_later_rating_needs(self):
        """The query response and POST /feedback share exactly one field; a
        client that drops it can never rate the answer it received."""
        expected = sample(QUERY_EP)
        result, _ = self._query()
        self.assertEqual(result[LINK_KEY], expected[LINK_KEY])

    def test_that_field_round_trips_into_a_feedback_request(self):
        """End to end: the id parsed out of the query sample is the id posted
        back to POST /feedback, under the name the contract gives it."""
        answer, _ = self._query()

        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok(sample(FEEDBACK_EP))
        ) as mock_open:
            sent = cw.send_feedback(answer[LINK_KEY], "up")

        self.assertTrue(sent)
        req = mock_open.call_args[0][0]
        posted = json.loads(req.data.decode("utf-8"))
        self.assertEqual(set(posted), set(ENDPOINTS[FEEDBACK_EP]["request_keys"]))
        self.assertEqual(posted[LINK_KEY], sample(QUERY_EP)[LINK_KEY])

    def test_rag_mode_carries_the_sample_id_into_the_step_record(self):
        """contextweave RAG mode persists the id as rag_meta.query_id, which is
        what worker_handler later feeds to maybe_send_valid_output_feedback."""
        # rag.py builds Bedrock clients at import time (as tests/test_bedrock_invoke.py
        # also has to account for); no AWS call is made by this test.
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
        rag = importlib.import_module("src.orchestrator.rag")
        expected = sample(QUERY_EP)

        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok(expected)
        ):
            context, meta = rag.get_rag_context_with_meta(
                {"topic": "serverless RAG", "objective": "", "audience": ""},
                {"features": {"explicit_rag": True},
                 "rag": {"mode": "contextweave", "top_k": 4}},
                "owner@example.com",
            )

        self.assertEqual(meta["query_id"], expected[LINK_KEY])
        self.assertEqual(meta["confidence"], expected["confidence"])
        self.assertEqual(meta["question_type"], expected["questionType"])
        self.assertIn(expected["answer"], context)


class FormatRagContextSampleTests(unittest.TestCase):
    """format_rag_context() must render the producer's real source shape.

    The sample's sources use the keys the contract lists in
    ``required_source_keys`` (``file`` / ``excerpt`` / ``weight``).  Rather than
    naming them again here, the assertions require that every string the
    contract put in a source reaches the rendered block — which is exactly what
    stops being true if one of those keys is renamed, since the client's
    ``_source_text`` would then return "" and drop the block entirely.
    """

    def setUp(self):
        self.payload = sample(QUERY_EP)
        self.rendered = cw.format_rag_context(self.payload)

    @staticmethod
    def _strings(src):
        return [v for v in src.values() if isinstance(v, str) and v.strip()]

    @staticmethod
    def _numbers(src):
        return [v for v in src.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)]

    def test_renders_the_answer(self):
        self.assertIn(self.payload["answer"], self.rendered)

    def test_renders_every_string_the_contract_put_in_a_source(self):
        self.assertTrue(self.payload["sources"], "sample has no sources to render")
        for src in self.payload["sources"]:
            for value in self._strings(src):
                self.assertIn(value, self.rendered,
                              f"source value {value!r} never reached RAG_CONTEXT")

    def test_renders_each_sources_numeric_weight(self):
        for src in self.payload["sources"]:
            for value in self._numbers(src):
                self.assertIn(f"{float(value):.2f}", self.rendered)

    def test_emits_one_block_per_source_plus_the_answer(self):
        blocks = [ln for ln in self.rendered.splitlines() if ln.startswith("[RAG #")]
        self.assertEqual(len(blocks), 1 + len(self.payload["sources"]))


# ---------------------------------------------------------------------------
# GET /health + GET /routing-decisions?mode=summary, through the client, into
# the composed GET /observability response
# ---------------------------------------------------------------------------


def _run_unified(health=None, summary=None, params=None):
    """Call the unified handler with the contract samples served over the wire.

    The ContextWeave half goes through the real client (urlopen stubbed), so the
    handler receives whatever ``contextweave_client`` actually makes of the
    contract samples.  DynamoDB is mocked as tests/test_agent_metrics_handler.py
    mocks it.
    """
    unified = importlib.import_module("src.orchestrator.unified_observability_handler")
    metrics = importlib.import_module("src.orchestrator.agent_metrics_handler")
    metrics._ddb_table = None

    responses = {
        cw.HEALTH_PATH: sample(HEALTH_EP) if health is None else health,
        cw.ROUTING_DECISIONS_PATH: sample(SUMMARY_EP) if summary is None else summary,
    }

    table = MagicMock()
    table.query.side_effect = lambda **_kw: {"Items": [], "ScannedCount": 0}

    env = {"CONTEXTWEAVE_URL": _URL, "OBSERVATORY_METRICS_TABLE": "contract-test-table"}
    with patch.dict(os.environ, env, clear=False):
        with patch("boto3.resource") as mock_resource, patch.object(
            cw.urllib.request, "urlopen", side_effect=_route_by_path(responses)
        ) as mock_open:
            mock_resource.return_value.Table.return_value = table
            resp = unified.handler({"queryStringParameters": params or {}}, None)

    metrics._ddb_table = None
    return json.loads(resp["body"]), resp, mock_open


class UnifiedObservabilityFromContractSamplesTests(unittest.TestCase):
    def test_routing_graph_section_is_the_health_samples_routing_graph(self):
        body, resp, _ = _run_unified()
        self.assertEqual(resp["statusCode"], 200)
        self.assertEqual(body["routingGraph"], sample(HEALTH_EP)["routingGraph"])

    def test_health_sample_keeps_every_key_the_contract_requires(self):
        """The handler consumes one key of /health, but the client must be able
        to read the whole documented body to get there."""
        with _setenv(CONTEXTWEAVE_URL=_URL), patch.object(
            cw.urllib.request, "urlopen", return_value=_ok(sample(HEALTH_EP))
        ):
            payload = cw.get_health()
        for key in ENDPOINTS[HEALTH_EP]["required_top_level_keys"]:
            self.assertIn(key, payload)
            self.assertEqual(payload[key], sample(HEALTH_EP)[key])

    def test_routing_decisions_section_is_the_summary_sample(self):
        body, _, _ = _run_unified()
        self.assertEqual(body["routingDecisions"], sample(SUMMARY_EP))

    def test_summary_sample_keeps_every_required_top_level_key(self):
        body, _, _ = _run_unified()
        for key in ENDPOINTS[SUMMARY_EP]["required_top_level_keys"]:
            self.assertIn(key, body["routingDecisions"])
            self.assertEqual(body["routingDecisions"][key], sample(SUMMARY_EP)[key])

    def test_every_group_keeps_every_required_group_key(self):
        body, _, _ = _run_unified()
        expected_groups = sample(SUMMARY_EP)["groups"]
        got_groups = body["routingDecisions"]["groups"]
        self.assertEqual(len(got_groups), len(expected_groups))
        for got, want in zip(got_groups, expected_groups):
            for key in ENDPOINTS[SUMMARY_EP]["required_group_keys"]:
                self.assertIn(key, got, f"group key {key!r} lost in transit")
                self.assertEqual(got[key], want[key])

    def test_unrated_groups_keep_null_averages_rather_than_zero(self):
        """A group nobody has rated reports null, never 0.0: a zero would read
        as perfect agreement on exactly the groups where nothing is known.  The
        nullable keys come from the contract's nullable_group_keys."""
        expected_groups = sample(SUMMARY_EP)["groups"]
        nullable = ENDPOINTS[SUMMARY_EP]["nullable_group_keys"]

        nulls_in_sample = [
            (i, key)
            for i, grp in enumerate(expected_groups)
            for key in nullable
            if grp.get(key) is None
        ]
        self.assertTrue(
            nulls_in_sample,
            "the contract sample no longer contains an unrated group, so this "
            "round trip cannot check null semantics",
        )

        body, _, _ = _run_unified()
        got_groups = body["routingDecisions"]["groups"]
        for index, key in nulls_in_sample:
            value = got_groups[index][key]
            self.assertIsNone(value, f"group {index} key {key!r} became {value!r}")
            self.assertNotEqual(value, 0)
            self.assertNotEqual(value, 0.0)

    def test_rated_groups_keep_their_numeric_averages(self):
        body, _, _ = _run_unified()
        expected_groups = sample(SUMMARY_EP)["groups"]
        nullable = ENDPOINTS[SUMMARY_EP]["nullable_group_keys"]
        checked = 0
        for got, want in zip(body["routingDecisions"]["groups"], expected_groups):
            for key in nullable:
                if want.get(key) is not None:
                    self.assertAlmostEqual(got[key], want[key])
                    checked += 1
        self.assertGreater(checked, 0)

    def test_summary_mode_and_filters_reach_the_wire(self):
        _, _, mock_open = _run_unified(params={"questionType": "skill_depth",
                                               "since": "2026-09-01"})
        urls = [call[0][0].full_url for call in mock_open.call_args_list]
        decision_urls = [u for u in urls if cw.ROUTING_DECISIONS_PATH in u]
        self.assertEqual(len(decision_urls), 1)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(decision_urls[0]).query)
        self.assertEqual(query["mode"], ["summary"])
        self.assertEqual(query["questionType"], ["skill_depth"])
        self.assertEqual(query["since"], ["2026-09-01"])
        # ...and the mode= the contract names in its endpoint key is the one sent.
        contract_mode = urllib.parse.parse_qs(
            urllib.parse.urlparse(SUMMARY_EP.split(" ", 1)[1]).query
        )["mode"]
        self.assertEqual(query["mode"], contract_mode)


if __name__ == "__main__":
    unittest.main()
