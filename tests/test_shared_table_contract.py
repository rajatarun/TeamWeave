"""Conformance of TeamWeave's writer and readers against the shared table contract.

The ``OBSERVATORY_METRICS`` DynamoDB table is a cross-repository interface: it is
written by several weave services in two languages and read by dashboards that
cannot see any writer's code.  The item shape is therefore pinned in
mcp-observatory's ``contracts/observatory_metrics_item.json``, vendored here as
``contracts/`` (see ``contracts/README.md``), and this module checks *this*
repository's three couplings to it:

* the writer  -- ``src/orchestrator/mcp_observatory.py`` (``_push_metric``)
* the readers -- ``src/orchestrator/agent_metrics_handler.py`` and
  ``src/orchestrator/unified_observability_handler.py``

Nothing here re-types the contract's values into the test body.  The expected
namespaces, operations and reader names are all read out of the JSON file, so
renaming a field or dropping an operation in the canonical contract makes these
tests fail here rather than silently in production.

The writer half mocks boto3 exactly the way ``tests/test_mcp_observatory.py``
does (``patch.object(..., "_get_ddb_table")`` and read the item back off
``put_item``); the reader half mocks it the way
``tests/test_agent_metrics_handler.py`` does (``patch("boto3.resource")`` with a
MagicMock table) and inspects the ``KeyConditionExpression`` the handler actually
built, so the pk values under test are the ones the code really queries.
"""
import importlib
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import boto3  # noqa: F401  -- imported before the handlers, as sibling tests do
from boto3.dynamodb.conditions import And, Equals, Key

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contracts.conformance import check_item, load_contract, readers_for  # noqa: E402

CONTRACT = load_contract()

_OBSERVATORY = CONTRACT["namespace_registry"]["OBSERVATORY"]

# How the contract's registry spells a reader: "<repo>:<module>".  The module
# half is taken from the real module object rather than typed here, so the
# label under test cannot drift from the code that does the querying — a rename
# on either side (the contract's entry, or the module) fails the checks below
# instead of quietly never matching.
_REPO_LABEL = "teamweave"
_READER_MODULES = (
    "src.orchestrator.agent_metrics_handler",
    "src.orchestrator.unified_observability_handler",
)


def _reader_label(module_path: str) -> str:
    module = importlib.import_module(module_path)
    return f"{_REPO_LABEL}:{module.__name__.rsplit('.', 1)[-1]}"


_TEAMWEAVE_READERS = [_reader_label(m) for m in _READER_MODULES]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_wrapper_result(output, *, action="allow", reason="within_budget"):
    """A minimal WrapperResult-like object, as tests/test_mcp_observatory.py builds."""
    span = MagicMock()
    span.trace_id = "trace-contract"
    span.prompt_tokens = 11
    span.completion_tokens = 7
    span.cost_usd = 0.0012
    span.shadow_disagreement_score = None
    span.shadow_numeric_variance = None

    decision = MagicMock()
    decision.action = action
    decision.reason = reason

    result = MagicMock()
    result.output = output
    result.span = span
    result.decision = decision
    return result


def _pk_values_from_condition(cond) -> list:
    """Every value compared for equality against the 'pk' attribute in a KeyCondition.

    ``query_by_pk`` builds ``Key("pk").eq(pk) [& Key("sk")...]``; this walks the
    resulting condition tree so the assertions below run against the pk the
    handler really sent to DynamoDB, not a string re-derived in the test.
    """
    found = []
    if isinstance(cond, Equals):
        left, right = cond._values
        if isinstance(left, Key) and left.name == "pk":
            found.append(right)
    elif isinstance(cond, And):
        for sub in cond._values:
            found.extend(_pk_values_from_condition(sub))
    return found


def _empty_page(**_kwargs):
    return {"Items": [], "ScannedCount": 0}


def _capture_queried_pks(run) -> list:
    """Run a reader against a mocked table and return the pks it queried, in order."""
    metrics = importlib.import_module("src.orchestrator.agent_metrics_handler")
    metrics._ddb_table = None

    table = MagicMock()
    table.query.side_effect = _empty_page

    env = {"OBSERVATORY_METRICS_TABLE": "contract-test-table", "CONTEXTWEAVE_URL": ""}
    with patch.dict(os.environ, env, clear=False):
        with patch("boto3.resource") as mock_resource:
            mock_resource.return_value.Table.return_value = table
            run()

    metrics._ddb_table = None

    pks = []
    for call in table.query.call_args_list:
        pks.extend(_pk_values_from_condition(call.kwargs["KeyConditionExpression"]))
    return pks


# ---------------------------------------------------------------------------
# TASK 2 -- writer conformance
# ---------------------------------------------------------------------------


class WriterConformanceTests(unittest.TestCase):
    """The item mcp_observatory really writes must satisfy the shared contract."""

    @classmethod
    def setUpClass(cls):
        cls.observatory = importlib.import_module("src.orchestrator.mcp_observatory")

    def setUp(self):
        self.observatory._ddb_table = None

    def _write_agent_span(self):
        """Exercise the real writer and return the item handed to put_item."""
        table = MagicMock()
        fake_result = _make_wrapper_result({"completion": []})
        with patch.object(
            self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)
        ):
            with patch.object(self.observatory, "_get_ddb_table", return_value=table):
                self.observatory.observe_agent_request(
                    MagicMock(), agent_id="a1", alias_id="al1",
                    session_id="s1", input_text="hello",
                )
        return table.put_item.call_args.kwargs["Item"]

    def _write_model_span(self):
        table = MagicMock()
        fake_result = _make_wrapper_result({"body": "{}"})
        with patch.object(
            self.observatory._wrapper, "invoke", new=AsyncMock(return_value=fake_result)
        ):
            with patch.object(self.observatory, "_get_ddb_table", return_value=table):
                self.observatory.observe_model_request(
                    MagicMock(), model_id="amazon.nova-lite-v1:0", body="{}",
                )
        return table.put_item.call_args.kwargs["Item"]

    def test_agent_span_item_conforms_to_the_shared_contract(self):
        problems = check_item(self._write_agent_span(), CONTRACT)
        self.assertEqual(problems, [], problems)

    def test_model_span_item_conforms_to_the_shared_contract(self):
        problems = check_item(self._write_model_span(), CONTRACT)
        self.assertEqual(problems, [], problems)

    def test_key_attributes_are_spelled_in_lower_case(self):
        """DynamoDB attribute names are case sensitive and the table declares
        'pk'/'sk'.  A writer that emits 'PK'/'SK' gets a ValidationException from
        PutItem -- and _push_metric swallows write exceptions, so it would report
        success while writing nothing at all."""
        item = self._write_agent_span()
        self.assertIn("pk", item)
        self.assertIn("sk", item)
        self.assertNotIn("PK", item)
        self.assertNotIn("SK", item)
        self.assertIn(CONTRACT["key_schema"]["partition_key"], item)
        self.assertIn(CONTRACT["key_schema"]["sort_key"], item)
        # Nothing else in the item may differ from these two only by case.
        lowered = [k.lower() for k in item]
        self.assertEqual(lowered.count("pk"), 1)
        self.assertEqual(lowered.count("sk"), 1)

    def test_writer_namespace_is_registered_and_actually_read(self):
        """Invariant I5: a writer's namespace must have at least one reader."""
        for item in (self._write_agent_span(), self._write_model_span()):
            namespace = item["pk"].split("#", 1)[0]
            self.assertIn(namespace, CONTRACT["namespace_registry"])
            self.assertNotEqual(
                readers_for(item["pk"]), [],
                f"{item['pk']} is written by this repo but read by nobody",
            )

    def test_written_operations_are_registered_discriminator_values(self):
        allowed = _OBSERVATORY["discriminator_values"]
        for item in (self._write_agent_span(), self._write_model_span()):
            self.assertIn(item["pk"].split("#", 1)[1], allowed)

    def test_sort_key_leads_with_the_timestamp_then_the_trace_id(self):
        """Readers range-query sk lexicographically (see query_by_pk), so the
        timestamp has to sort first."""
        item = self._write_agent_span()
        timestamp, _, trace = item["sk"].partition("#")
        self.assertEqual(trace, "trace-contract")
        self.assertEqual(timestamp, item["timestamp"])
        self.assertTrue(timestamp.startswith("20"))

    def test_ttl_is_present_and_in_the_future(self):
        """Invariant I4 -- rows expire rather than accumulating in a shared table."""
        item = self._write_agent_span()
        self.assertIn("ttl", item)
        self.assertGreater(int(item["ttl"]), int(time.time()))


# ---------------------------------------------------------------------------
# TASK 3 -- reader conformance
# ---------------------------------------------------------------------------


class ReaderQueryConformanceTests(unittest.TestCase):
    """Every pk the dashboards query must be one the contract says exists."""

    def _agent_metrics_pks(self):
        metrics = importlib.import_module("src.orchestrator.agent_metrics_handler")
        return _capture_queried_pks(
            lambda: metrics.handler({"queryStringParameters": {"operation": "all"}}, None)
        )

    def _unified_pks(self):
        unified = importlib.import_module("src.orchestrator.unified_observability_handler")
        return _capture_queried_pks(
            lambda: unified.handler({"queryStringParameters": {}}, None)
        )

    def test_agent_metrics_handler_queries_only_contract_conformant_pks(self):
        pks = self._agent_metrics_pks()
        self.assertTrue(pks, "the reader queried nothing; the capture is broken")
        for pk in pks:
            namespace, _, discriminator = pk.partition("#")
            self.assertTrue(discriminator, f"pk {pk!r} has no discriminator")
            self.assertIn(
                namespace, CONTRACT["namespace_registry"],
                f"pk {pk!r} uses a namespace absent from the contract's registry",
            )

    def test_unified_observability_handler_queries_only_contract_conformant_pks(self):
        pks = self._unified_pks()
        self.assertTrue(pks, "the reader queried nothing; the capture is broken")
        for pk in pks:
            namespace, _, discriminator = pk.partition("#")
            self.assertTrue(discriminator)
            self.assertIn(namespace, CONTRACT["namespace_registry"])

    def test_both_readers_query_the_same_partitions(self):
        """unified_observability_handler reuses agent_metrics_handler's query
        logic; if that ever forks, the two dashboards start disagreeing."""
        self.assertEqual(sorted(set(self._agent_metrics_pks())),
                         sorted(set(self._unified_pks())))

    def test_reader_operation_list_matches_the_contracts_discriminator_values(self):
        """The contract's registry and the reader's real query list must agree.

        A mismatch in either direction is a genuine finding, not a test bug:
        an operation the code queries but the contract omits means some writer's
        rows are undeclared; an operation the contract declares but the code
        never queries means that writer's telemetry reaches no dashboard.
        """
        metrics = importlib.import_module("src.orchestrator.agent_metrics_handler")
        in_code = set(metrics._ALL_OPERATION_PKS)
        in_contract = set(_OBSERVATORY["discriminator_values"])

        self.assertEqual(
            in_code, in_contract,
            "reader queries operations the contract omits: "
            f"{sorted(in_code - in_contract)}; contract declares operations no "
            f"reader queries: {sorted(in_contract - in_code)}",
        )

    def test_contract_lists_these_readers_for_every_operation_they_query(self):
        """For each operation in _ALL_OPERATION_PKS, readers_for() must name
        this repository's two readers -- the registry and the real query list
        agreeing in the direction the registry is actually used for."""
        metrics = importlib.import_module("src.orchestrator.agent_metrics_handler")
        self.assertEqual(
            sorted(_TEAMWEAVE_READERS),
            sorted(r for r in _OBSERVATORY["readers"] if r.startswith(_REPO_LABEL + ":")),
            "the contract's teamweave readers and this repo's reader modules differ",
        )

        for operation in metrics._ALL_OPERATION_PKS:
            pk = f"OBSERVATORY#{operation}"
            readers = readers_for(pk, CONTRACT)
            for reader in _TEAMWEAVE_READERS:
                self.assertIn(
                    reader, readers,
                    f"{reader} queries {pk} but the contract does not list it as a reader",
                )

    def test_every_queried_pk_reports_these_readers(self):
        """The same check, driven by the pks the handler really emitted."""
        for pk in set(self._agent_metrics_pks()) | set(self._unified_pks()):
            readers = readers_for(pk, CONTRACT)
            for reader in _TEAMWEAVE_READERS:
                self.assertIn(reader, readers, f"{pk} is queried but {reader} is not registered")


class UnreadNamespaceGapTests(unittest.TestCase):
    """Namespaces written to the shared table that these dashboards never show.

    These assertions pin a known platform gap rather than a desired behaviour.
    The shared table has writers whose rows land in partitions no reader here
    enumerates, so the telemetry is durable, billable, and invisible:

    * ``SPAN#*`` is what the shared library exporter
      (``mcp_observatory.aws.DynamoDBSpanExporter``) writes.  Migrating any
      service off its vendored copy onto that exporter -- the stated point of
      platform edge E2 -- silently removes it from these dashboards.
    * ``OBSERVATORY#screenshot`` is the shape ScreenWeave emits: a registered
      namespace, but a discriminator outside the reader's ``_ALL_OPERATION_PKS``,
      so ``readers_for`` reports no reader for it either.

    The contract records both as ``status: "unread"`` / an unlisted
    discriminator.  Resolving them is a platform decision in mcp-observatory's
    docs/integration-audit.md, not something this repository can fix alone;
    these tests exist so the gap cannot quietly stop being true.
    """

    def test_span_namespace_is_written_by_the_library_exporter_and_read_by_nobody(self):
        self.assertEqual(readers_for("SPAN#anything", CONTRACT), [])
        entry = CONTRACT["namespace_registry"]["SPAN"]
        self.assertEqual(entry["status"], "unread")
        self.assertEqual(entry["readers"], [])

    def test_screenweave_operation_is_in_a_read_namespace_but_no_reader_sees_it(self):
        self.assertEqual(readers_for("OBSERVATORY#screenshot", CONTRACT), [])
        # The namespace itself is read -- it is the discriminator that is unlisted.
        self.assertNotEqual(readers_for("OBSERVATORY#invoke_agent", CONTRACT), [])
        self.assertNotIn("screenshot", _OBSERVATORY["discriminator_values"])

    def test_the_other_unread_namespaces_are_declared_as_such(self):
        for namespace in ("WRAPPER", "INVOCATION"):
            entry = CONTRACT["namespace_registry"][namespace]
            self.assertEqual(entry["status"], "unread", namespace)
            self.assertEqual(readers_for(f"{namespace}#anything", CONTRACT), [])


if __name__ == "__main__":
    unittest.main()
