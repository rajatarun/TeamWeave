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

Contract v2.0.0 moved reads off the partition key.  Under v1 a reader queried
whole partitions by exact pk, so a writer's rows were visible only if it had
guessed the same prefix grammar the reader enumerated -- three of five writers
had not, and their telemetry was durable, billable and permanently invisible.
Reads now go through the ``SpanTimelineIndex`` GSI (``span_date`` + ``timestamp``),
which changes what these tests are for:

* the writer half no longer asks "is your namespace one somebody reads?" (I5,
  superseded) but "do you carry the two index key attributes and an operation?"
  (I6-I8) -- because a GSI indexes only items holding both of its keys, so
  omitting ``span_date`` is exactly as invisible as a mismatched prefix was,
  and this is the check that turns that silence into a failing test;
* the reader half asserts the *shape of the query*: the contract's index, its
  partition key, its sort key, and that pk is not a partition anywhere -- so
  the v1 defect cannot be reintroduced by a later refactor;
* and both halves are checked against a fallback path, because the index lives
  in a different stack and a reader that cannot tolerate its absence makes
  deploy order load-bearing.

Nothing here re-types the contract's values into the test body.  The index name,
its key attributes, the namespaces and the reader names are all read out of the
JSON file, so renaming any of them in the canonical contract fails these tests
here rather than silently in production.

The writer half mocks boto3 exactly the way ``tests/test_mcp_observatory.py``
does (``patch.object(..., "_get_ddb_table")`` and read the item back off
``put_item``); the reader half mocks it the way
``tests/test_agent_metrics_handler.py`` does (``patch("boto3.resource")`` with a
MagicMock table) and inspects the query keyword arguments the handler actually
built, so the index and keys under test are the ones the code really queries.
"""
import importlib
import json
import os
import re
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import boto3  # noqa: F401  -- imported before the handlers, as sibling tests do
from boto3.dynamodb.conditions import And, Attr, Key
from botocore.exceptions import ClientError

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contracts.conformance import check_item, load_contract, readers_for  # noqa: E402

CONTRACT = load_contract()

_OBSERVATORY = CONTRACT["namespace_registry"]["OBSERVATORY"]

# The read path, taken from the contract rather than typed here.
_GSI = CONTRACT["gsi"]
_INDEX_NAME = _GSI["name"]
_INDEX_PARTITION_KEY = _GSI["partition_key"]
_INDEX_SORT_KEY = _GSI["sort_key"]

_SPAN_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

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


def _metrics():
    return importlib.import_module("src.orchestrator.agent_metrics_handler")


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


def _comparisons(condition) -> list[tuple[str, str, str, list]]:
    """Flatten a boto3 condition into (kind, attribute, operator, values) tuples.

    ``kind`` is "Key" or "Attr", which is the distinction that matters here:
    the whole point of v2 is that ``operation`` moved from a Key (a partition
    you must know the name of) to an Attr (a filter).
    """
    found: list[tuple[str, str, str, list]] = []
    values = list(getattr(condition, "_values", ()) or ())
    if isinstance(condition, And):
        for sub in values:
            found.extend(_comparisons(sub))
        return found
    if values and isinstance(values[0], (Key, Attr)):
        found.append((
            type(values[0]).__name__,
            values[0].name,
            getattr(condition, "expression_operator", ""),
            values[1:],
        ))
    return found


def _empty_page(**_kwargs):
    return {"Items": [], "ScannedCount": 0}


def _capture_queries(run, table=None) -> list[dict]:
    """Run a reader against a mocked table and return the query kwargs it sent."""
    metrics = _metrics()
    metrics._ddb_table = None

    if table is None:
        table = MagicMock()
        table.query.side_effect = _empty_page

    env = {"OBSERVATORY_METRICS_TABLE": "contract-test-table", "CONTEXTWEAVE_URL": ""}
    with patch.dict(os.environ, env, clear=False):
        with patch("boto3.resource") as mock_resource:
            mock_resource.return_value.Table.return_value = table
            run()

    metrics._ddb_table = None
    return [call.kwargs for call in table.query.call_args_list]


def _run_agent_metrics(params=None):
    metrics = _metrics()
    return lambda: metrics.handler(
        {"queryStringParameters": params or {"operation": "all"}}, None
    )


def _run_unified():
    unified = importlib.import_module("src.orchestrator.unified_observability_handler")
    return lambda: unified.handler({"queryStringParameters": {}}, None)


def _client_error(code: str, message: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "Query")


def _index_failing_table(exc: BaseException):
    """A table whose SpanTimelineIndex queries raise; base-table queries work."""
    table = MagicMock()

    def _query(**kwargs):
        if kwargs.get("IndexName") == _INDEX_NAME:
            raise exc
        return {"Items": [], "ScannedCount": 0}

    table.query.side_effect = _query
    return table


def _pk_partitions(queries: list[dict]) -> list:
    """Values compared for equality against the base-table pk, across queries."""
    found = []
    for kwargs in queries:
        for kind, name, operator, values in _comparisons(kwargs["KeyConditionExpression"]):
            if kind == "Key" and name == "pk" and operator == "=":
                found.extend(values)
    return found


def _index_partitions(queries: list[dict]) -> list:
    found = []
    for kwargs in queries:
        for kind, name, operator, values in _comparisons(kwargs["KeyConditionExpression"]):
            if kind == "Key" and name == _INDEX_PARTITION_KEY and operator == "=":
                found.extend(values)
    return found


# ---------------------------------------------------------------------------
# writer conformance
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
        """check_item now enforces I6-I8 as well; this is the whole v2 check."""
        problems = check_item(self._write_agent_span(), CONTRACT)
        self.assertEqual(problems, [], problems)

    def test_model_span_item_conforms_to_the_shared_contract(self):
        problems = check_item(self._write_model_span(), CONTRACT)
        self.assertEqual(problems, [], problems)

    def test_every_required_attribute_of_the_contract_is_written(self):
        """Driven by the contract's own required_attributes list.

        Adding a required attribute upstream fails here on the next re-vendor,
        rather than producing rows that are quietly missing from the index.
        """
        for item in (self._write_agent_span(), self._write_model_span()):
            for attribute in CONTRACT["required_attributes"]:
                self.assertIn(attribute, item, f"writer omits required {attribute!r}")
                self.assertNotEqual(item[attribute], "", attribute)

    def test_writer_carries_both_index_key_attributes(self):
        """I6/I7: a GSI indexes only items holding BOTH of its key attributes.

        A span missing either one is not in the index at all -- invisible to
        every dashboard, with no error raised anywhere, which is exactly the
        v1 failure this migration exists to end.
        """
        for item in (self._write_agent_span(), self._write_model_span()):
            self.assertIn(_INDEX_PARTITION_KEY, item)
            self.assertIn(_INDEX_SORT_KEY, item)
            self.assertTrue(_SPAN_DATE_RE.match(str(item[_INDEX_PARTITION_KEY])),
                            item[_INDEX_PARTITION_KEY])

    def test_span_date_agrees_with_the_timestamp_it_indexes(self):
        """I7: disagreement would file the row under a day it did not happen on.

        Both must also be UTC: the index sort key is compared lexicographically,
        so a local-time timestamp sorts into the wrong place in its own day.
        """
        for item in (self._write_agent_span(), self._write_model_span()):
            span_date = str(item[_INDEX_PARTITION_KEY])
            timestamp = str(item[_INDEX_SORT_KEY])
            self.assertEqual(timestamp[:10], span_date)
            # The writer derives both from one UTC clock reading; assert the
            # value it produced really is today in UTC and not local time.
            self.assertEqual(
                span_date,
                time.strftime("%Y-%m-%d", time.gmtime()),
            )

    def test_operation_is_written_as_a_top_level_attribute(self):
        """I8: readers filter and group on `operation`, not on the pk prefix."""
        for item, expected in ((self._write_agent_span(), "invoke_agent"),
                               (self._write_model_span(), "invoke_model")):
            self.assertEqual(item["operation"], expected)
            # ...and it agrees with the pk the writer happens to use, so the
            # legacy fallback path and the index path group rows identically.
            self.assertEqual(item["pk"].split("#", 1)[1], item["operation"])

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

    def test_writer_namespace_is_still_a_registered_one(self):
        """I2 survives v2; I5 does not.

        The pk is now the writer's own business -- reachability is a property
        of the index keys, not of the namespace -- but the grammar is still
        checked so rows written before v2 stay interpretable.
        """
        for item in (self._write_agent_span(), self._write_model_span()):
            namespace = item["pk"].split("#", 1)[0]
            self.assertIn(namespace, CONTRACT["namespace_registry"])

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

    def test_dropping_span_date_is_caught_by_the_contract_checker(self):
        """The guard above is only worth having if it can fail.

        A row without span_date is absent from the index; this asserts that
        such a row is reported as a violation rather than passing quietly.
        """
        item = dict(self._write_agent_span())
        item.pop(_INDEX_PARTITION_KEY)
        problems = check_item(item, CONTRACT)
        self.assertTrue(any("I6" in p for p in problems), problems)

    def test_span_date_disagreeing_with_timestamp_is_caught(self):
        item = dict(self._write_agent_span())
        item[_INDEX_PARTITION_KEY] = "1999-01-01"
        problems = check_item(item, CONTRACT)
        self.assertTrue(any("I7" in p for p in problems), problems)


# ---------------------------------------------------------------------------
# reader conformance -- the query shape, not the partition vocabulary
# ---------------------------------------------------------------------------


class ReaderIndexConformanceTests(unittest.TestCase):
    """Both dashboards must read through the index the contract declares."""

    def setUp(self):
        _metrics()._index_fallback_warned = False

    def _agent_metrics_queries(self):
        return _capture_queries(_run_agent_metrics())

    def _unified_queries(self):
        return _capture_queries(_run_unified())

    def test_reader_constants_match_the_contracts_gsi_block(self):
        """The names the code queries with are the contract's, not a copy."""
        metrics = _metrics()
        self.assertEqual(metrics.SPAN_TIMELINE_INDEX, _INDEX_NAME)
        self.assertEqual(metrics.SPAN_TIMELINE_PARTITION_KEY, _INDEX_PARTITION_KEY)
        self.assertEqual(metrics.SPAN_TIMELINE_SORT_KEY, _INDEX_SORT_KEY)

    def test_agent_metrics_handler_queries_the_contracts_index(self):
        queries = self._agent_metrics_queries()
        self.assertTrue(queries, "the reader queried nothing; the capture is broken")
        for kwargs in queries:
            self.assertEqual(kwargs.get("IndexName"), _INDEX_NAME)

    def test_unified_observability_handler_queries_the_contracts_index(self):
        queries = self._unified_queries()
        self.assertTrue(queries, "the reader queried nothing; the capture is broken")
        for kwargs in queries:
            self.assertEqual(kwargs.get("IndexName"), _INDEX_NAME)

    def test_readers_partition_on_the_contracts_gsi_partition_key(self):
        for queries in (self._agent_metrics_queries(), self._unified_queries()):
            partitions = _index_partitions(queries)
            self.assertEqual(len(partitions), len(queries))
            for value in partitions:
                self.assertTrue(_SPAN_DATE_RE.match(str(value)), value)

    def test_readers_range_on_the_contracts_gsi_sort_key(self):
        for queries in (self._agent_metrics_queries(), self._unified_queries()):
            for kwargs in queries:
                sort_conditions = [
                    c for c in _comparisons(kwargs["KeyConditionExpression"])
                    if c[0] == "Key" and c[1] == _INDEX_SORT_KEY
                ]
                self.assertTrue(sort_conditions, kwargs["KeyConditionExpression"])

    def test_no_reader_partitions_on_the_base_table_pk(self):
        """The v1 defect, pinned shut.

        Querying by pk is what made a writer's rows depend on guessing this
        repository's prefix vocabulary.  If a refactor reintroduces it on the
        primary path, this fails.
        """
        for queries in (self._agent_metrics_queries(), self._unified_queries()):
            self.assertEqual(_pk_partitions(queries), [])

    def test_operation_is_a_filter_not_a_partition(self):
        queries = _capture_queries(_run_agent_metrics({"operation": "invoke_agent"}))
        self.assertTrue(queries)
        for kwargs in queries:
            filters = _comparisons(kwargs["FilterExpression"])
            self.assertIn(
                ("Attr", "operation", "=", ["invoke_agent"]),
                [(k, n, o, list(v)) for k, n, o, v in filters],
            )
            for kind, name, _operator, _values in _comparisons(kwargs["KeyConditionExpression"]):
                self.assertNotEqual(name, "operation", "operation is still a key")

    def test_operation_all_means_no_filter_rather_than_n_queries(self):
        """Under v1 "all" was an enumeration of known pks, so an operation no
        reader had listed was unreachable.  On the index it is simply unfiltered."""
        queries = _capture_queries(_run_agent_metrics({"operation": "all"}))
        self.assertTrue(queries)
        for kwargs in queries:
            self.assertNotIn("FilterExpression", kwargs)

    def test_both_readers_query_the_same_index_and_partitions(self):
        """unified_observability_handler reuses agent_metrics_handler's query
        logic; if that ever forks, the two dashboards start disagreeing."""
        self.assertEqual(
            sorted(set(_index_partitions(self._agent_metrics_queries()))),
            sorted(set(_index_partitions(self._unified_queries()))),
        )

    def test_a_writer_using_an_unenumerated_prefix_is_not_excluded_by_the_query(self):
        """The property v2 buys, stated as a test.

        No condition in either reader's query mentions the pk, so a row written
        under any prefix at all is returned as long as it carries the index
        keys.  ``readers_for`` -- which answers the v1 question -- would still
        report nobody for such a pk; that is now archaeology, not reachability.
        """
        queries = self._agent_metrics_queries()
        for kwargs in queries:
            names = {name for _kind, name, _op, _values in
                     _comparisons(kwargs["KeyConditionExpression"])}
            self.assertEqual(names, {_INDEX_PARTITION_KEY, _INDEX_SORT_KEY})
        self.assertEqual(readers_for("SPAN#exporter-written-row", CONTRACT), [])

    def test_reader_fallback_operation_list_matches_the_contracts_registry(self):
        """The legacy path still enumerates pks, so its coverage still matters.

        While the index is absent this list is the whole read path, and an
        operation missing from it is a writer nobody can see -- the v1 bug,
        for as long as the fallback is in use.
        """
        metrics = _metrics()
        self.assertEqual(
            set(metrics._ALL_OPERATION_PKS),
            set(_OBSERVATORY["discriminator_values"]),
        )

    def test_contract_registry_still_names_this_repositorys_reader_modules(self):
        """Archaeology, per the contract's namespace_registry_note: kept so a
        module rename here is reflected in the registry other repos read."""
        self.assertEqual(
            sorted(_TEAMWEAVE_READERS),
            sorted(r for r in _OBSERVATORY["readers"] if r.startswith(_REPO_LABEL + ":")),
            "the contract's teamweave readers and this repo's reader modules differ",
        )


# ---------------------------------------------------------------------------
# reader fallback -- the index ships in another stack
# ---------------------------------------------------------------------------


class ReaderIndexFallbackTests(unittest.TestCase):
    """Index absence falls back to the legacy pk path; nothing else does.

    ``SpanTimelineIndex`` is created by the shared stack, and a new GSI is not
    queryable until its backfill completes.  Without this the deploy order of
    two repositories would be load-bearing.  With it, the only thing that must
    never happen is a *different* failure being mistaken for absence: answering
    a throttled or mis-credentialled query out of the legacy partitions would
    report "no traffic" for every writer whose prefix this repo does not list,
    which is a wrong dashboard rather than a broken one.
    """

    _MISSING_INDEX_MESSAGE = (
        "The table does not have the specified index: SpanTimelineIndex"
    )

    def setUp(self):
        _metrics()._index_fallback_warned = False

    def tearDown(self):
        _metrics()._index_fallback_warned = False

    def _queries_with(self, exc, params=None):
        table = _index_failing_table(exc)
        return _capture_queries(_run_agent_metrics(params), table=table)

    def test_validation_exception_naming_the_index_falls_back_to_pk_queries(self):
        queries = self._queries_with(
            _client_error("ValidationException", self._MISSING_INDEX_MESSAGE)
        )
        legacy = _pk_partitions(queries)
        self.assertEqual(
            sorted(legacy),
            sorted(f"OBSERVATORY#{op}" for op in _metrics()._ALL_OPERATION_PKS),
        )

    def test_resource_not_found_falls_back_to_pk_queries(self):
        queries = self._queries_with(
            _client_error("ResourceNotFoundException", "Requested resource not found")
        )
        self.assertTrue(_pk_partitions(queries))

    def test_fallback_still_answers_200_with_the_normal_response_shape(self):
        metrics = _metrics()
        metrics._ddb_table = None
        table = _index_failing_table(
            _client_error("ValidationException", self._MISSING_INDEX_MESSAGE)
        )
        env = {"OBSERVATORY_METRICS_TABLE": "contract-test-table"}
        with patch.dict(os.environ, env, clear=False):
            with patch("boto3.resource") as mock_resource:
                mock_resource.return_value.Table.return_value = table
                resp = metrics.handler({"queryStringParameters": {"operation": "all"}}, None)
        metrics._ddb_table = None

        self.assertEqual(resp["statusCode"], 200)
        body = json.loads(resp["body"])
        self.assertEqual(body["items"], [])
        self.assertEqual(body["count"], 0)

    def test_aggregate_mode_falls_back_too(self):
        queries = self._queries_with(
            _client_error("ValidationException", self._MISSING_INDEX_MESSAGE),
            params={"operation": "all", "aggregate": "by_operation"},
        )
        self.assertTrue(_pk_partitions(queries))

    def test_the_unified_handler_shares_the_fallback(self):
        table = _index_failing_table(
            _client_error("ValidationException", self._MISSING_INDEX_MESSAGE)
        )
        queries = _capture_queries(_run_unified(), table=table)
        self.assertTrue(_pk_partitions(queries))

    def test_fallback_logs_exactly_one_warning_per_process(self):
        metrics = _metrics()
        exc = _client_error("ValidationException", self._MISSING_INDEX_MESSAGE)
        with patch.object(metrics, "log") as mock_log:
            self._queries_with(exc)
            self._queries_with(exc)
            self.assertEqual(mock_log.warning.call_count, 1)
            event, = mock_log.warning.call_args.args
            self.assertIn("index", event)

    def test_throttling_is_not_mistaken_for_a_missing_index(self):
        """A real error must reach the handler's 500, not be answered from the
        legacy partitions -- which for most writers hold nothing."""
        queries = self._queries_with(
            _client_error("ProvisionedThroughputExceededException", "slow down")
        )
        self.assertEqual(_pk_partitions(queries), [], "throttling triggered the fallback")

    def test_a_validation_exception_about_something_else_is_not_index_absence(self):
        queries = self._queries_with(
            _client_error("ValidationException",
                          "ExpressionAttributeValues contains invalid value")
        )
        self.assertEqual(_pk_partitions(queries), [])

    def test_an_unrelated_exception_propagates(self):
        metrics = _metrics()
        with self.assertRaises(RuntimeError):
            metrics.fetch_all_for_aggregate(
                _index_failing_table(RuntimeError("connection reset")),
                operation="all", agent_id=None,
                start_iso=None, end_iso=None, filter_expr=None,
            )

    def test_an_unrelated_exception_surfaces_as_500_not_as_stale_data(self):
        metrics = _metrics()
        metrics._ddb_table = None
        table = _index_failing_table(RuntimeError("connection reset"))
        env = {"OBSERVATORY_METRICS_TABLE": "contract-test-table"}
        with patch.dict(os.environ, env, clear=False):
            with patch("boto3.resource") as mock_resource:
                mock_resource.return_value.Table.return_value = table
                resp = metrics.handler({"queryStringParameters": {"operation": "all"}}, None)
        metrics._ddb_table = None

        self.assertEqual(resp["statusCode"], 500)
        self.assertEqual(table.query.call_args_list[-1].kwargs.get("IndexName"), _INDEX_NAME)

    def test_index_absence_detector_ignores_plain_exceptions(self):
        metrics = _metrics()
        for exc in (RuntimeError("boom"), ValueError("nope"), KeyError("pk")):
            self.assertFalse(metrics._is_index_missing_error(exc), exc)


# ---------------------------------------------------------------------------
# what v2 did and did not resolve
# ---------------------------------------------------------------------------


class NamespaceRegistryIsNowArchaeologyTests(unittest.TestCase):
    """Under v1 these namespaces were a live defect; under v2 they are history.

    ``SPAN`` (the shared ``mcp_observatory.aws.DynamoDBSpanExporter``),
    ``WRAPPER`` and ``INVOCATION`` (toolweave) are partitions no reader here
    enumerates.  That used to mean their rows were durable, billable and
    invisible.  It no longer does: reads go through ``SpanTimelineIndex``, so
    those rows appear on these dashboards as soon as they carry ``span_date``
    and ``timestamp`` -- and the contract now marks every registry entry
    ``legacy-informational`` to say so.

    What is *not* resolved: the index is not retroactive.  Rows written before
    span_date existed are in no index and remain reachable only by their
    original pk, which for these namespaces means no reader at all.  These
    tests pin that distinction so the registry's demotion cannot be read as a
    claim that the old rows came back.
    """

    def test_every_registry_entry_is_marked_legacy_informational(self):
        for namespace, entry in CONTRACT["namespace_registry"].items():
            self.assertEqual(entry["status"], "legacy-informational", namespace)

    def test_readers_for_answers_a_historical_question_only(self):
        # Still empty for the unread namespaces -- but reachability no longer
        # depends on it, which is what the reader tests above assert.
        for namespace in ("SPAN", "WRAPPER", "INVOCATION"):
            self.assertEqual(readers_for(f"{namespace}#anything", CONTRACT), [])
        self.assertNotEqual(readers_for("OBSERVATORY#invoke_agent", CONTRACT), [])

    def test_i5_is_recorded_as_superseded_rather_than_dropped(self):
        i5 = [line for line in CONTRACT["invariants"] if line.startswith("I5")]
        self.assertEqual(len(i5), 1, CONTRACT["invariants"])
        self.assertIn("superseded", i5[0])

    def test_the_index_key_invariants_exist(self):
        numbers = {line.split(":", 1)[0] for line in CONTRACT["invariants"]}
        self.assertTrue({"I6", "I7", "I8"} <= numbers, numbers)

    def test_screenweave_operation_is_reachable_without_being_registered(self):
        """A discriminator outside _ALL_OPERATION_PKS had no reader under v1.

        It still has none by the registry's reckoning, but the index path does
        not consult the registry, so its rows are returned like any others.
        """
        self.assertEqual(readers_for("OBSERVATORY#screenshot", CONTRACT), [])
        self.assertNotIn("screenshot", _OBSERVATORY["discriminator_values"])


class VendoredCopyTests(unittest.TestCase):
    """The vendored copy must be the producer's, at the version it claims."""

    def test_contract_is_at_the_version_this_repository_was_migrated_for(self):
        self.assertEqual(CONTRACT["version"], "2.0.0")

    def test_contract_declares_the_index_the_readers_use(self):
        self.assertEqual(_GSI["projection"], "ALL")
        self.assertEqual(
            (_INDEX_NAME, _INDEX_PARTITION_KEY, _INDEX_SORT_KEY),
            ("SpanTimelineIndex", "span_date", "timestamp"),
        )


if __name__ == "__main__":
    unittest.main()
