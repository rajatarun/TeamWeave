"""DynamoDB stores no floats, and a step record carries arbitrary payloads.

A real run died here: `_contextweave_context` puts the knowledge layer's
`confidence` -- a float -- into the meta the worker persists as
`inputs_json.rag_meta`, and boto3's resource refuses to serialise it:

    Failed DynamoDB put_item for table=... pk=RUN#... sk=STEP#...director:
    Float types are not supported. Use Decimal types instead.

The first step of the pipeline, after the retrieval and the agent turn had
both already been paid for.

The fakes here serialise with boto3's *real* `TypeSerializer` rather than a
hand-written rule about which values it rejects. That is the component that
raised in production, and a fake written from recollection of its behaviour
would agree with whatever the code does.
"""
import math
import unittest
from decimal import Decimal

from boto3.dynamodb.types import TypeSerializer

from src.orchestrator.db import DbDao


class FakeTable:
    """Accepts exactly what DynamoDB accepts, by asking boto3."""

    def __init__(self):
        self.items = []

    def put_item(self, Item):
        TypeSerializer().serialize(Item)
        self.items.append(Item)
        return {}


def dao_with_fake_table():
    dao = DbDao.__new__(DbDao)
    dao.table_name = "fake-table"
    dao.table = FakeTable()
    return dao


# ── the production failure, reproduced ──────────────────────────────────────

class TestTheRunThatDied(unittest.TestCase):
    def test_a_step_carrying_contextweave_rag_meta_is_written(self):
        dao = dao_with_fake_table()
        dao.put_step(
            run_id="74b51bc5-aa65-4b4a-88aa-6ac5f8cfc28b",
            step_id="TVT_DEPT-001_PBM-001_director",
            status="SUCCEEDED",
            inputs={
                "rag_meta": {
                    "provider": "contextweave",
                    "query_id": "q-1",
                    "confidence": 0.8312,
                    "question_type": "skill_depth",
                    "cache_hit": False,
                }
            },
            output={"brief": "..."},
            error=None,
            artifact_uri=None,
        )
        stored = dao.table.items[0]
        self.assertEqual(stored["inputs_json"]["rag_meta"]["confidence"], Decimal("0.8312"))

    def test_the_meta_the_retrieval_layer_produces_really_does_hold_a_float(self):
        """The other half: a fix in the DAO is only needed while the producer
        emits a float. If ContextWeave's confidence ever arrives as a string
        the DAO check above would still pass while testing nothing."""
        from tests.test_contextweave_rag_mode import _load_rag

        rag = _load_rag()          # boto3/psycopg stubbed, as that module does
        contextweave_client = rag.contextweave_client

        payload = {
            "queryId": "q-1",
            "confidence": 0.82,
            "questionType": "skill_depth",
            "cacheHit": False,
            "answer": "a",
            "sources": [],
        }
        orig_configured = contextweave_client.is_configured
        orig_query = contextweave_client.query_expertise
        contextweave_client.is_configured = lambda: True
        contextweave_client.query_expertise = lambda q, k: payload
        try:
            _context, meta = rag._contextweave_context({"summary": "s"}, {"min_confidence": 0.5}, 6)
        finally:
            contextweave_client.is_configured = orig_configured
            contextweave_client.query_expertise = orig_query
        self.assertIsInstance(meta["confidence"], float)


# ── the general rule, since the payloads are arbitrary ──────────────────────

class TestArbitraryPayloads(unittest.TestCase):
    def test_a_float_anywhere_in_an_agent_output_is_written(self):
        """An agent answering with a score would have failed identically."""
        dao = dao_with_fake_table()
        dao.put_step(
            run_id="r", step_id="s", status="SUCCEEDED",
            inputs={},
            output={"drafts": [{"text": "a", "score": 0.91}], "meta": {"temperature": 0.7}},
            error=None, artifact_uri=None,
        )
        stored = dao.table.items[0]["output_json"]
        self.assertEqual(stored["drafts"][0]["score"], Decimal("0.91"))
        self.assertEqual(stored["meta"]["temperature"], Decimal("0.7"))

    def test_run_meta_goes_through_the_same_conversion(self):
        dao = dao_with_fake_table()
        dao.put_run_meta("r", "RUNNING", {"elapsed": 1.25})
        self.assertEqual(dao.table.items[0]["data"]["elapsed"], Decimal("1.25"))

    def test_the_binary_expansion_is_not_what_is_stored(self):
        """`Decimal(0.1)` is 55 significant digits; DynamoDB accepts 38, so
        converting via the float itself trades this failure for a later one."""
        dao = dao_with_fake_table()
        dao.put_step(run_id="r", step_id="s", status="OK", inputs={"x": 0.1},
                     output=None, error=None, artifact_uri=None)
        stored = dao.table.items[0]["inputs_json"]["x"]
        self.assertEqual(stored, Decimal("0.1"))
        self.assertLess(len(stored.as_tuple().digits), 38)

    def test_non_finite_floats_do_not_reach_the_table_as_numbers(self):
        """json.loads accepts NaN and Infinity, so an agent's output can carry
        one. Decimal('NaN') is refused by the same serializer."""
        dao = dao_with_fake_table()
        dao.put_step(run_id="r", step_id="s", status="OK",
                     inputs={"a": float("nan"), "b": float("inf"), "c": float("-inf")},
                     output=None, error=None, artifact_uri=None)
        stored = dao.table.items[0]["inputs_json"]
        for key in ("a", "b", "c"):
            self.assertIsInstance(stored[key], str, key)

    def test_booleans_stay_booleans(self):
        """bool is a subclass of int, not float -- but a conversion written
        against `isinstance(x, (int, float))` would store cache_hit as 0."""
        dao = dao_with_fake_table()
        dao.put_step(run_id="r", step_id="s", status="OK",
                     inputs={"cache_hit": False, "hit": True, "n": 3},
                     output=None, error=None, artifact_uri=None)
        stored = dao.table.items[0]["inputs_json"]
        self.assertIs(stored["cache_hit"], False)
        self.assertIs(stored["hit"], True)
        self.assertIsInstance(stored["n"], int)
        self.assertNotIsInstance(stored["n"], Decimal)

    def test_decimals_already_supplied_are_left_alone(self):
        dao = dao_with_fake_table()
        dao.put_step(run_id="r", step_id="s", status="OK", inputs={"d": Decimal("1.5")},
                     output=None, error=None, artifact_uri=None)
        self.assertEqual(dao.table.items[0]["inputs_json"]["d"], Decimal("1.5"))


class TestEveryWriteGoesThroughTheConversion(unittest.TestCase):
    """The fix is one chokepoint, so a write that bypasses it is the whole
    defect back. Today every put in the DAO is `_safe_put`; a new method
    calling `self.table.put_item` (or a batch writer) directly would serialise
    a raw float again and fail only at run time, on whichever payload happens
    to carry one."""

    def test_no_write_bypasses_safe_put(self):
        import ast
        import inspect

        from src.orchestrator import db as db_module

        tree = ast.parse(inspect.getsource(db_module))
        offenders = []
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                if fn.name == "_safe_put":
                    continue
                for node in ast.walk(fn):
                    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                        continue
                    if node.func.attr in {"put_item", "batch_writer", "update_item", "transact_write_items"}:
                        target = node.func.value
                        # self.table.put_item(...) -- a write against the table
                        if isinstance(target, ast.Attribute) and target.attr == "table":
                            offenders.append(f"{cls.name}.{fn.name} -> {node.func.attr}")
        self.assertEqual(
            offenders, [],
            "these write to the table without going through _safe_put, so floats "
            "in their payloads are not converted: " + ", ".join(offenders),
        )


class TestTheFakeWouldHaveCaughtIt(unittest.TestCase):
    """A fake that accepts anything reports success for the broken code too."""

    def test_the_fake_table_rejects_a_raw_float(self):
        table = FakeTable()
        with self.assertRaises(TypeError):
            table.put_item(Item={"pk": "p", "sk": "s", "x": 0.5})

    def test_the_fake_table_rejects_a_nan_decimal(self):
        table = FakeTable()
        with self.assertRaises(TypeError):
            table.put_item(Item={"pk": "p", "sk": "s", "x": Decimal("NaN")})


if __name__ == "__main__":
    unittest.main()
