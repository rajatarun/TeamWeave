import importlib
import sys
import types
import unittest
from unittest.mock import patch


class RagContextTests(unittest.TestCase):
    def test_skips_rag_when_explicit_rag_feature_disabled(self):
        fake_boto3 = types.ModuleType("boto3")
        fake_boto3.client = lambda *_args, **_kwargs: object()

        fake_psycopg = types.ModuleType("psycopg")
        fake_psycopg.Cursor = object
        fake_psycopg.connect = lambda *_args, **_kwargs: None

        fake_sql = types.ModuleType("sql")
        fake_sql.SQL = lambda s: s
        fake_sql.Identifier = lambda s: s
        fake_psycopg.sql = fake_sql

        fake_db = types.ModuleType("src.orchestrator.db")
        fake_db.DbDao = object

        with patch.dict(sys.modules, {"boto3": fake_boto3, "psycopg": fake_psycopg, "src.orchestrator.db": fake_db}):
            rag = importlib.import_module("src.orchestrator.rag")
            rag = importlib.reload(rag)

            request_obj = {"topic": "python", "objective": "learn", "audience": "dev"}
            team_globals = {
                "features": {"explicit_rag": False},
                "rag": {"mode": "explicit", "top_k": 5},
            }

            with patch.object(rag, "retrieve_from_vector_store") as mock_retrieve:
                result = rag.get_rag_context(request_obj, team_globals, owner="Jane Doe")

        self.assertEqual(result, "")
        mock_retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
