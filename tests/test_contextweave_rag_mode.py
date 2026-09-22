"""Unit tests for the "contextweave" RAG mode and its config validation."""
import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch

_URL = "https://contextweave.example.com"

_PAYLOAD = {
    "queryId": "q-abc",
    "answer": "He runs GraphRAG on Neptune plus pgvector.",
    "sources": [{"file": "architecture.md", "excerpt": "Neptune holds routing weights.", "weight": 1.0}],
    "confidence": 0.88,
    "questionType": "architecture",
    "cacheHit": True,
}


def _load_rag():
    """Import src.orchestrator.rag with boto3/psycopg stubbed (see test_rag.py)."""
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
        return importlib.reload(importlib.import_module("src.orchestrator.rag"))


_GLOBALS = {
    "features": {"explicit_rag": True},
    "rag": {"mode": "contextweave", "top_k": 4},
}
_REQUEST = {"topic": "graph rag", "objective": "explain", "audience": "engineers"}


class ContextWeaveModeTests(unittest.TestCase):
    def setUp(self):
        self.rag = _load_rag()

    def test_maps_response_into_prompt_builder_rag_context_shape(self):
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": _URL}), patch.object(
            self.rag.contextweave_client, "query_expertise", return_value=_PAYLOAD
        ) as mock_query:
            context, meta = self.rag.get_rag_context_with_meta(_REQUEST, _GLOBALS, owner="Tarun")

        # Same "[RAG #n] SOURCE: ..." block list explicit (pgvector) mode emits.
        self.assertEqual(
            context.splitlines(),
            [
                "[RAG #1] SOURCE: contextweave:answer (confidence 0.88)",
                "He runs GraphRAG on Neptune plus pgvector.",
                "---",
                "[RAG #2] SOURCE: architecture.md (weight 1.00)",
                "Neptune holds routing weights.",
                "---",
            ],
        )
        mock_query.assert_called_once_with("graph rag explain engineers", 4)

    def test_carries_query_id_and_provenance_in_meta(self):
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": _URL}), patch.object(
            self.rag.contextweave_client, "query_expertise", return_value=_PAYLOAD
        ):
            _context, meta = self.rag.get_rag_context_with_meta(_REQUEST, _GLOBALS, owner="Tarun")

        self.assertEqual(meta["provider"], "contextweave")
        self.assertEqual(meta["query_id"], "q-abc")
        self.assertEqual(meta["confidence"], 0.88)
        self.assertEqual(meta["question_type"], "architecture")
        self.assertTrue(meta["cache_hit"])

    def test_degrades_to_empty_context_when_knowledge_layer_fails(self):
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": _URL}), patch.object(
            self.rag.contextweave_client, "query_expertise", return_value=None
        ):
            context, meta = self.rag.get_rag_context_with_meta(_REQUEST, _GLOBALS, owner="Tarun")

        self.assertEqual(context, "")
        self.assertEqual(meta, {})

    def test_degrades_to_empty_context_when_url_missing(self):
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": ""}), patch.object(
            self.rag.contextweave_client, "query_expertise"
        ) as mock_query:
            context, meta = self.rag.get_rag_context_with_meta(_REQUEST, _GLOBALS, owner="Tarun")

        self.assertEqual((context, meta), ("", {}))
        mock_query.assert_not_called()

    def test_drops_answer_below_min_confidence(self):
        team_globals = {
            "features": {"explicit_rag": True},
            "rag": {"mode": "contextweave", "min_confidence": 0.9},
        }
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": _URL}), patch.object(
            self.rag.contextweave_client, "query_expertise", return_value=_PAYLOAD
        ):
            context, meta = self.rag.get_rag_context_with_meta(_REQUEST, team_globals, owner="Tarun")

        self.assertEqual((context, meta), ("", {}))

    def test_respects_the_explicit_rag_feature_switch(self):
        team_globals = {"features": {"explicit_rag": False}, "rag": {"mode": "contextweave"}}
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": _URL}), patch.object(
            self.rag.contextweave_client, "query_expertise"
        ) as mock_query:
            self.assertEqual(self.rag.get_rag_context_with_meta(_REQUEST, team_globals, owner="T"), ("", {}))
        mock_query.assert_not_called()

    def test_existing_modes_are_unchanged(self):
        # explicit mode still renders hits as blocks and history still prefixes.
        explicit_globals = {"features": {"explicit_rag": True}, "rag": {"mode": "explicit", "top_k": 2}}
        with patch.dict(os.environ, {"VECTOR_DB_TABLE": "rag_chunks"}), patch.object(
            self.rag,
            "retrieve_from_vector_store",
            return_value=[{"source": "doc.md", "text": "chunk"}],
        ):
            context, meta = self.rag.get_rag_context_with_meta(_REQUEST, explicit_globals, owner="T")
        self.assertEqual(context.splitlines(), ["[RAG #1] SOURCE: doc.md", "chunk", "---"])
        self.assertEqual(meta, {})

        class _Dao:
            def list_completed_topic_levels(self, owner, limit):
                return ["python/basics"]

        history_globals = {"features": {"explicit_rag": True}, "rag": {"mode": "history"}}
        context, meta = self.rag.get_rag_context_with_meta(_REQUEST, history_globals, owner="T", dao=_Dao())
        self.assertEqual(context, "COMPLETED_TASKS_HISTORY:\n- python/basics")
        self.assertEqual(meta, {})

    def test_get_rag_context_still_returns_a_bare_string(self):
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": _URL}), patch.object(
            self.rag.contextweave_client, "query_expertise", return_value=_PAYLOAD
        ):
            context = self.rag.get_rag_context(_REQUEST, _GLOBALS, owner="Tarun")
        self.assertIsInstance(context, str)
        self.assertIn("[RAG #1] SOURCE: contextweave:answer", context)


class PromptBuilderIntegrationTests(unittest.TestCase):
    """The mapped context must land in the prompt's RAG_CONTEXT block, and the
    retrieval provenance must not leak into the prompt."""

    def _team_and_agent(self):
        from src.orchestrator.models import AgentConfig, BedrockRef, TeamConfig, TeamGlobals

        globals_obj = TeamGlobals(
            north_star="ship",
            default_channel="linkedin",
            hard_constraints=[],
            features={"explicit_rag": True},
            rag={"mode": "contextweave"},
            artifact_store={},
            revision={},
        )
        agent = AgentConfig(
            id="strategist",
            name="Strategist",
            bedrock=BedrockRef(agentId="a", aliasId="b"),
            goal_template="Draft a strategy.",
            schema_ref="strategy_pack_v1",
        )
        return TeamConfig(team={}, globals=globals_obj, agents=[agent], workflow=[], schemas={}), agent

    def test_contextweave_context_renders_as_verified_experience(self):
        from src.orchestrator.contextweave_client import format_rag_context
        from src.orchestrator.prompt_builder import build_prompt

        team, agent = self._team_and_agent()
        rag_context = format_rag_context(_PAYLOAD)
        step_inputs = {"request": _REQUEST, "rag_context": rag_context, "rag_meta": {"query_id": "q-abc"}}

        prompt = build_prompt(team, agent, step_inputs, {}, rag_context, "", "")

        self.assertIn("VERIFIED_EXPERIENCE", prompt)
        self.assertIn("[RAG #1] SOURCE: contextweave:answer", prompt)
        self.assertIn("Neptune holds routing weights.", prompt)

    def test_rag_meta_is_not_exposed_to_the_agent(self):
        from src.orchestrator.prompt_builder import build_prompt

        team, agent = self._team_and_agent()
        step_inputs = {"request": _REQUEST, "rag_context": "ctx", "rag_meta": {"query_id": "q-abc"}}

        prompt = build_prompt(team, agent, step_inputs, {}, "ctx", "", "")

        self.assertNotIn("q-abc", prompt)
        self.assertNotIn("STEP_INPUTS_JSON", prompt)


class RagConfigValidationTests(unittest.TestCase):
    def setUp(self):
        fake_boto3 = types.ModuleType("boto3")
        fake_boto3.client = lambda *_args, **_kwargs: object()
        with patch.dict(sys.modules, {"boto3": fake_boto3}):
            self.loader = importlib.reload(importlib.import_module("src.orchestrator.config_loader"))

    def test_accepts_contextweave_mode_with_url_and_options(self):
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": _URL}):
            self.loader._validate_rag(
                {"mode": "contextweave", "top_k": 6, "min_confidence": 0.5}, "team", "v1"
            )

    def test_rejects_contextweave_mode_without_url(self):
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": ""}):
            with self.assertRaises(ValueError) as ctx:
                self.loader._validate_rag({"mode": "contextweave"}, "tarun_visibility_team", "v1")
        self.assertIn("CONTEXTWEAVE_URL", str(ctx.exception))

    def test_rejects_non_numeric_options(self):
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": _URL}):
            with self.assertRaises(ValueError) as ctx:
                self.loader._validate_rag({"mode": "contextweave", "min_confidence": "high"}, "t", "v1")
        self.assertIn("min_confidence", str(ctx.exception))

    def test_existing_modes_still_load(self):
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": ""}):
            for mode in ("explicit", "history", "kb", "none"):
                self.loader._validate_rag({"mode": mode, "top_k": 1}, "t", "v1")

    def test_unknown_mode_is_warned_not_raised(self):
        with patch.dict(os.environ, {"CONTEXTWEAVE_URL": ""}):
            self.loader._validate_rag({"mode": "telepathy"}, "t", "v1")


if __name__ == "__main__":
    unittest.main()
