"""
tests/test_tool_registry.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for tool_registry: execute_tool, execute_pre_tools, execute_post_tools,
and source_key argument resolution.
"""

import unittest

from src.orchestrator.tool_registry import (
    TOOL_REGISTRY,
    _build_tool_args,
    _resolve_source_key,
    execute_post_tools,
    execute_pre_tools,
    execute_tool,
    register_tool,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _fake_tool(text: str = "") -> dict:
    return {"fake_result": text.upper()}


def _failing_tool(**kwargs):
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# execute_tool
# ---------------------------------------------------------------------------


class ExecuteToolTests(unittest.TestCase):
    def setUp(self):
        register_tool("fake", _fake_tool)
        register_tool("failing", _failing_tool)

    def test_calls_registered_tool(self):
        result = execute_tool("fake", {"text": "hello"})
        self.assertEqual(result, {"fake_result": "HELLO"})

    def test_raises_key_error_for_unknown_tool(self):
        with self.assertRaises(KeyError) as ctx:
            execute_tool("nonexistent_tool_xyz", {})
        self.assertIn("nonexistent_tool_xyz", str(ctx.exception))

    def test_parse_document_is_registered(self):
        self.assertIn("parse_document", TOOL_REGISTRY)

    def test_reconstruct_document_is_registered(self):
        self.assertIn("reconstruct_document", TOOL_REGISTRY)


# ---------------------------------------------------------------------------
# _resolve_source_key
# ---------------------------------------------------------------------------


class ResolveSourceKeyTests(unittest.TestCase):
    def _inputs(self):
        return {
            "request": {"document_text": "My CV text", "job_description": "Senior Eng"},
            "analyzer": {"sections": [{"index": 0, "header": "SUMMARY"}]},
        }

    def test_resolves_single_key(self):
        result = _resolve_source_key("analyzer", self._inputs())
        self.assertIsInstance(result, dict)
        self.assertIn("sections", result)

    def test_resolves_dotted_path(self):
        result = _resolve_source_key("request.document_text", self._inputs())
        self.assertEqual(result, "My CV text")

    def test_returns_none_for_missing_root(self):
        result = _resolve_source_key("missing_key", self._inputs())
        self.assertIsNone(result)

    def test_returns_none_for_missing_leaf(self):
        result = _resolve_source_key("request.nonexistent", self._inputs())
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# _build_tool_args
# ---------------------------------------------------------------------------


class BuildToolArgsTests(unittest.TestCase):
    def _inputs(self):
        return {"request": {"document_text": "CV here"}}

    def test_resolves_source_key_to_leaf_param(self):
        tool_cfg = {"name": "parse_document", "args": {"source_key": "request.document_text"}}
        args = _build_tool_args(tool_cfg, self._inputs())
        # leaf param name = "document_text"
        self.assertIn("document_text", args)
        self.assertEqual(args["document_text"], "CV here")
        self.assertNotIn("source_key", args)

    def test_passes_through_non_source_key_args(self):
        tool_cfg = {"name": "fake", "args": {"separator": "\n---\n"}}
        args = _build_tool_args(tool_cfg, self._inputs())
        self.assertEqual(args["separator"], "\n---\n")

    def test_empty_args(self):
        tool_cfg = {"name": "fake"}
        args = _build_tool_args(tool_cfg, self._inputs())
        self.assertEqual(args, {})


# ---------------------------------------------------------------------------
# execute_pre_tools
# ---------------------------------------------------------------------------


class ExecutePreToolsTests(unittest.TestCase):
    def setUp(self):
        register_tool("fake", _fake_tool)
        register_tool("failing", _failing_tool)

    def _base_inputs(self):
        return {"request": {"document_text": "SUMMARY\nGreat engineer."}}

    def test_injects_tool_result_under_tool_results(self):
        step_def = {
            "step": "analyzer",
            "pre_tools": [{"name": "parse_document", "args": {"source_key": "request.document_text"}}],
        }
        inputs = self._base_inputs()
        result = execute_pre_tools(step_def, inputs)
        self.assertIn("tool_results", result)
        self.assertIn("parse_document", result["tool_results"])
        # parse_document returns sections list
        self.assertIn("sections", result["tool_results"]["parse_document"])

    def test_no_pre_tools_returns_inputs_unchanged(self):
        step_def = {"step": "analyzer"}
        inputs = {"request": {}}
        result = execute_pre_tools(step_def, inputs)
        self.assertEqual(result, {"request": {}})

    def test_failing_pre_tool_is_skipped_gracefully(self):
        step_def = {
            "step": "test",
            "pre_tools": [{"name": "failing", "args": {}}],
        }
        inputs = self._base_inputs()
        # Should not raise; failing tool is skipped
        result = execute_pre_tools(step_def, inputs)
        self.assertNotIn("failing", result.get("tool_results", {}))

    def test_multiple_pre_tools_all_injected(self):
        register_tool("fake_a", lambda **kw: {"a": 1})
        register_tool("fake_b", lambda **kw: {"b": 2})
        step_def = {
            "step": "test",
            "pre_tools": [
                {"name": "fake_a", "args": {}},
                {"name": "fake_b", "args": {}},
            ],
        }
        result = execute_pre_tools(step_def, {})
        self.assertIn("fake_a", result["tool_results"])
        self.assertIn("fake_b", result["tool_results"])


# ---------------------------------------------------------------------------
# execute_post_tools
# ---------------------------------------------------------------------------


class ExecutePostToolsTests(unittest.TestCase):
    def setUp(self):
        register_tool("fake", _fake_tool)
        register_tool("failing", _failing_tool)

    def test_merges_tool_result_into_out_json(self):
        sections = [
            {"index": 0, "header": "SUMMARY", "content": "Great engineer."},
            {"index": 1, "header": "SKILLS", "content": "Python, AWS"},
        ]
        out_json = {"sections": sections, "summary_of_changes": "Updated keywords."}
        step_def = {
            "step": "formatter",
            "post_tools": [
                {"name": "reconstruct_document", "args": {"source_key": "formatter"}}
            ],
        }
        step_inputs = {"request": {}, "formatter": out_json}
        result = execute_post_tools(step_def, out_json, step_inputs)
        self.assertIn("document_text", result)
        self.assertIn("SUMMARY", result["document_text"])

    def test_no_post_tools_returns_out_json_unchanged(self):
        out_json = {"sections": [], "summary_of_changes": "none"}
        result = execute_post_tools({"step": "formatter"}, out_json, {})
        self.assertIs(result, out_json)

    def test_failing_post_tool_skipped_gracefully(self):
        out_json = {"sections": []}
        step_def = {
            "step": "test",
            "post_tools": [{"name": "failing", "args": {}}],
        }
        result = execute_post_tools(step_def, out_json, {})
        # Must not raise; out_json returned as-is
        self.assertIn("sections", result)

    def test_post_tool_does_not_overwrite_existing_fields(self):
        # reconstruct_document adds document_text; existing fields preserved
        sections = [{"index": 0, "header": "SUMMARY", "content": "Text."}]
        out_json = {
            "sections": sections,
            "summary_of_changes": "keep me",
        }
        step_def = {
            "step": "formatter",
            "post_tools": [{"name": "reconstruct_document", "args": {"source_key": "formatter"}}],
        }
        step_inputs = {"formatter": out_json}
        result = execute_post_tools(step_def, out_json, step_inputs)
        self.assertEqual(result["summary_of_changes"], "keep me")


if __name__ == "__main__":
    unittest.main()
