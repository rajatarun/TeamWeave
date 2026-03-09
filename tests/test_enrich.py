import sys
import types
import unittest

# enrich imports boto3 at module load time; provide a lightweight stub for unit tests.
sys.modules.setdefault("boto3", types.SimpleNamespace(client=lambda *args, **kwargs: None))

from src.orchestrator.enrich import _build_prompt, _trim_for_prompt


class EnrichPromptSafetyTests(unittest.TestCase):
    def test_trim_for_prompt_handles_broken_unicode_escape_sequences(self):
        value = "start " + "\\u12" * 1500
        trimmed = _trim_for_prompt(value, max_str_len=100)
        self.assertTrue(trimmed.endswith("...[truncated]"))
        self.assertIsInstance(trimmed, str)

    def test_build_prompt_never_raises_on_large_nested_inputs(self):
        step_inputs = {
            "payload": {
                "text": "prefix " + "\\u12" * 2000,
                "nested": [{"a": "x" * 5000}] * 30,
            }
        }

        prompt = _build_prompt(
            agent_name="test-agent",
            schema_ref="schemas/test",
            raw_output={"status": "ok"},
            schema={"type": "object"},
            has_placeholders=True,
            step_inputs=step_inputs,
        )

        self.assertIn("PIPELINE CONTEXT", prompt)
        self.assertIn("...[truncated]", prompt)
        self.assertIn("more items truncated", prompt)


if __name__ == "__main__":
    unittest.main()
