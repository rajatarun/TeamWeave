import unittest

from src.orchestrator.json_utils import build_standard_response, extract_json_payload


class ExtractJsonPayloadTests(unittest.TestCase):
    def test_parses_plain_json(self):
        self.assertEqual(extract_json_payload('{"status":"ok"}'), {"status": "ok"})

    def test_parses_json_fence_block(self):
        raw = """Here's the result:\n```json\n{\"status\":\"ok\"}\n```"""
        self.assertEqual(extract_json_payload(raw), {"status": "ok"})

    def test_parses_embedded_json_object(self):
        raw = "Result summary: {\"status\":\"ok\",\"count\":2} End."
        self.assertEqual(extract_json_payload(raw), {"status": "ok", "count": 2})

    def test_parses_json_with_escaped_newline_tab_artifacts(self):
        raw = "{\n\t\"status\":\"ok\",\n\t\"count\":2\n}"
        self.assertEqual(extract_json_payload(raw), {"status": "ok", "count": 2})

    def test_raises_when_no_json(self):
        with self.assertRaises(ValueError):
            extract_json_payload("no payload")

    def test_build_standard_response(self):
        payload = build_standard_response("raw output", "unable to locate valid JSON payload in model response")
        self.assertEqual(payload["status"], "fallback_response")
        self.assertEqual(payload["data"]["content"], "raw output")
        self.assertTrue(payload["_meta"]["coerced_from_non_json"])


if __name__ == "__main__":
    unittest.main()
