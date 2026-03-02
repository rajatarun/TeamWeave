import unittest

from src.orchestrator.json_utils import extract_json_payload


class ExtractJsonPayloadTests(unittest.TestCase):
    def test_parses_plain_json(self):
        self.assertEqual(extract_json_payload('{"status":"ok"}'), {"status": "ok"})

    def test_parses_json_fence_block(self):
        raw = """Here's the result:\n```json\n{\"status\":\"ok\"}\n```"""
        self.assertEqual(extract_json_payload(raw), {"status": "ok"})

    def test_parses_embedded_json_object(self):
        raw = "Result summary: {\"status\":\"ok\",\"count\":2} End."
        self.assertEqual(extract_json_payload(raw), {"status": "ok", "count": 2})

    def test_raises_when_no_json(self):
        with self.assertRaises(ValueError):
            extract_json_payload("no payload")


if __name__ == "__main__":
    unittest.main()
