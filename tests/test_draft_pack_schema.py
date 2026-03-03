import json
from pathlib import Path
import unittest

from src.orchestrator.schema_validate import validate_or_unwrap_output


class DraftPackSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema_path = Path("config/examples/schemas/draft_pack_v1.json")
        cls.schema = json.loads(schema_path.read_text(encoding="utf-8"))

    def test_accepts_post_field(self):
        payload = {
            "drafts": [
                {
                    "variant": "A",
                    "post": "Hello LinkedIn",
                    "hashtags": ["#ai"],
                }
            ]
        }
        self.assertEqual(validate_or_unwrap_output(payload, self.schema), payload)

    def test_accepts_linkedin_post_and_optional_metadata(self):
        payload = {
            "drafts": [
                {
                    "variant": "B",
                    "linkedin_post": "Hello network",
                    "hashtags": ["#cloud"],
                    "char_count_estimate": 1200,
                    "cta_question": "What has worked for you?",
                }
            ]
        }
        self.assertEqual(validate_or_unwrap_output(payload, self.schema), payload)


if __name__ == "__main__":
    unittest.main()
