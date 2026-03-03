import unittest

from jsonschema.exceptions import ValidationError

from src.orchestrator.schema_validate import validate_or_unwrap_output


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "goal": {"type": "string"},
        "audience": {"type": "string"},
    },
    "required": ["goal", "audience"],
}


class SchemaValidateTests(unittest.TestCase):
    def test_validate_or_unwrap_output_accepts_direct_object(self):
        payload = {"goal": "Increase visibility", "audience": "Founders"}
        self.assertEqual(validate_or_unwrap_output(payload, SCHEMA), payload)

    def test_validate_or_unwrap_output_unwraps_single_key_envelope(self):
        payload = {
            "creative_brief": {
                "goal": "Increase visibility",
                "audience": "Founders",
            }
        }
        self.assertEqual(
            validate_or_unwrap_output(payload, SCHEMA),
            {"goal": "Increase visibility", "audience": "Founders"},
        )

    def test_validate_or_unwrap_output_raises_when_envelope_invalid(self):
        payload = {"creative_brief": {"goal": "Increase visibility"}}
        with self.assertRaises(ValidationError):
            validate_or_unwrap_output(payload, SCHEMA)


if __name__ == "__main__":
    unittest.main()
