import json
import unittest

from src.orchestrator.structured_transform import (
    _coerce_to_template,
    normalize_target_schema,
    transform_json_to_schema,
)


class _FakeBody:
    def __init__(self, payload: str):
        self._payload = payload

    def read(self):
        return self._payload.encode("utf-8")


class _FakeClient:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def invoke_model(self, modelId, body):
        request_payload = json.loads(body)
        assert request_payload["messages"][0]["role"] == "user"
        result = {"content": [{"text": self.response_text}]}
        return {"body": _FakeBody(json.dumps(result))}


class StructuredTransformTests(unittest.TestCase):
    def test_transform_json_to_schema_supports_json_schema_targets(self):
        input_json = {"first_name": "John", "last_name": "Doe", "phone": "512-000-0000"}
        target_schema = {
            "type": "object",
            "properties": {
                "fullName": {"type": "string"},
                "contact": {
                    "type": "object",
                    "properties": {
                        "phone": {"type": "string"},
                        "location": {"type": "string"},
                    },
                },
            },
        }
        model_output = json.dumps({"fullName": "John Doe", "contact": {"phone": "512-000-0000"}})

        transformed = transform_json_to_schema(input_json, target_schema, client=_FakeClient(model_output))

        self.assertEqual(
            transformed,
            {"fullName": "John Doe", "contact": {"phone": "512-000-0000", "location": ""}},
        )

    def test_normalize_target_schema(self):
        self.assertEqual(
            normalize_target_schema({"type": "object", "properties": {"name": {"type": "string"}}}),
            {"name": "string"},
        )

    def test_coerce_to_template_converts_types(self):
        template = {"name": "string", "active": "boolean", "score": "integer"}
        value = {"name": 42, "active": "yes", "score": "9"}

        transformed = _coerce_to_template(value, template)

        self.assertEqual(transformed, {"name": "42", "active": True, "score": 9})


if __name__ == "__main__":
    unittest.main()
