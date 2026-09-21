import json
import unittest

from src.orchestrator.structured_transform import (
    MODEL_ID,
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
    """A Bedrock client that speaks Converse.

    The transform used InvokeModel, whose request body is defined by the model
    *provider*: an Anthropic-shaped body is malformed for Nova, so the model id
    and the payload shape were coupled. Converse normalises that, and this fake
    asserts the Converse shape so a regression to a provider-specific body
    fails here rather than as a ValidationException in production.
    """

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_model_id = None
        self.last_request = None

    def converse(self, *, modelId, messages, inferenceConfig=None, **kwargs):
        self.last_model_id = modelId
        self.last_request = {"messages": messages, "inferenceConfig": inferenceConfig}
        assert messages[0]["role"] == "user"
        # Converse content is a list of blocks, never a bare string.
        assert isinstance(messages[0]["content"], list), "Converse takes content blocks"
        assert messages[0]["content"][0]["text"]
        return {"output": {"message": {"content": [{"text": self.response_text}]}}}

    def invoke_model(self, **_kwargs):  # pragma: no cover - must not be reached
        raise AssertionError(
            "structured_transform must use Converse; InvokeModel couples the "
            "model id to a provider-specific body shape"
        )


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


    def test_transform_uses_the_configured_model_and_normalizes_newlines_tabs(self):
        input_json = {"name": "John"}
        target_schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        fake_client = _FakeClient(json.dumps({"summary": "Line1\n\tLine2"}))

        transformed = transform_json_to_schema(input_json, target_schema, client=fake_client)

        self.assertEqual(fake_client.last_model_id, MODEL_ID)
        self.assertEqual(transformed, {"summary": "Line1  Line2"})

    def test_the_repair_model_is_not_the_legacy_one_bedrock_refuses(self):
        """This test used to pin the legacy id, which enforced the bug.

        Bedrock answers that model with ResourceNotFoundException -- "marked by
        provider as Legacy and you have not been actively using the model in
        the last 30 days" -- so every repair failed and a working pipeline
        returned its answer in a fallback envelope instead of the declared
        schema. The run still succeeded, which is how it survived a green
        deploy and a passing test that asserted the broken value.
        """
        self.assertNotEqual(MODEL_ID, "anthropic.claude-3-haiku-20240307-v1:0")
        self.assertNotIn("claude-3-haiku-20240307", MODEL_ID)

    def test_the_repair_model_can_be_changed_without_a_code_change(self):
        # It was a bare constant, so moving off a dead model needed a deploy of
        # new code rather than a parameter.
        import importlib
        import os

        from src.orchestrator import structured_transform as st

        os.environ["STRUCTURED_TRANSFORM_MODEL_ID"] = "us.amazon.nova-lite-v1:0"
        try:
            reloaded = importlib.reload(st)
            self.assertEqual(reloaded.MODEL_ID, "us.amazon.nova-lite-v1:0")
        finally:
            os.environ.pop("STRUCTURED_TRANSFORM_MODEL_ID", None)
            importlib.reload(st)

    def test_the_stack_supplies_the_repair_model(self):
        from pathlib import Path as _Path

        template = (_Path(__file__).resolve().parents[1] / "infra" / "template.yaml").read_text()
        self.assertIn("STRUCTURED_TRANSFORM_MODEL_ID:", template)

    def test_transform_normalizes_escaped_newline_tab_artifacts_in_json_text(self):
        input_json = {"name": "John"}
        target_schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        fake_client = _FakeClient("```json\n{\n\t\"summary\":\"Done\"\n}\n```")

        transformed = transform_json_to_schema(input_json, target_schema, client=fake_client)

        self.assertEqual(transformed, {"summary": "Done"})

    def test_transform_normalizes_escaped_smart_quotes_in_json_text(self):
        input_json = {"name": "John"}
        target_schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        fake_client = _FakeClient('{\”summary\”:\”Done\”}')

        transformed = transform_json_to_schema(input_json, target_schema, client=fake_client)

        self.assertEqual(transformed, {"summary": "Done"})

    def test_transform_extracts_json_from_text_wrapper(self):
        input_json = {"name": "John"}
        target_schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        fake_client = _FakeClient('Here is the transformed result: {"summary":"Done"}')

        transformed = transform_json_to_schema(input_json, target_schema, client=fake_client)

        self.assertEqual(transformed, {"summary": "Done"})

    def test_transform_extracts_nested_json_from_fallback_payload(self):
        input_json = {"name": "John"}
        target_schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        fake_client = _FakeClient(
            '{"status":"fallback_response","data":{"content":"{\\n  \\\"summary\\\": \\\"Done\\\"\\n}"},"_meta":{"coerced_from_non_json":true}}'
        )

        transformed = transform_json_to_schema(input_json, target_schema, client=fake_client)

        self.assertEqual(transformed, {"summary": "Done"})

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


class ConverseContractTests(unittest.TestCase):
    """The repair call must not be coupled to one provider's body shape."""

    def test_it_calls_converse_not_invoke_model(self):
        client = _FakeClient(json.dumps({"summary": "ok"}))
        transform_json_to_schema(
            {"a": 1}, {"type": "object", "properties": {"summary": {"type": "string"}}},
            client=client,
        )
        # _FakeClient.invoke_model raises; reaching here means Converse was used.
        self.assertIsNotNone(client.last_request)
        self.assertEqual(client.last_model_id, MODEL_ID)

    def test_the_token_limit_is_passed_through(self):
        client = _FakeClient(json.dumps({"summary": "ok"}))
        transform_json_to_schema(
            {"a": 1}, {"type": "object", "properties": {"summary": {"type": "string"}}},
            client=client, max_tokens=77,
        )
        self.assertEqual(client.last_request["inferenceConfig"]["maxTokens"], 77)

    def test_a_reply_with_no_content_does_not_raise(self):
        """An empty Converse response must degrade, not explode.

        The caller already handles a failed transform by keeping the agent's
        original answer; an IndexError here would turn a recoverable miss into
        a failed step.
        """
        from src.orchestrator.structured_transform import _text_from_converse

        self.assertEqual(_text_from_converse({}), "")
        self.assertEqual(_text_from_converse({"output": {}}), "")
        self.assertEqual(_text_from_converse({"output": {"message": {"content": []}}}), "")

    def test_multiple_content_blocks_are_joined(self):
        from src.orchestrator.structured_transform import _text_from_converse

        response = {"output": {"message": {"content": [{"text": '{"a":'}, {"text": " 1}"}]}}}
        self.assertEqual(_text_from_converse(response), '{"a": 1}')

    def test_no_provider_specific_body_remains_in_the_module(self):
        # anthropic_version in an InvokeModel body is the coupling this removed.
        from pathlib import Path as _Path

        source = (_Path(__file__).resolve().parents[1]
                  / "src" / "orchestrator" / "structured_transform.py").read_text()
        code = "\n".join(l for l in source.splitlines() if not l.strip().startswith("#"))
        self.assertNotIn("anthropic_version", code)
        self.assertNotIn("invoke_model", code)
