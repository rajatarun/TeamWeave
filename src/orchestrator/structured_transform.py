import json
from typing import Any, Dict, Optional

from .bedrock_wrappers import invoke_model_request
from .json_utils import extract_json_payload


MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


def transform_json_to_schema(
    input_json: Dict[str, Any],
    target_schema: Dict[str, Any],
    *,
    region_name: str = "us-east-1",
    client: Optional[Any] = None,
    max_tokens: int = 1024,
    model_id: str = MODEL_ID,
) -> Dict[str, Any]:
    """Use Bedrock Claude to transform an input object into the requested schema shape."""
    runtime = client
    if runtime is None:
        import boto3

        runtime = boto3.client("bedrock-runtime", region_name=region_name)

    normalized_target_schema = normalize_target_schema(target_schema)
    prompt = f"""Transform the input JSON into the target schema.
Map fields as best as you can.
Return ONLY valid JSON with exactly the same keys/shape as Target Schema.

Input JSON:
{json.dumps(input_json, indent=2)}

Target Schema:
{json.dumps(normalized_target_schema, indent=2)}
"""

    response = invoke_model_request(
        runtime,
        model_id=model_id,
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ),
    )

    result = json.loads(response["body"].read())
    text_payload = result["content"][0]["text"]
    transformed = _extract_transform_payload(text_payload)
    transformed = _normalize_json_string_values(transformed)
    return _coerce_to_template(transformed, normalized_target_schema)


def _extract_transform_payload(text_payload: str) -> Any:
    """Parse model output into JSON, including nested fallback payload content."""
    parsed = extract_json_payload(text_payload)

    if isinstance(parsed, dict) and parsed.get("status") == "fallback_response":
        nested = (parsed.get("data") or {}).get("content")
        if isinstance(nested, (dict, list)):
            parsed = nested
        elif isinstance(nested, str) and nested.strip():
            try:
                parsed = extract_json_payload(nested)
            except Exception:
                # Keep the original parsed payload for downstream coercion.
                pass

    return parsed


def _normalize_json_text(raw_json: str) -> str:
    """Normalize raw JSON text before parsing."""
    normalized = (raw_json or "").strip()
    normalized = normalized.replace("```json", "").replace("```JSON", "").replace("```", "")
    normalized = normalized.replace("\\n", " ").replace("\\t", " ")
    normalized = normalized.replace("\n", " ").replace("\t", " ")
    normalized = normalized.replace('\\"', '"')
    normalized = normalized.replace("\\”", '"').replace("\\“", '"')
    normalized = normalized.replace("”", '"').replace("“", '"')
    return normalized


def _normalize_json_string_values(value: Any) -> Any:
    """Normalize string values by removing newline/tab spacing artifacts."""
    if isinstance(value, dict):
        return {key: _normalize_json_string_values(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_normalize_json_string_values(item) for item in value]

    if isinstance(value, str):
        return value.replace("\n", " ").replace("\t", " ").strip()

    return value


def normalize_target_schema(target_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Accept either template schema or JSON Schema and return a template schema."""
    if not isinstance(target_schema, dict):
        return target_schema

    # Already in template format (example: {"name": "string"})
    if "type" not in target_schema and "properties" not in target_schema:
        return target_schema

    return _json_schema_to_template(target_schema)


def _json_schema_to_template(schema: Dict[str, Any]) -> Any:
    schema_type = schema.get("type")

    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties") or {}
        return {key: _json_schema_to_template(prop_schema) for key, prop_schema in properties.items()}

    if schema_type == "array":
        items = schema.get("items") or {}
        return [_json_schema_to_template(items)]

    if isinstance(schema_type, list):
        if "object" in schema_type:
            return _json_schema_to_template({"type": "object", "properties": schema.get("properties", {})})
        if "array" in schema_type:
            return _json_schema_to_template({"type": "array", "items": schema.get("items", {})})
        if "string" in schema_type:
            return "string"
        if "integer" in schema_type:
            return "integer"
        if "number" in schema_type:
            return "number"
        if "boolean" in schema_type:
            return "boolean"

    if schema_type == "string":
        return "string"
    if schema_type == "integer":
        return "integer"
    if schema_type == "number":
        return "number"
    if schema_type == "boolean":
        return "boolean"

    return "string"


def _coerce_to_template(value: Any, template: Any) -> Any:
    """Best-effort conversion of model output into the exact template shape."""
    if isinstance(template, dict):
        source = value if isinstance(value, dict) else {}
        return {key: _coerce_to_template(source.get(key), nested_template) for key, nested_template in template.items()}

    if isinstance(template, list):
        item_template = template[0] if template else None
        source_items = value if isinstance(value, list) else []
        if item_template is None:
            return source_items
        return [_coerce_to_template(item, item_template) for item in source_items]

    if template == "string":
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)

    if template == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0

    if template == "integer":
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    if template == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    return value
