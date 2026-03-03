from typing import Any, Dict

from jsonschema import Draft202012Validator, validate
from jsonschema.exceptions import ValidationError


def validate_output(output: Dict[str, Any], schema: Dict[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)
    validate(instance=output, schema=schema)


def validate_or_unwrap_output(output: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate output, unwrapping one-key envelopes when possible."""
    if _looks_like_creative_brief_schema(schema):
        output = _normalize_creative_brief_output(output)

    try:
        validate_output(output, schema)
        return output
    except ValidationError:
        if isinstance(output, dict) and len(output) == 1:
            inner = next(iter(output.values()))
            if isinstance(inner, dict):
                validate_output(inner, schema)
                return inner
        raise


def _looks_like_creative_brief_schema(schema: Dict[str, Any]) -> bool:
    return schema.get("title") == "CreativeBriefV1"


def _normalize_creative_brief_output(output: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(output, dict):
        return output
    if "goal" in output or "objective" not in output:
        return output

    normalized = dict(output)
    normalized["goal"] = normalized["objective"]
    return normalized


def format_validation_error(e: ValidationError) -> str:
    path = ".".join([str(p) for p in e.path]) if e.path else ""
    return f"{e.message} at {path}".strip()
