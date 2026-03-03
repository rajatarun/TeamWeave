from typing import Any, Dict

from jsonschema import Draft202012Validator, validate
from jsonschema.exceptions import ValidationError


def validate_output(output: Dict[str, Any], schema: Dict[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)
    validate(instance=output, schema=schema)


def validate_or_unwrap_output(output: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate output, unwrapping one-key envelopes when possible."""
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


def format_validation_error(e: ValidationError) -> str:
    path = ".".join([str(p) for p in e.path]) if e.path else ""
    return f"{e.message} at {path}".strip()
