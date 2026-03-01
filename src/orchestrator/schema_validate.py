from typing import Any, Dict
from jsonschema import Draft202012Validator, validate
from jsonschema.exceptions import ValidationError

def validate_output(output: Dict[str, Any], schema: Dict[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)
    validate(instance=output, schema=schema)

def format_validation_error(e: ValidationError) -> str:
    path = ".".join([str(p) for p in e.path]) if e.path else ""
    return f"{e.message} at {path}".strip()
