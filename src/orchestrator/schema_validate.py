from typing import Any, Dict

from jsonschema import Draft202012Validator, validate
from jsonschema.exceptions import ValidationError


HEALTH_INSIGHTS = "health_insights_v1"

# One repair pass. The previous schema required a list of questions, so a
# model still produces one; accepting that as schema-valid is how the team
# answered with questions after the records had already been read.
_REPROMPT = (
    "The previous answer asked questions or omitted insights and suggestions. "
    "Rewrite it as statements grounded in the records: each insight cites a "
    "source, date, and value; each suggestion is an action. data_gaps are "
    "observations, not questions. Do not ask the person anything."
)


def _is_question(text: str) -> bool:
    collapsed = " ".join(str(text or "").split())
    return bool(collapsed) and collapsed.endswith("?")


def _mostly_questions(items: list) -> bool:
    filled = [s for s in items if str(s).strip()]
    if not filled:
        return False
    return sum(_is_question(s) for s in filled) * 2 >= len(filled)


def output_is_mostly_questions(output: Dict[str, Any]) -> bool:
    """The substance of a health answer is questions.

    follow_ups may hold one or two optional items and is not counted: a
    single trailing question must not fail an answer whose insights and
    suggestions are statements. data_gaps are counted one by one, because
    that field is where a missing fact is stated, and a question there is
    the old answer coming back under another name.
    """
    if not isinstance(output, dict):
        return False
    insights = output.get("insights") if isinstance(output.get("insights"), list) else []
    suggestions = output.get("suggestions") if isinstance(output.get("suggestions"), list) else []
    findings: list = []
    for item in insights:
        if isinstance(item, dict):
            findings.append(str(item.get("title") or ""))
            findings.append(str(item.get("finding") or ""))
    actions: list = []
    for item in suggestions:
        if isinstance(item, dict):
            actions.append(str(item.get("action") or ""))
            actions.append(str(item.get("rationale") or ""))
    summary = [str(output.get("summary") or "")]
    if (
        _mostly_questions(findings)
        or _mostly_questions(actions)
        or _mostly_questions(summary + findings + actions)
    ):
        return True
    gaps = output.get("data_gaps") if isinstance(output.get("data_gaps"), list) else []
    return any(_is_question(str(gap)) for gap in gaps)


def validate_output(output: Dict[str, Any], schema: Dict[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)
    validate(instance=output, schema=schema)
    if schema.get("title") == HEALTH_INSIGHTS and output_is_mostly_questions(output):
        raise ValidationError(
            "health insights are mostly questions; write findings and actions, "
            "and state missing information as observations in data_gaps"
        )


def _health_insights_ok(output: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    try:
        validate_output(output, schema)
    except ValidationError:
        return False
    return True


def settle_health_insights(output: Dict[str, Any], schema: Dict[str, Any], transform) -> tuple:
    """Keep a valid health insights answer, or repair a question-shaped one once.

    Returns ``(output, accepted)``. The worker calls this instead of the
    generic transform for this schema, so an answer that already has insights
    and suggestions is stored as the agent wrote it. Anything else — a
    question list, or a body missing those fields — goes through ``transform``
    once. If that repair is still questions, it is not accepted.
    """
    if not isinstance(schema, dict) or schema.get("title") != HEALTH_INSIGHTS:
        return output, True
    if _health_insights_ok(output, schema):
        return output, True
    try:
        rewritten = transform(
            {"rejected_because": _REPROMPT, "previous": output}, schema)
    except Exception:
        return output, False
    if not isinstance(rewritten, dict):
        return output, False
    return rewritten, _health_insights_ok(rewritten, schema)


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
                if _looks_like_creative_brief_schema(schema):
                    inner = _normalize_creative_brief_output(inner)
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
