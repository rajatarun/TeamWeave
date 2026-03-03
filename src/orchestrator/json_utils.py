import json
from typing import Any


def _normalize_json_text(raw_text: str) -> str:
    """Normalize model text before attempting JSON parse."""
    normalized = (raw_text or "").strip()
    if not normalized:
        return ""

    # Remove markdown fences and normalize common escape artifacts.
    normalized = normalized.replace("```json", "").replace("```JSON", "").replace("```", "")
    normalized = normalized.replace("\\n", " ").replace("\\t", " ")
    normalized = normalized.replace("\n", " ").replace("\t", " ")
    normalized = normalized.replace('\\"', '"').replace("\\”", '"').replace("\\“", '"')
    normalized = normalized.replace("”", '"').replace("“", '"')
    return normalized.strip()


def _decode_nested_json_string(value: Any, depth: int = 0) -> Any:
    """Decode double-encoded JSON strings (common LLM artifact)."""
    if depth >= 3:
        return value

    if isinstance(value, dict):
        return {k: _decode_nested_json_string(v, depth) for k, v in value.items()}

    if isinstance(value, list):
        return [_decode_nested_json_string(item, depth) for item in value]

    if not isinstance(value, str):
        return value

    candidate = _normalize_json_text(value)
    if not candidate:
        return value

    if (candidate.startswith("{") and candidate.endswith("}")) or (
        candidate.startswith("[") and candidate.endswith("]")
    ):
        try:
            return _decode_nested_json_string(json.loads(candidate), depth + 1)
        except Exception:
            pass

    if (candidate.startswith('"') and candidate.endswith('"')) or (candidate.startswith("'") and candidate.endswith("'")):
        try:
            return _decode_nested_json_string(json.loads(candidate), depth + 1)
        except Exception:
            pass

    return value


def _loads_with_normalization(candidate: str) -> Any:
    """Try parsing candidate text as JSON with normalization and unescape fallbacks."""
    last_error: Exception | None = None

    try:
        return _decode_nested_json_string(json.loads(candidate))
    except Exception as exc:
        last_error = exc

    normalized = _normalize_json_text(candidate)
    try:
        return _decode_nested_json_string(json.loads(normalized))
    except Exception as exc:
        last_error = exc

    if '\\"' in normalized or "\\”" in normalized or "\\“" in normalized:
        unescaped = normalized.replace('\\"', '"').replace("\\”", '"').replace("\\“", '"')
        try:
            return _decode_nested_json_string(json.loads(unescaped))
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError("unable to parse candidate")


def extract_json_payload(raw_text: str) -> Any:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise ValueError("empty response")

    try:
        return _loads_with_normalization(raw_text)
    except Exception:
        pass

    if "```" in raw_text:
        parts = raw_text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate:
                continue
            try:
                return _loads_with_normalization(candidate)
            except Exception:
                continue

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw_text):
        if ch not in "[{":
            continue
        try:
            candidate = raw_text[idx:]
            try:
                parsed, end = decoder.raw_decode(candidate)
            except Exception:
                normalized_candidate = _normalize_json_text(candidate)
                parsed, end = decoder.raw_decode(normalized_candidate)
            if isinstance(parsed, (dict, list)):
                return _decode_nested_json_string(parsed)
        except Exception:
            continue

    raise ValueError("unable to locate valid JSON payload in model response")


def build_standard_response(raw_text: str, reason: str) -> Any:
    content = (raw_text or "").strip()
    return {
        "status": "fallback_response",
        "message": "Model response was not valid JSON. Returning standardized payload.",
        "data": {
            "content": content,
        },
        "_meta": {
            "coerced_from_non_json": True,
            "reason": reason,
        },
    }
