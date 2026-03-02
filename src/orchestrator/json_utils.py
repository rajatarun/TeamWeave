import json
from typing import Any


def extract_json_payload(raw_text: str) -> Any:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise ValueError("empty response")

    try:
        return json.loads(raw_text)
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
                return json.loads(candidate)
            except Exception:
                continue

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw_text):
        if ch not in "[{":
            continue
        try:
            parsed, end = decoder.raw_decode(raw_text[idx:])
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            continue

    raise ValueError("unable to locate valid JSON payload in model response")
