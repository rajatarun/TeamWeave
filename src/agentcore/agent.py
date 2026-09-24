"""One agent turn: apply the instruction, call the model, return the text.

Deliberately free of the bedrock-agentcore SDK. `app.py` is the entrypoint the
runtime boots and it imports this; keeping the logic separate means the tests
exercise it without installing Starlette and uvicorn, and a packaging problem
shows up as an import error in `app.py` rather than as a silent behaviour
change here.

TeamWeave already does the surrounding work: prompt_builder composes the whole
per-turn prompt -- role, step goal, output contract, RAG context -- and the
worker validates the result against the step's JSON Schema afterwards.
Duplicating any of that here would give the two substrates different
behaviour, which is the one thing a migration must not do.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

DEFAULT_MODEL_ID = "us.amazon.nova-micro-v1:0"
DEFAULT_MAX_TOKENS = 4096
# What a Classic agent carried as its instruction. The per-agent part -- role,
# step goal, output contract -- is already composed into the prompt by
# prompt_builder, so this only has to hold the contract steady.
DEFAULT_INSTRUCTION = (
    "You are an agent in a TeamWeave pipeline. Follow the ROLE and STEP_GOAL "
    "given in the message. The REQUEST is the assignment, not the answer: do "
    "the planning, exploration, comparison, or writing the goal asks for. "
    "Restating or mirroring the user's words is a failed turn. Obey the "
    "OUTPUT CONTRACT exactly: return only valid JSON matching the step's "
    "schema, with no markdown and no commentary."
)

_client = None


def _bedrock():
    global _client
    if _client is None:
        import boto3

        _client = boto3.client("bedrock-runtime")
    return _client


def extract_prompt(payload: Any) -> str:
    """Pull the prompt out of an invocation payload.

    The SDK passes payloads to the entrypoint unchanged, so this accepts the
    shape TeamWeave sends and tolerates a bare string or a couple of common
    aliases rather than failing on a caller that is almost right.
    """
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="ignore")
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except ValueError:
                return text
        else:
            return text
    if isinstance(payload, dict):
        for key in ("prompt", "inputText", "input", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _text_from_converse(response: Dict[str, Any]) -> str:
    blocks = (((response or {}).get("output") or {}).get("message") or {}).get("content") or []
    return "".join(b.get("text", "") for b in blocks if isinstance(b, dict)).strip()


def run_turn(payload: Any, *, client=None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """One agent turn. Pure enough to test without the runtime or AWS."""
    env = env if env is not None else os.environ
    prompt = extract_prompt(payload)
    if not prompt:
        # A blank prompt is a caller error, and returning an empty answer would
        # surface downstream as a schema validation failure pointing at the
        # model instead of at the request.
        return {"error": "payload contained no prompt", "result": ""}

    # Per-turn instruction wins over the runtime's default: one runtime serves
    # every TeamWeave agent, so the identity has to arrive with the request
    # rather than be baked into the deployment.
    instruction = ""
    if isinstance(payload, dict):
        instruction = str(payload.get("instruction") or "").strip()
    if not instruction:
        instruction = (env.get("AGENT_INSTRUCTION") or "").strip()
    if not instruction:
        instruction = DEFAULT_INSTRUCTION
    # Per-turn model wins over the runtime's default, for the same reason the
    # instruction does: one generic runtime serves every agent of a team, so
    # anything that differs between agents has to arrive with the request.
    # AGENT_MODEL_ID is baked in at CreateAgentRuntime time, so a model read
    # only from there makes every agent's declared model_id in team.json
    # decorative -- the config says four models and the runtime serves one.
    model_id = ""
    if isinstance(payload, dict):
        model_id = str(payload.get("modelId") or payload.get("model_id") or "").strip()
    if not model_id:
        model_id = (env.get("AGENT_MODEL_ID") or "").strip()
    if not model_id:
        model_id = DEFAULT_MODEL_ID
    try:
        max_tokens = int(env.get("AGENT_MAX_TOKENS") or DEFAULT_MAX_TOKENS)
    except ValueError:
        max_tokens = DEFAULT_MAX_TOKENS

    request: Dict[str, Any] = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0},
    }
    if instruction:
        request["system"] = [{"text": instruction}]

    response = (client or _bedrock()).converse(**request)
    return {"result": _text_from_converse(response), "modelId": model_id}
