from typing import Optional


def invoke_agent_request(
    runtime_client,
    *,
    agent_id: str,
    alias_id: str,
    session_id: str,
    input_text: str,
    guardrail_id: Optional[str] = None,
    guardrail_version: Optional[str] = None,
) -> dict:
    """Wrapper around Bedrock Agent Runtime invoke_agent API."""
    kwargs = {
        "agentId": agent_id,
        "agentAliasId": alias_id,
        "sessionId": session_id,
        "inputText": input_text,
    }
    if guardrail_id:
        kwargs["guardrailConfiguration"] = {
            "guardrailId": guardrail_id,
            "guardrailVersion": guardrail_version or "DRAFT",
        }
    return runtime_client.invoke_agent(**kwargs)


def invoke_model_request(
    runtime_client,
    *,
    model_id: str,
    body: str,
    content_type: Optional[str] = None,
    accept: Optional[str] = None,
) -> dict:
    """Wrapper around Bedrock Runtime invoke_model API."""
    request = {
        "modelId": model_id,
        "body": body,
    }
    if content_type:
        request["contentType"] = content_type
    if accept:
        request["accept"] = accept

    return runtime_client.invoke_model(**request)
