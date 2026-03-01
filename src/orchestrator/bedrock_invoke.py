import time
from typing import Optional
import boto3
from .logger import get_logger
from .models import StepFailed

log = get_logger("bedrock_invoke")
brt = boto3.client("bedrock-agent-runtime")

def invoke_agent(agent_id: str, alias_id: str, session_id: str, input_text: str, max_retries: int = 2) -> str:
    if not agent_id or not alias_id:
        raise StepFailed("invoke_agent", "Missing agentId/aliasId in config")

    last_err: Optional[Exception] = None
    for attempt in range(0, max_retries + 1):
        try:
            resp = brt.invoke_agent(
                agentId=agent_id,
                agentAliasId=alias_id,
                sessionId=session_id,
                inputText=input_text,
            )
            out_chunks = []
            stream = resp.get("completion")
            if stream is None:
                raise RuntimeError("InvokeAgent missing 'completion' stream")
            for event in stream:
                chunk = event.get("chunk")
                if chunk and chunk.get("bytes"):
                    out_chunks.append(chunk["bytes"].decode("utf-8", errors="ignore"))
            return "".join(out_chunks).strip()
        except Exception as e:
            last_err = e
            log.warning("InvokeAgent failed", extra={"attempt": attempt, "err": str(e)[:240]})
            time.sleep(1.3 * (attempt + 1))
    raise StepFailed("invoke_agent", f"InvokeAgent failed after retries: {last_err}")
