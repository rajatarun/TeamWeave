import json
import time
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .bedrock_wrappers import invoke_agent_request
from .logger import get_logger
from .models import StepFailed

log = get_logger("bedrock_invoke")
brt = boto3.client(
    "bedrock-agent-runtime",
    config=Config(
        read_timeout=1800,
        connect_timeout=60,
        retries={"max_attempts": 0},
    ),
)
def invoke_agent(agent_id: str, alias_id: str, session_id: str, input_text: str, max_retries: int = 2) -> str:
    if not agent_id or not alias_id:
        raise StepFailed("invoke_agent", "Missing agentId/aliasId in config")

    last_err: Optional[Exception] = None
    for attempt in range(0, max_retries + 1):
        try:
            log.info(
                "Invoking Bedrock agent",
                extra={
                    "agent_id": agent_id,
                    "alias_id": alias_id,
                    "session_id": session_id,
                    "attempt": attempt,
                    "input_text": input_text[:1000],
                },
            )
            resp = invoke_agent_request(
                brt,
                agent_id=agent_id,
                alias_id=alias_id,
                session_id=session_id,
                input_text=input_text,
            )

            guardrail_action = resp.get("amazon-bedrock-guardrailAction")
            guardrail_trace = resp.get("amazon-bedrock-trace")
            if guardrail_action == "INTERVENED" or guardrail_trace:
                log.info(
                    "Bedrock guardrail trace",
                    extra={
                        "amazon-bedrock-guardrailAction": guardrail_action,
                        "amazon-bedrock-trace": json.dumps(guardrail_trace, default=str)[:4000],
                    },
                )

            out_chunks = []
            stream = resp.get("completion")
            if stream is None:
                raise RuntimeError("InvokeAgent missing 'completion' stream")
            for event in stream:
                chunk = event.get("chunk")
                if chunk and chunk.get("bytes"):
                    out_chunks.append(chunk["bytes"].decode("utf-8", errors="ignore"))

                event_guardrail_action = event.get("amazon-bedrock-guardrailAction")
                event_guardrail_trace = event.get("amazon-bedrock-trace")
                if event_guardrail_action == "INTERVENED" or event_guardrail_trace:
                    log.info(
                        "Bedrock guardrail trace event",
                        extra={
                            "amazon-bedrock-guardrailAction": event_guardrail_action,
                            "amazon-bedrock-trace": json.dumps(event_guardrail_trace, default=str)[:4000],
                        },
                    )
            return "".join(out_chunks).strip()
        except Exception as e:
            if isinstance(e, ClientError):
                error_code = ((e.response or {}).get("Error") or {}).get("Code", "")
                if error_code in {"AccessDeniedException", "UnrecognizedClientException", "ExpiredTokenException"}:
                    message = (
                        "InvokeAgent permission/auth failure: "
                        f"{error_code}. Verify IAM permissions for bedrock:InvokeAgent and "
                        "that credentials are valid in this runtime."
                    )
                    log.error(message, extra={"agent_id": agent_id, "alias_id": alias_id, "session_id": session_id})
                    raise StepFailed("invoke_agent", message) from e
            last_err = e
            log.warning("InvokeAgent failed", extra={"attempt": attempt, "err": str(e)[:240]})
            time.sleep(1.3 * (attempt + 1))
    raise StepFailed("invoke_agent", f"InvokeAgent failed after retries: {last_err}")
