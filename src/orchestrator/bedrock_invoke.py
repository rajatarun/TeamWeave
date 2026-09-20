"""Run one agent turn, with retries, against whichever substrate is selected.

The two public functions are unchanged in name, signature and behaviour.
What moved out is the part that knows it is talking to Bedrock Agents
Classic -- that now lives behind ``agent_runtime.AgentRuntime``, so the
substrate can be swapped without touching the retry policy, the Observatory
gate, or the StepFailed contract that ``worker_handler`` depends on.
"""
import os
import time
from typing import Optional, Tuple

from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError

from .agent_runtime import AgentRef, brt, get_runtime  # noqa: F401  (brt re-exported)
from .logger import get_logger
from .models import StepFailed

log = get_logger("bedrock_invoke")

_AUTH_ERROR_CODES = {"AccessDeniedException", "UnrecognizedClientException", "ExpiredTokenException"}


def _log_transport_failure(e: Exception, agent_id: str, alias_id: str, attempt: int) -> None:
    if isinstance(e, ConnectTimeoutError):
        log.error(
            "InvokeAgent connect timeout — possible VPC endpoint routing issue",
            extra={
                "agent_id": agent_id,
                "alias_id": alias_id,
                "endpoint_url": os.environ.get("AWS_ENDPOINT_URL_BEDROCK_AGENT_RUNTIME", "<sdk-default>"),
                "err": str(e)[:400],
            },
        )
    elif isinstance(e, ReadTimeoutError):
        log.error(
            "InvokeAgent read timeout",
            extra={"agent_id": agent_id, "alias_id": alias_id, "attempt": attempt, "err": str(e)[:400]},
        )


def _auth_failure_message(e: Exception) -> str:
    """The message for an auth/permission ClientError, or "" if it is not one."""
    if not isinstance(e, ClientError):
        return ""
    error_code = ((e.response or {}).get("Error") or {}).get("Code", "")
    if error_code not in _AUTH_ERROR_CODES:
        return ""
    return (
        f"InvokeAgent permission/auth failure: {error_code}. "
        "Verify IAM permissions for bedrock:InvokeAgent and "
        "that credentials are valid in this runtime."
    )


def _invoke(
    op: str,
    agent_id: str,
    alias_id: str,
    session_id: str,
    input_text: str,
    max_retries: int,
    shadow_alias_id: Optional[str],
    reraise_step_failed: bool,
    runtime_arn: str = "",
    qualifier: str = "",
) -> Tuple[str, dict]:
    """The retry loop both public functions share."""
    runtime = get_runtime()
    # Both substrates' coordinates travel together. Building a Classic-only
    # ref here is what made AGENT_RUNTIME=agentcore unreachable: every call
    # arrived without a runtimeArn and was rejected before it was attempted.
    ref = AgentRef(
        agent_id=agent_id,
        alias_id=alias_id,
        runtime_arn=runtime_arn,
        qualifier=qualifier,
    )

    problem = runtime.missing_fields(ref)
    if problem:
        raise StepFailed(op, problem)

    last_err: Optional[Exception] = None
    for attempt in range(0, max_retries + 1):
        try:
            log.info(
                "Invoking agent",
                extra={
                    "runtime": runtime.name,
                    "agent_id": agent_id,
                    "alias_id": alias_id,
                    "session_id": session_id,
                    "attempt": attempt,
                    "input_text": input_text[:1000],
                },
            )
            return runtime.invoke(
                ref,
                session_id=session_id,
                input_text=input_text,
                shadow_alias_id=shadow_alias_id,
            )
        except StepFailed as e:
            # Only invoke_agent_with_metrics did this before the two loops were
            # merged: invoke_agent swallowed a StepFailed and retried it. That
            # difference looks accidental rather than intended, but changing it
            # is a behaviour change, so the caller still chooses.
            if reraise_step_failed:
                raise
            # Keep the cause: the final message reports it, and dropping it
            # turned the give-up line into "failed after retries: None".
            last_err = e
            log.warning("Invoke failed", extra={"attempt": attempt, "err": str(e)[:240]})
            time.sleep(1.3 * (attempt + 1))
            continue
        except Exception as e:
            _log_transport_failure(e, agent_id, alias_id, attempt)
            message = _auth_failure_message(e)
            if message:
                log.error(message, extra={"agent_id": agent_id, "alias_id": alias_id, "session_id": session_id})
                raise StepFailed(op, message) from e
            last_err = e
            log.warning("Invoke failed", extra={"attempt": attempt, "err": str(e)[:240]})
            time.sleep(1.3 * (attempt + 1))

    raise StepFailed(op, f"InvokeAgent failed after retries: {last_err}")


def invoke_agent(
    agent_id: str,
    alias_id: str,
    session_id: str,
    input_text: str,
    max_retries: int = 2,
    shadow_alias_id: Optional[str] = None,
    runtime_arn: str = "",
    qualifier: str = "",
) -> str:
    text, _ = _invoke(
        "invoke_agent",
        agent_id,
        alias_id,
        session_id,
        input_text,
        max_retries,
        shadow_alias_id,
        reraise_step_failed=False,
        runtime_arn=runtime_arn,
        qualifier=qualifier,
    )
    return text


def invoke_agent_with_metrics(
    agent_id: str,
    alias_id: str,
    session_id: str,
    input_text: str,
    max_retries: int = 2,
    shadow_alias_id: Optional[str] = None,
    runtime_arn: str = "",
    qualifier: str = "",
) -> tuple:
    """Invoke an agent and return (response_text, span_metrics_dict).

    Identical retry behaviour to invoke_agent but surfaces the mcp-observatory
    span metrics (including composite_risk_score) so callers can rank responses
    for DPO training data collection.
    """
    return _invoke(
        "invoke_agent_with_metrics",
        agent_id,
        alias_id,
        session_id,
        input_text,
        max_retries,
        shadow_alias_id,
        reraise_step_failed=True,
        runtime_arn=runtime_arn,
        qualifier=qualifier,
    )
