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
from .model_map import (
    emit_model_event,
    estimate_cost,
    is_model_unavailable,
    resolve_model,
)
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
    team: str = "",
    model_id: str = "",
    model_category: str = "",
) -> Tuple[str, dict]:
    """The retry loop both public functions share.

    The model comes from ``resolve_model``. ``model_id`` is an override and
    is logged there. A throttle or an unavailable model advances the chain.
    """
    runtime = get_runtime()
    choice = resolve_model(model_category or "default", override=model_id)
    # Both substrates' coordinates travel together. Building a Classic-only
    # ref here is what made AGENT_RUNTIME=agentcore unreachable: every call
    # arrived without a runtimeArn and was rejected before it was attempted.
    # The model id is filled per attempt so a fallback is a new turn, not a
    # retry of the model that just refused.
    base = dict(
        agent_id=agent_id,
        alias_id=alias_id,
        runtime_arn=runtime_arn,
        qualifier=qualifier,
        team=team,
        max_tokens=choice.max_tokens,
        temperature=choice.temperature,
    )
    probe = AgentRef(**base, model_id=choice.model_id)
    problem = runtime.missing_fields(probe)
    if problem:
        raise StepFailed(op, problem)

    last_err: Optional[Exception] = None
    for index, spec in enumerate(choice.chain):
        ref = AgentRef(**base, model_id=spec.model_id)
        switched = False
        for attempt in range(0, max_retries + 1):
            started = time.perf_counter()
            try:
                log.info(
                    "Invoking agent",
                    extra={
                        "runtime": runtime.name,
                        "agent_id": agent_id,
                        "alias_id": alias_id,
                        "session_id": session_id,
                        "attempt": attempt,
                        "model_id": spec.model_id,
                        "model_category": choice.category,
                        "input_text": input_text[:1000],
                    },
                )
                text, span = _call_model(
                    runtime, ref, spec.provider, session_id, input_text, shadow_alias_id,
                )
                _emit_turn(
                    choice, spec, index, session_id, agent_id, team, span,
                    started, success=True, quality="",
                )
                return text, span
            except StepFailed as e:
                if reraise_step_failed:
                    raise
                last_err = e
                log.warning("Invoke failed", extra={"attempt": attempt, "err": str(e)[:240]})
                time.sleep(1.3 * (attempt + 1))
                continue
            except Exception as e:
                _log_transport_failure(e, agent_id, alias_id, attempt)
                message = _auth_failure_message(e)
                if message and not is_model_unavailable(e):
                    log.error(message, extra={"agent_id": agent_id, "alias_id": alias_id, "session_id": session_id})
                    raise StepFailed(op, message) from e
                if is_model_unavailable(e) and index + 1 < len(choice.chain):
                    _emit_turn(
                        choice, spec, index, session_id, agent_id, team, {},
                        started, success=False, quality="", error=str(e)[:200],
                    )
                    log.warning(
                        "model_fallback",
                        extra={
                            "category": choice.category,
                            "model_id": spec.model_id,
                            "next": choice.chain[index + 1].model_id,
                            "err": str(e)[:200],
                        },
                    )
                    last_err = e
                    switched = True
                    break
                last_err = e
                log.warning("Invoke failed", extra={"attempt": attempt, "err": str(e)[:240]})
                time.sleep(1.3 * (attempt + 1))
        if not switched:
            break

    raise StepFailed(op, f"InvokeAgent failed after retries: {last_err}")


def _call_model(runtime, ref, provider, session_id, input_text, shadow_alias_id):
    if provider == "gemini":
        from .gemini import generate_text
        result = generate_text(
            input_text,
            model=ref.model_id,
            max_tokens=ref.max_tokens or 2048,
            temperature=ref.temperature if ref.temperature is not None else 0,
        )
        return result.get("text") or "", {
            "prompt_tokens": result.get("prompt_tokens") or 0,
            "completion_tokens": result.get("completion_tokens") or 0,
        }
    return runtime.invoke(
        ref,
        session_id=session_id,
        input_text=input_text,
        shadow_alias_id=shadow_alias_id,
    )


def _emit_turn(choice, spec, index, session_id, agent_id, team, span, started,
               *, success, quality, error=""):
    prompt_tokens = int((span or {}).get("prompt_tokens") or 0)
    completion_tokens = int((span or {}).get("completion_tokens") or 0)
    emit_model_event({
        "agent": agent_id,
        "team": team,
        "run_id": session_id,
        "category": choice.category,
        "model_id": spec.model_id,
        "fallback_used": index > 0,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": estimate_cost(spec.model_id, prompt_tokens, completion_tokens),
        "success": success,
        "error": error,
        "result_quality": quality,
        "cost_tier": spec.cost_tier,
        "record_kind": "invocation",
    })


def invoke_agent(
    agent_id: str,
    alias_id: str,
    session_id: str,
    input_text: str,
    max_retries: int = 2,
    shadow_alias_id: Optional[str] = None,
    runtime_arn: str = "",
    qualifier: str = "",
    team: str = "",
    model_id: str = "",
    model_category: str = "",
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
        team=team,
        model_id=model_id,
        model_category=model_category,
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
    team: str = "",
    model_id: str = "",
    model_category: str = "",
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
        team=team,
        model_id=model_id,
        model_category=model_category,
    )
