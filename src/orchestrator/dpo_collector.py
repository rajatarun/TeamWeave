"""DPO training data collection for TeamWeave.

When ``DPO_TRAINING_BUCKET`` is set, each workflow step invokes the Bedrock
agent **twice** with independent session IDs.  The invocation with the lower
``composite_risk_score`` is returned as the pipeline output.  When the score
delta between the two invocations exceeds ``DPO_DELTA_THRESHOLD`` (default
0.4), a chosen/rejected training record is uploaded to S3 at:

    s3://{DPO_TRAINING_BUCKET}/{team}/{step_id}/{run_id}/dpo_{timestamp}.json

DPO collection is entirely opt-in (controlled by the ``DPO_TRAINING_BUCKET``
environment variable) and best-effort: upload failures are logged as warnings
and never propagate to the pipeline.  When the bucket is not configured the
module has zero overhead — no extra Bedrock calls, no S3 operations.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

import boto3

from .logger import get_logger

log = get_logger("dpo_collector")

_DEFAULT_DELTA_THRESHOLD = 0.4
_s3_client = None


def _get_s3() -> Any:
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def dpo_bucket() -> str:
    """Return the DPO_TRAINING_BUCKET env var, or an empty string if not set."""
    return os.environ.get("DPO_TRAINING_BUCKET", "")


def dpo_delta_threshold() -> float:
    """Return DPO_DELTA_THRESHOLD as a float, defaulting to 0.4."""
    try:
        return float(os.environ.get("DPO_DELTA_THRESHOLD", str(_DEFAULT_DELTA_THRESHOLD)))
    except (ValueError, TypeError):
        return _DEFAULT_DELTA_THRESHOLD


def dpo_project() -> str:
    """Return the project namespace for S3 key partitioning.

    Reads DPO_PROJECT (set to the CloudFormation stack name by the main template).
    Falls back to "default" so the key is always valid even in local runs.
    """
    return os.environ.get("DPO_PROJECT", "default")


def _upload_dpo_record(
    bucket: str,
    project: str,
    team: str,
    step_id: str,
    run_id: str,
    prompt: str,
    context: Dict[str, Any],
    chosen: str,
    rejected: str,
    chosen_score: float,
    rejected_score: float,
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
) -> None:
    """Upload a DPO training record to S3. Best-effort — swallows all exceptions."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")
    ts_safe = now.strftime("%Y%m%dT%H%M%S%f")

    key = f"{project}/{team}/{step_id}/{run_id}/dpo_{ts_safe}.json"
    record = {
        "schema_version": "dpo-v1",
        "timestamp": timestamp,
        "project": project,
        "team": team,
        "step_id": step_id,
        "run_id": run_id,
        "prompt": prompt,
        "context": context,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_composite_score": chosen_score,
        "rejected_composite_score": rejected_score,
        "delta": abs(rejected_score - chosen_score),
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
    }
    body = json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")
    try:
        _get_s3().put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        log.info(
            "dpo_record_uploaded",
            extra={
                "bucket": bucket,
                "key": key,
                "project": project,
                "team": team,
                "step_id": step_id,
                "run_id": run_id,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "delta": record["delta"],
            },
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "dpo_upload_failed",
            extra={"bucket": bucket, "key": key, "err": str(exc)[:400]},
        )


def collect_dpo_step(
    invoke_fn: Callable[[str], Tuple[str, dict]],
    *,
    team: str,
    step_id: str,
    run_id: str,
    prompt: str,
    context: Dict[str, Any],
    session_id_a: str,
    session_id_b: str,
) -> str:
    """Run dual invocation for a workflow step and collect a DPO training record.

    Parameters
    ----------
    invoke_fn:
        Callable that accepts a ``session_id`` (str) and returns
        ``(response_text, span_metrics_dict)``.  The caller binds agent_id
        and alias_id into this callable via a closure.
    team, step_id, run_id, prompt, context:
        Metadata written into the DPO training record.
    session_id_a, session_id_b:
        Session IDs for the two independent invocations (must differ so
        the agent treats them as separate conversations).

    Returns
    -------
    str
        The text response from the better invocation (lower
        ``composite_risk_score``).  When scores are equal or both None,
        response A is returned.

    Raises
    ------
    Any exception raised by ``invoke_fn`` for invocation A is re-raised so
    the pipeline is never silently degraded.  Exceptions from invocation B
    are caught, logged as a warning, and cause the function to fall back to
    response A without uploading a DPO record.
    """
    bucket = dpo_bucket()
    threshold = dpo_delta_threshold()

    # ── Invocation A (primary) ─────────────────────────────────────────────
    text_a, metrics_a = invoke_fn(session_id_a)
    score_a: Optional[float] = metrics_a.get("composite_risk_score")

    # ── Invocation B (secondary) ───────────────────────────────────────────
    try:
        text_b, metrics_b = invoke_fn(session_id_b)
        score_b: Optional[float] = metrics_b.get("composite_risk_score")
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "dpo_invocation_b_failed_using_a",
            extra={"step_id": step_id, "run_id": run_id, "err": str(exc)[:400]},
        )
        return text_a

    # ── Score ranking ──────────────────────────────────────────────────────
    if score_a is None and score_b is None:
        log.info(
            "dpo_skipped_no_scores",
            extra={"step_id": step_id, "run_id": run_id},
        )
        return text_a  # cannot rank; fall back to A

    # Treat None as infinity so a real score always wins
    eff_a = float(score_a) if score_a is not None else float("inf")
    eff_b = float(score_b) if score_b is not None else float("inf")

    if eff_a <= eff_b:
        better_text, worse_text = text_a, text_b
        chosen_score, rejected_score = eff_a, eff_b
        metrics_chosen, metrics_rejected = metrics_a, metrics_b
    else:
        better_text, worse_text = text_b, text_a
        chosen_score, rejected_score = eff_b, eff_a
        metrics_chosen, metrics_rejected = metrics_b, metrics_a

    delta = abs(rejected_score - chosen_score)

    log.info(
        "dpo_scores_compared",
        extra={
            "step_id": step_id,
            "run_id": run_id,
            "score_a": score_a,
            "score_b": score_b,
            "chosen_score": chosen_score,
            "delta": delta,
            "threshold": threshold,
            "will_upload": delta >= threshold and bool(bucket),
        },
    )

    # ── Upload if delta meets threshold ────────────────────────────────────
    if delta >= threshold and bucket:
        _upload_dpo_record(
            bucket=bucket,
            project=dpo_project(),
            team=team,
            step_id=step_id,
            run_id=run_id,
            prompt=prompt,
            context=context,
            chosen=better_text,
            rejected=worse_text,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
        )

    return better_text
