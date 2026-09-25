#!/usr/bin/env python3
"""Which image models this account can actually call.

Two deploys were spent guessing model ids. One real Canvas id came back
marked Legacy because the account had no active access, and another Canvas
id came back as an identifier that does not exist. Those are different
problems with the same symptom, and neither is answerable from a model
catalogue held in someone's memory.

`ListFoundationModels` answers both from the account itself. It is on the
`bedrock` control-plane client (not `bedrock-runtime`), takes
`byOutputModality`, and returns `modelLifecycle.status` as ACTIVE or LEGACY
alongside `inferenceTypesSupported` -- so "exists", "is not retired" and "can
be called on demand" are three separate readable facts rather than one guess.

Read-only. Usable on its own, and called by `pipeline_smoke.py` when an image
step fails so the warning names real candidates instead of advice.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# The families bedrock_image knows how to build a request body for. A model
# this account can call is still unusable here if its provider takes a
# different body -- see bedrock_image.build_body.
KNOWN_FAMILIES = ("amazon.nova-canvas", "amazon.titan-image")


def knows_body_shape(model_id: str) -> bool:
    return any(model_id.startswith(f) for f in KNOWN_FAMILIES)


def list_image_models(client) -> List[Dict[str, Any]]:
    response = client.list_foundation_models(byOutputModality="IMAGE")
    out = []
    for summary in response.get("modelSummaries") or []:
        model_id = summary.get("modelId") or ""
        lifecycle = (summary.get("modelLifecycle") or {}).get("status") or "UNKNOWN"
        inference = summary.get("inferenceTypesSupported") or []
        out.append({
            "modelId": model_id,
            "provider": summary.get("providerName") or "",
            "lifecycle": lifecycle,
            "onDemand": "ON_DEMAND" in inference,
            "bodyKnown": knows_body_shape(model_id),
        })
    return sorted(out, key=lambda m: m["modelId"])


def usable(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Active, callable on demand, and a body shape we can build."""
    return [m for m in models if m["lifecycle"] == "ACTIVE" and m["onDemand"] and m["bodyKnown"]]


def summarise(models: List[Dict[str, Any]], limit: int = 6) -> str:
    """One line for a CI annotation."""
    good = usable(models)
    if good:
        ids = ", ".join(m["modelId"] for m in good[:limit])
        return f"image models this account can call: {ids}"
    callable_ids = [m["modelId"] for m in models
                    if m["lifecycle"] == "ACTIVE" and m["onDemand"]]
    if callable_ids:
        return (
            "no image model this account can call has a request body "
            f"bedrock_image knows how to build. Active on-demand models: "
            f"{', '.join(callable_ids[:limit])} -- adding one needs its body "
            "shape in bedrock_image.build_body, not just a new id"
        )
    if models:
        return (
            "this account lists image models but none are ACTIVE and on-demand: "
            + ", ".join(f"{m['modelId']} ({m['lifecycle']})" for m in models[:limit])
            + " -- grant model access in the Bedrock console"
        )
    return "this account lists no image-output models at all in this region"


def describe(region: str, client=None) -> str:
    """Best-effort one-liner. Never raises: this runs inside error reporting."""
    try:
        models = list_image_models(client or boto3.client("bedrock", region_name=region))
    except (ClientError, BotoCoreError, Exception) as exc:  # noqa: BLE001
        return f"could not list image models ({type(exc).__name__}: {str(exc)[:160]})"
    return summarise(models)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        models = list_image_models(boto3.client("bedrock", region_name=args.region))
    except (ClientError, BotoCoreError) as exc:
        print(f"could not list foundation models: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(models, indent=2))
        return 0

    if not models:
        print(f"No image-output models listed in {args.region}.")
        return 1

    width = max(len(m["modelId"]) for m in models)
    print(f"{'MODEL ID'.ljust(width)}  LIFECYCLE  ON-DEMAND  BODY KNOWN")
    for m in models:
        print(f"{m['modelId'].ljust(width)}  {m['lifecycle']:<9}  "
              f"{'yes' if m['onDemand'] else 'no':<9}  {'yes' if m['bodyKnown'] else 'no'}")
    print()
    print(summarise(models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
