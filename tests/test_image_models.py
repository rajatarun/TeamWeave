"""Reading the account's real image-model catalogue instead of guessing.

Two deploys were spent on guessed ids. `amazon.nova-canvas-v1:0` answered
*"marked by provider as Legacy and you have not been actively using the model
in the last 30 days"* -- a real id with no active access -- and
`amazon.nova-canvas-v2:0` answered *"The provided model identifier is
invalid"* -- an id that does not exist. Same symptom, different problems, and
neither is answerable from recollection.

`ListFoundationModels` is on the `bedrock` control-plane client and reports
`modelLifecycle.status` and `inferenceTypesSupported`, so "exists", "is not
retired" and "can be called on demand" stay three separate facts. The shapes
below come from botocore's own service model.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("image_models", REPO / "scripts" / "image_models.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def summary(model_id, lifecycle="ACTIVE", on_demand=True, provider="Amazon"):
    return {
        "modelId": model_id,
        "providerName": provider,
        "outputModalities": ["IMAGE"],
        "inferenceTypesSupported": ["ON_DEMAND"] if on_demand else ["PROVISIONED"],
        "modelLifecycle": {"status": lifecycle},
    }


class FakeBedrock:
    def __init__(self, models, capture=None):
        self.models = models
        self.capture = capture if capture is not None else {}

    def list_foundation_models(self, **kw):
        self.capture.update(kw)
        return {"modelSummaries": self.models}


def test_it_asks_only_for_image_models():
    capture = {}
    mod.list_image_models(FakeBedrock([], capture))
    assert capture == {"byOutputModality": "IMAGE"}


def test_lifecycle_and_on_demand_are_reported_separately():
    """The two failures had different causes; collapsing them loses that."""
    models = mod.list_image_models(FakeBedrock([
        summary("amazon.nova-canvas-v1:0", lifecycle="LEGACY"),
        summary("amazon.titan-image-generator-v2:0"),
        summary("amazon.provisioned-only-v1:0", on_demand=False),
    ]))
    by_id = {m["modelId"]: m for m in models}
    assert by_id["amazon.nova-canvas-v1:0"]["lifecycle"] == "LEGACY"
    assert by_id["amazon.titan-image-generator-v2:0"]["lifecycle"] == "ACTIVE"
    assert by_id["amazon.provisioned-only-v1:0"]["onDemand"] is False


def test_a_model_we_cannot_build_a_body_for_is_flagged():
    """Callable is not the same as usable: the InvokeModel body is
    provider-defined and bedrock_image only knows two families."""
    models = mod.list_image_models(FakeBedrock([
        summary("stability.sd3-large-v1:0", provider="Stability AI"),
        summary("amazon.nova-canvas-v1:0"),
    ]))
    by_id = {m["modelId"]: m for m in models}
    assert by_id["stability.sd3-large-v1:0"]["bodyKnown"] is False
    assert by_id["amazon.nova-canvas-v1:0"]["bodyKnown"] is True


def test_usable_needs_all_three():
    models = mod.list_image_models(FakeBedrock([
        summary("amazon.nova-canvas-v1:0", lifecycle="LEGACY"),      # retired
        summary("amazon.titan-image-generator-v1", on_demand=False),  # not on demand
        summary("stability.sd3-large-v1:0"),                          # unknown body
        summary("amazon.titan-image-generator-v2:0"),                 # usable
    ]))
    assert [m["modelId"] for m in mod.usable(models)] == ["amazon.titan-image-generator-v2:0"]


# ── the one-line summary a CI annotation carries ────────────────────────────

def test_the_summary_names_usable_models():
    models = mod.list_image_models(FakeBedrock([summary("amazon.nova-canvas-v1:0")]))
    assert "amazon.nova-canvas-v1:0" in mod.summarise(models)


def test_a_callable_model_with_an_unknown_body_says_so():
    """Otherwise the next person swaps the id and hits Malformed input."""
    models = mod.list_image_models(FakeBedrock([summary("stability.sd3-large-v1:0")]))
    text = mod.summarise(models)
    assert "build_body" in text
    assert "stability.sd3-large-v1:0" in text


def test_only_retired_models_points_at_the_console():
    models = mod.list_image_models(FakeBedrock([
        summary("amazon.nova-canvas-v1:0", lifecycle="LEGACY"),
    ]))
    text = mod.summarise(models)
    assert "LEGACY" in text
    assert "Bedrock console" in text


def test_no_image_models_at_all_is_its_own_message():
    assert "no image-output models" in mod.summarise([])


# ── describe() runs inside error reporting ──────────────────────────────────

def test_describe_never_raises():
    """It is called while reporting another failure. Raising there would
    replace a useful warning with a traceback about the warning."""
    class Boom:
        def list_foundation_models(self, **kw):
            raise RuntimeError("no credentials")

    text = mod.describe("us-east-1", client=Boom())
    assert "could not list image models" in text
    assert "RuntimeError" in text


def test_describe_returns_the_summary_when_it_works():
    client = FakeBedrock([summary("amazon.nova-canvas-v1:0")])
    assert "amazon.nova-canvas-v1:0" in mod.describe("us-east-1", client=client)
