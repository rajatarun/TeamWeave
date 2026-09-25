"""The model map is the only place a model id is chosen.

Prices, cost tier, and latency live on the model records. A category names
a primary. An unknown category resolves to default. A throttle advances the
fallback list. Estimated cost is tokens times those prices, and the
observatory write is allowed to fail.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_REGION", "us-east-1")

import pytest
from botocore.exceptions import ClientError

from src.orchestrator import bedrock_invoke
from src.orchestrator.model_map import (
    estimate_cost,
    is_model_unavailable,
    load_model_map,
    reset_cache,
    resolve_model,
    validate_model_map,
)

REPO = Path(__file__).resolve().parents[1]

# A family prefix bedrock_image uses to recognise a body shape. It is not a
# model id, and it is not in the map.
_FAMILY_PREFIXES = frozenset({"amazon.nova-canvas", "amazon.titan-image"})

_TOKEN = re.compile(
    r"(?:(?:us|eu|global|apac|jp|au)\.)?"
    r"(?:amazon|anthropic|meta|mistral|deepseek|cohere)\.[A-Za-z0-9][A-Za-z0-9._:-]*"
    r"|gemini-[0-9][A-Za-z0-9._-]*"
)

_SCAN_ROOTS = (
    REPO / "src",
    REPO / "config",
    REPO / "scripts",
    REPO / "infra",
    REPO / ".github",
    REPO / "docs" / "model_map.md",
)
_SKIP_FILES = {REPO / "config" / "model_map.yaml"}
_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".sh"}


@pytest.fixture(autouse=True)
def _fresh_map():
    reset_cache()
    yield
    reset_cache()


def test_the_shipped_map_loads():
    doc = load_model_map()
    assert validate_model_map(doc) == []
    assert "default" in doc["categories"]
    for name, cat in doc["categories"].items():
        primary = doc["models"][cat["primary"]]
        assert cat["cost_tier"] == primary["cost_tier"], name
        assert primary["price_checked"] == "2026-09-25"
        assert primary["price_source"].startswith("http")


def test_an_unknown_category_resolves_to_default(caplog):
    logging.getLogger("model_map").propagate = True
    caplog.set_level(logging.WARNING, logger="model_map")
    choice = resolve_model("not_a_category")
    assert choice.used_default is True
    assert choice.category == "default"
    assert choice.model_id == "deepseek.v3.2"
    assert choice.cost_tier == "low"
    assert any(r.message == "unknown_model_category" for r in caplog.records)


def test_an_override_becomes_the_primary_and_is_logged(caplog):
    logging.getLogger("model_map").propagate = True
    caplog.set_level(logging.WARNING, logger="model_map")
    choice = resolve_model(
        "writing_creative",
        override="us.anthropic.claude-sonnet-4-6",
    )
    assert choice.override is True
    assert choice.model_id == "us.anthropic.claude-sonnet-4-6"
    assert choice.cost_tier == "standard"
    assert [spec.model_id for spec in choice.fallbacks] == [
        "deepseek.v3.2",
    ]
    assert any(r.message == "model_id_override" for r in caplog.records)


_SONNET_46 = "us.anthropic.claude-sonnet-4-6"
_UNREACHABLE = (
    "us.anthropic.claude-sonnet-5",
    "us.anthropic.claude-opus-5",
)


def test_quality_categories_pay_for_sonnet_and_the_rest_stay_cheap():
    doc = load_model_map()
    expensive = {"coding", "planning", "finance", "health_medical"}
    reachable = [
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "deepseek.v3.2",
        "gemini-3.8-flash",
    ]
    for name in expensive:
        assert doc["categories"][name]["primary"] == _SONNET_46
        assert doc["categories"][name]["cost_tier"] == "standard"
        assert doc["categories"][name]["fallbacks"] == reachable
    for name, cat in doc["categories"].items():
        if name in expensive:
            continue
        assert cat["cost_tier"] in {"low", "infra"}, name
        assert cat["primary"] != _SONNET_46, name


def test_unreachable_models_never_resolve(caplog):
    logging.getLogger("model_map").propagate = True
    caplog.set_level(logging.WARNING, logger="model_map")
    doc = load_model_map()
    assert set(_UNREACHABLE) <= set(doc["unavailable"])
    for bad in _UNREACHABLE:
        assert bad not in doc["models"]
    for name in doc["categories"]:
        choice = resolve_model(name)
        ids = [spec.model_id for spec in choice.chain]
        for bad in _UNREACHABLE:
            assert bad not in ids, (name, bad)
        refused = resolve_model(name, override=_UNREACHABLE[0])
        assert refused.model_id == choice.model_id
        assert refused.override is False
        assert _UNREACHABLE[0] not in [spec.model_id for spec in refused.chain]
    assert any(r.message == "unavailable_model_refused" for r in caplog.records)


def test_estimate_cost_is_tokens_times_the_map_price():
    # $3.30 input + $16.50 output: base $3/$15 plus the 10% US geo premium.
    assert estimate_cost(_SONNET_46, 1_000_000, 1_000_000) == 19.8
    assert estimate_cost("deepseek.v3.2", 1_000_000, 1_000_000) == 2.47
    assert estimate_cost("gemini-3.1-flash-lite", 2_000_000, 0) == 0.5
    assert estimate_cost("gemini-3.1-flash-lite-image", images=2) == 0.0672
    assert estimate_cost("no-such-model", 1_000_000, 1_000_000) == 0.0


def test_a_malformed_request_is_not_a_reason_to_switch_models():
    exc = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Malformed input request"}},
        "Converse",
    )
    assert is_model_unavailable(exc) is False


def test_a_throttle_is_a_reason_to_switch_models():
    exc = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "Converse",
    )
    assert is_model_unavailable(exc) is True


def test_a_throttle_advances_to_the_fallback(monkeypatch):
    class Boom(Exception):
        pass

    class Runtime:
        name = "stub"

        def __init__(self):
            self.calls = []

        def missing_fields(self, ref):
            return ""

        def invoke(self, ref, *, session_id, input_text, shadow_alias_id=None):
            self.calls.append(ref.model_id)
            if len(self.calls) == 1:
                raise Boom("ThrottlingException: slow down")
            return "ok", {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}

    runtime = Runtime()
    seen = []
    monkeypatch.setattr(bedrock_invoke, "get_runtime", lambda: runtime)
    monkeypatch.setattr(bedrock_invoke, "emit_model_event", seen.append)
    monkeypatch.setattr(bedrock_invoke.time, "sleep", lambda *_a, **_k: None)

    text = bedrock_invoke.invoke_agent(
        "a", "b", "run-1", "hello", max_retries=0, model_category="default",
    )
    assert text == "ok"
    assert runtime.calls == [
        "deepseek.v3.2",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ]
    invocation = next(row for row in seen if row.get("record_kind") == "invocation" and row["success"])
    assert invocation["fallback_used"] is True
    assert invocation["estimated_cost_usd"] == 6.0
    assert invocation["cost_tier"] == "low"
    assert invocation["model_id"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_a_malformed_request_does_not_advance(monkeypatch):
    class Runtime:
        name = "stub"

        def __init__(self):
            self.calls = []

        def missing_fields(self, ref):
            return ""

        def invoke(self, ref, *, session_id, input_text, shadow_alias_id=None):
            self.calls.append(ref.model_id)
            raise ClientError(
                {"Error": {"Code": "ValidationException", "Message": "Malformed input request"}},
                "Converse",
            )

    runtime = Runtime()
    monkeypatch.setattr(bedrock_invoke, "get_runtime", lambda: runtime)
    monkeypatch.setattr(bedrock_invoke, "emit_model_event", lambda *_a, **_k: None)
    monkeypatch.setattr(bedrock_invoke.time, "sleep", lambda *_a, **_k: None)

    with pytest.raises(bedrock_invoke.StepFailed):
        bedrock_invoke.invoke_agent(
            "a", "b", "run-1", "hello", max_retries=0, model_category="default",
        )
    assert runtime.calls == ["deepseek.v3.2"]


def test_observatory_failures_are_swallowed(monkeypatch):
    from src.orchestrator.model_map import emit_model_event

    def _boom(_record):
        raise RuntimeError("observatory down")

    monkeypatch.setattr(
        "src.orchestrator.mcp_observatory.record_model_selection", _boom,
    )
    emit_model_event({
        "agent": "a",
        "team": "t",
        "run_id": "r",
        "category": "default",
        "model_id": "deepseek.v3.2",
        "estimated_cost_usd": 0.01,
        "cost_tier": "low",
        "success": True,
    })


def test_every_agent_names_a_category_the_map_knows():
    import json

    known = set(load_model_map()["categories"])
    teams = REPO / "config" / "examples" / "teams"
    found = 0
    for path in teams.glob("*/v1/team.json"):
        doc = json.loads(path.read_text())
        for agent in doc["agents"]:
            found += 1
            category = agent.get("model_category")
            assert category in known, (path.name, agent["id"], category)
            assert "model_id" not in (agent.get("bedrock") or {}), agent["id"]
    assert found >= 14


def test_the_template_embedding_id_is_the_map_primary():
    primary = resolve_model("embeddings").model_id
    template = (REPO / "infra" / "template.yaml").read_text()
    assert primary in template
    from src.orchestrator.health_kb import embedding_model_id
    assert embedding_model_id() == primary


def test_no_model_id_outside_the_map():
    """Every model id in the runtime, configs, and templates is a map key."""
    known = set(load_model_map()["models"])
    offenders = []
    files = []
    for root in _SCAN_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in _SUFFIXES:
                continue
            if path in _SKIP_FILES or "__pycache__" in path.parts:
                continue
            files.append(path)
    assert files, "the scan found nothing"
    for path in files:
        text = path.read_text(errors="ignore")
        for match in _TOKEN.finditer(text):
            token = match.group(0).rstrip(".,);\"'")
            if not any(ch.isdigit() for ch in token):
                continue
            if token in _FAMILY_PREFIXES or token in known:
                continue
            offenders.append(f"{path.relative_to(REPO)}: {token}")
    assert offenders == []
