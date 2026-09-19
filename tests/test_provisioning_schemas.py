"""The schema catalogue, and where `schema_ref` comes from.

`schema_ref` names the JSON Schema an agent's output is validated against.
Nothing served the list, so a caller had to know the fourteen valid strings by
heart, and `POST /agents` accepted any string it was given — the mistake only
surfaced later, when a run failed output validation far from the call that
caused it. Meanwhile every role already declared a `schema_ref`, so the field
was asking the caller to re-derive a decision the role had made.

None of this needs AWS: the catalogue is read from the bundled files, and the
role lookup is stubbed.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = REPO / "config" / "examples" / "schemas"


@pytest.fixture(scope="module")
def handler_module():
    """Import the provisioning Lambda without its AWS-dependent import side effects."""
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("ARTIFACT_BUCKET", "test-bucket")
    os.environ.setdefault("BEDROCK_ROLE_ARN", "arn:aws:iam::123456789012:role/test")
    # It imports `.provision_team`, so it has to be loaded as part of its
    # package (a PEP 420 namespace package — there is no __init__.py), the
    # same way Lambda resolves the `config/examples/lambda_handler.handler`
    # entry point. Loading it by file path fails on that relative import.
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return importlib.import_module("config.examples.lambda_handler")


@pytest.fixture
def cfg(handler_module):
    return handler_module.get_config()


def test_config_points_at_the_bundled_schemas(cfg):
    assert Path(cfg["schemas_dir"]).resolve() == SCHEMAS_DIR.resolve()


def test_catalogue_lists_every_schema_file(handler_module, cfg):
    on_disk = {p.stem for p in SCHEMAS_DIR.glob("*.json")}
    served = {entry["schema_ref"] for entry in handler_module.list_schemas(cfg)}
    assert served == on_disk
    assert served, "no schemas found — the catalogue would be silently empty"


def test_catalogue_carries_the_title_and_fields(handler_module, cfg):
    entry = next(e for e in handler_module.list_schemas(cfg) if e["schema_ref"] == "creative_brief_v1")
    raw = json.loads((SCHEMAS_DIR / "creative_brief_v1.json").read_text())
    # A list of bare ids does not let anyone choose between two schemas.
    assert entry["title"] == raw["title"]
    assert entry["fields"] == sorted(raw["properties"].keys())
    assert entry["required"] == raw["required"]


def test_every_role_names_a_schema_that_exists(handler_module, cfg):
    # The defaulting below is only safe because this holds. If a role ever
    # names a schema that was renamed or deleted, agent creation for that role
    # would start failing, and this is where that shows up.
    roles = json.loads((REPO / "config" / "examples" / "roles.json").read_text())["roles"]
    known = handler_module.schema_refs(cfg)
    unknown = sorted({r["schema_ref"] for r in roles if r.get("schema_ref") not in known})
    assert unknown == []


def test_list_endpoint_returns_the_catalogue(handler_module, cfg):
    resp = handler_module.handle_schemas("GET", None, {}, cfg)
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["count"] == len(body["schemas"]) > 0


def test_single_schema_endpoint(handler_module, cfg):
    resp = handler_module.handle_schemas("GET", "approval_v1", {}, cfg)
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"])["schema_ref"] == "approval_v1"


def test_unknown_schema_names_the_valid_ones(handler_module, cfg):
    # "not found" alone leaves the caller exactly where they started.
    resp = handler_module.handle_schemas("GET", "no_such_v9", {}, cfg)
    assert resp["statusCode"] == 404
    assert "approval_v1" in json.loads(resp["body"])["error"]


def test_schemas_route_is_registered(handler_module):
    assert "schemas" in handler_module._ROUTES


def test_write_methods_are_rejected(handler_module, cfg):
    # The catalogue is the deployed bundle; it is not writable over HTTP.
    assert handler_module.handle_schemas("POST", None, {"x": 1}, cfg)["statusCode"] == 405


def test_missing_directory_yields_an_empty_catalogue(handler_module, cfg, tmp_path):
    broken = dict(cfg, schemas_dir=str(tmp_path / "nope"))
    assert handler_module.list_schemas(broken) == []


def test_unreadable_schema_does_not_hide_the_others(handler_module, cfg, tmp_path):
    (tmp_path / "good_v1.json").write_text(json.dumps({"title": "Good", "properties": {"a": {}}}))
    (tmp_path / "broken_v1.json").write_text("{not json")
    entries = handler_module.list_schemas(dict(cfg, schemas_dir=str(tmp_path)))
    by_ref = {e["schema_ref"]: e for e in entries}
    assert set(by_ref) == {"good_v1", "broken_v1"}
    assert by_ref["broken_v1"]["error"] == "unreadable"
    assert by_ref["good_v1"]["fields"] == ["a"]


# ── POST /agents: where schema_ref comes from ────────────────────────────────


@pytest.fixture
def post_agent(handler_module, cfg, monkeypatch):
    """Call the agents POST path with Bedrock and the role store stubbed out."""
    created = {}

    def fake_create_bedrock_agent(**kwargs):
        created.clear()
        created.update(kwargs)
        return "AGENT123", "ALIAS123"

    monkeypatch.setattr(handler_module, "create_bedrock_agent", fake_create_bedrock_agent)
    monkeypatch.setattr(handler_module, "_load_roles", lambda _cfg: {"roles": []})
    monkeypatch.setattr(
        handler_module,
        "build_role_index",
        lambda _roles: {"PBM-001": {"role_id": "PBM-001", "schema_ref": "creative_brief_v1"}},
    )
    monkeypatch.setattr(handler_module, "sanitise_agent_name", lambda name: name)

    def call(body):
        resp = handler_module.handle_agents("POST", None, body, cfg)
        return resp, created

    return call


BASE = {"name": "writer", "role_id": "PBM-001", "goal_template": "Write {topic}"}


def test_schema_ref_defaults_to_the_role(post_agent):
    # The form used to demand this field, so a caller had to look up what the
    # role had already decided — and could contradict it.
    resp, created = post_agent(dict(BASE))
    assert resp["statusCode"] == 201
    assert created["schema_ref"] == "creative_brief_v1"
    assert json.loads(resp["body"])["schema_ref"] == "creative_brief_v1"


def test_an_explicit_schema_ref_still_overrides(post_agent):
    resp, created = post_agent(dict(BASE, schema_ref="draft_pack_v1"))
    assert resp["statusCode"] == 201
    assert created["schema_ref"] == "draft_pack_v1"


def test_blank_schema_ref_is_treated_as_absent(post_agent):
    # A form that submits "" for an untouched field must not defeat the default.
    resp, created = post_agent(dict(BASE, schema_ref="   "))
    assert resp["statusCode"] == 201
    assert created["schema_ref"] == "creative_brief_v1"


def test_an_unknown_schema_ref_is_rejected_before_the_agent_is_built(post_agent):
    resp, created = post_agent(dict(BASE, schema_ref="typo_v1"))
    assert resp["statusCode"] == 400
    error = json.loads(resp["body"])["error"]
    assert "typo_v1" in error and "creative_brief_v1" in error
    assert created == {}, "a rejected request must not reach Bedrock"


def test_schema_ref_is_no_longer_a_required_field(post_agent):
    resp, _ = post_agent({"name": "writer", "role_id": "PBM-001"})
    assert resp["statusCode"] == 400
    assert "goal_template" in json.loads(resp["body"])["error"]
    assert "schema_ref" not in json.loads(resp["body"])["error"]


def test_an_unknown_role_lists_the_valid_roles(post_agent):
    resp, created = post_agent(dict(BASE, role_id="NOPE-999"))
    assert resp["statusCode"] == 400
    assert "PBM-001" in json.loads(resp["body"])["error"]
    assert created == {}
