"""`health_prep` reads the person's own record, as that person, and nobody else does.

ContextWeave keeps medical records in a separate database behind the only
authorizer on that API, for one reason: a content team asking about "work under
pressure" must not be able to retrieve a chunk of a discharge summary. Wiring a
tool to that store puts the property back at risk, so it has two barriers and
this file is about both.

**No service credential.** The call carries the bearer token the person
presented when they started the run. There is nothing in the worker's
environment that could read the record on its own, so there is no run that can
read it without a person behind it.

**A team allowlist in code.** Team configs are JSON in S3, edited with no deploy
and no review. A restriction written only there is not a restriction.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orchestrator import tool_registry, tool_rules  # noqa: E402
from orchestrator import contextweave_client  # noqa: E402
from orchestrator.tools import weave_tools  # noqa: E402

TEAM_JSON = json.loads(
    (REPO / "config" / "examples" / "teams" / "health_prep" / "v1" / "team.json").read_text())


# ── barrier 1: it is called as the person, or not at all ────────────────────

def test_the_tool_needs_the_callers_token():
    assert tool_rules.requires_caller_token("query_health_record")


def test_without_a_token_it_refuses_rather_than_reporting_an_empty_record(monkeypatch):
    """"Nothing found" and "I could not ask" are different answers, and only
    one of them is true. Reporting the first would tell the agent the person's
    records are empty."""
    called = []
    monkeypatch.setattr(contextweave_client, "_post",
                        lambda *a, **k: called.append(1) or {})
    result = contextweave_client.query_health_record("chest tightness", token="")
    assert "error" in result
    assert not called, "a request was made without a token"


def test_the_token_is_sent_as_a_bearer_header(monkeypatch):
    seen = {}

    def fake_post(path, body, max_retries=2, bearer=""):
        seen.update(path=path, body=body, bearer=bearer)
        return {"answer": "a", "found": True, "sources": ["s"]}

    monkeypatch.setattr(contextweave_client, "_post", fake_post)
    contextweave_client.query_health_record("chest tightness", token="tok-123")
    assert seen["bearer"] == "tok-123"
    assert seen["path"] == "/health/query"
    assert seen["body"]["question"] == "chest tightness"


def test_the_bearer_header_reaches_the_request(monkeypatch):
    """Driven through the real `_post`, because a client that accepted the
    argument and never set the header would pass the test above."""
    captured = {}

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"found": false, "answer": ""}'

    def fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.headers)
        return FakeResponse()

    monkeypatch.setattr(contextweave_client, "base_url", lambda: "https://cw.example")
    monkeypatch.setattr(contextweave_client.urllib.request, "urlopen", fake_urlopen)
    contextweave_client.query_health_record("q", token="tok-abc")
    # urllib title-cases header names.
    assert captured["headers"].get("Authorization") == "Bearer tok-abc"


def test_the_token_comes_from_the_worker_not_from_the_config(monkeypatch):
    """A config that set `caller_token` in its args would choose the identity
    the record is read under. It is overwritten, not defaulted."""
    seen = {}
    monkeypatch.setitem(tool_registry.TOOL_REGISTRY, "query_health_record",
                        lambda **kw: seen.update(kw) or {})
    tool_registry.execute_tool(
        "query_health_record",
        {"question": "q", "caller_token": "token-from-s3"},
        team="health_prep", caller_token="token-from-the-person")
    assert seen["caller_token"] == "token-from-the-person"


def test_a_tool_with_no_such_rule_never_receives_the_token(monkeypatch):
    seen = {}
    monkeypatch.setitem(tool_registry.TOOL_REGISTRY, "crawl_site",
                        lambda **kw: seen.update(kw) or {})
    tool_registry.execute_tool("crawl_site", {"url": "https://x", "caller_token": "leak"},
                               team="health_prep", caller_token="tok")
    assert "caller_token" not in seen


# ── barrier 2: one team, enforced where S3 cannot reach ─────────────────────

def test_only_health_prep_may_call_it(monkeypatch):
    monkeypatch.setitem(tool_registry.TOOL_REGISTRY, "query_health_record",
                        lambda **kw: {"answer": "a medical record"})
    with pytest.raises(PermissionError) as raised:
        tool_registry.execute_tool("query_health_record", {"question": "q"},
                                   team="linkedin_quick_post", caller_token="tok")
    assert "health_prep" in str(raised.value)


def test_an_unset_team_is_refused_rather_than_allowed(monkeypatch):
    """Fail closed. `execute_post_tools` passes no team, so a post-tool
    declaring this is refused by construction rather than by remembering to."""
    monkeypatch.setitem(tool_registry.TOOL_REGISTRY, "query_health_record",
                        lambda **kw: {"answer": "a medical record"})
    with pytest.raises(PermissionError):
        tool_registry.execute_tool("query_health_record", {"question": "q"},
                                   team="", caller_token="tok")


def test_health_prep_itself_is_allowed(monkeypatch):
    monkeypatch.setitem(tool_registry.TOOL_REGISTRY, "query_health_record",
                        lambda **kw: {"found": False})
    assert tool_registry.execute_tool("query_health_record", {"question": "q"},
                                      team="health_prep", caller_token="tok") == {"found": False}


def test_the_restriction_is_not_expressed_only_in_the_team_config():
    """If the allowlist lived in team.json it would be editable in S3 without
    a deploy, which is the same as not existing."""
    rule = tool_rules.RULES["query_health_record"]
    assert rule.only_teams == ("health_prep",)
    source = (REPO / "src" / "orchestrator" / "tool_registry.py").read_text()
    assert "team_allowed" in source, "nothing enforces the allowlist at execution"


# ── the wiring, in both directions ──────────────────────────────────────────

def test_health_prep_declares_the_tool():
    """Both steps retrieve. The insights step used to see only the log, so it
    wrote questions from a paraphrase and never from the records."""
    assert [s["step"] for s in TEAM_JSON["workflow"]] == ["HP_log", "HP_insights"]
    for step in TEAM_JSON["workflow"]:
        pre = step.get("pre_tools") or []
        assert [t["name"] for t in pre] == ["query_health_record"], step["step"]
        assert pre[0]["args"]["source_key"] == "request.experiencing"
        assert pre[0]["args"]["facets"] is True


def test_no_other_team_declares_it():
    teams = (REPO / "config" / "examples" / "teams")
    offenders = [p.relative_to(REPO) for p in teams.rglob("team.json")
                 if "query_health_record" in p.read_text()
                 and json.loads(p.read_text())["team"]["name"] != "health_prep"]
    assert not offenders, f"a team outside the allowlist declares it: {offenders}"


def test_the_agent_is_told_what_a_failed_lookup_looks_like():
    """An agent that cannot tell "the lookup failed" from "your records say
    nothing" fills the blank in, and a fabricated medical history is exactly as
    schema-valid as a real one."""
    constraints = " ".join(TEAM_JSON["globals"]["hard_constraints"])
    assert "query_health_record" in constraints
    assert "error" in constraints and "found false" in constraints


def test_the_token_never_enters_the_step_record():
    """`dao.put_step` persists `step_inputs`. The token is passed as an
    argument to `execute_pre_tools` for exactly that reason."""
    worker = (REPO / "src" / "orchestrator" / "worker_handler.py").read_text()
    assert "caller_token=caller_token" in worker
    assert "step_inputs[\"caller_token\"]" not in worker
    assert "step_inputs['caller_token']" not in worker


def test_the_trigger_forwards_the_token_it_was_given():
    trigger = (REPO / "src" / "orchestrator" / "trigger_handler.py").read_text()
    assert '"caller_token": _bearer_token(event)' in trigger
