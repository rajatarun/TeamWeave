"""Finding a sibling by its card instead of guessing its paths.

`contextweave_client` built every request as `{CONTEXTWEAVE_URL}` plus a path
constant held in *this* repository. That works until ContextWeave moves a
route, at which point the break arrives as a 404 the RAG layer degrades past
in silence -- the run loses its grounding and nobody is told.

What these pin is that discovery is strictly additive. It is the failure mode
that matters: a discovery layer that hard-failed, or that sent HTTP to a gRPC
endpoint, would be worse than the hardcoding it replaces.
"""
from __future__ import annotations

import json
import os
import urllib.error
from unittest import mock

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from src.orchestrator import a2a_discovery as disc  # noqa: E402

CARD = {
    "name": "ContextWeave",
    "supportedInterfaces": [
        {"url": "https://cw.example.com/prod", "protocolBinding": "HTTP+JSON",
         "protocolVersion": "1.0"}
    ],
    "skills": [{"id": "query-expertise"}, {"id": "feedback"}],
}


@pytest.fixture(autouse=True)
def clean_cache():
    disc.clear_cache()
    yield
    disc.clear_cache()


def serving(card, status=200):
    """A urlopen that returns this card."""
    body = json.dumps(card).encode()
    response = mock.MagicMock()
    response.read.return_value = body
    response.__enter__.return_value = response
    return mock.patch.object(disc.urllib.request, "urlopen", return_value=response)


# ── reading the card ───────────────────────────────────────────────────────

def test_the_card_is_fetched_from_the_well_known_uri():
    with serving(CARD) as urlopen:
        assert disc.fetch_agent_card("https://cw.example.com") == CARD
    assert urlopen.call_args.args[0].full_url == \
        "https://cw.example.com/.well-known/agent-card.json"


def test_the_serving_url_comes_from_the_card_not_the_seed():
    # The whole point: the sibling says where it answers.
    with serving(CARD):
        assert disc.resolve_base_url("https://seed.example.com") == "https://cw.example.com/prod"


def test_the_binding_is_matched_not_assumed():
    # A2A orders interfaces by preference and a client takes the first it
    # supports. Taking the first entry regardless would send HTTP+JSON to a
    # gRPC endpoint on any sibling that lists more than one.
    card = {"supportedInterfaces": [
        {"url": "https://grpc.example.com", "protocolBinding": "GRPC"},
        {"url": "https://http.example.com", "protocolBinding": "HTTP+JSON"},
    ]}
    assert disc.interface_url(card) == "https://http.example.com"


def test_no_matching_binding_yields_nothing_rather_than_the_wrong_one():
    card = {"supportedInterfaces": [{"url": "https://grpc.example.com", "protocolBinding": "GRPC"}]}
    assert disc.interface_url(card) == ""


def test_skills_can_be_read_before_calling():
    assert disc.skill_ids(CARD) == ["query-expertise", "feedback"]
    assert disc.offers_skill(CARD, "feedback")
    assert not disc.offers_skill(CARD, "summarise-everything")


# ── never worse than the hardcoding it replaces ────────────────────────────

def test_a_sibling_serving_no_card_falls_back_to_the_seed():
    # Every sibling predates A2A. The old behaviour is the correct fallback.
    error = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    with mock.patch.object(disc.urllib.request, "urlopen", side_effect=error):
        assert disc.resolve_base_url("https://seed.example.com") == "https://seed.example.com"


def test_an_unreachable_sibling_falls_back_to_the_seed():
    with mock.patch.object(disc.urllib.request, "urlopen",
                           side_effect=urllib.error.URLError("no route")):
        assert disc.resolve_base_url("https://seed.example.com") == "https://seed.example.com"


def test_a_malformed_card_falls_back_to_the_seed():
    response = mock.MagicMock()
    response.read.return_value = b"<html>not json</html>"
    response.__enter__.return_value = response
    with mock.patch.object(disc.urllib.request, "urlopen", return_value=response):
        assert disc.resolve_base_url("https://seed.example.com") == "https://seed.example.com"


def test_a_card_that_is_not_an_object_falls_back():
    with serving(["not", "a", "card"]):
        assert disc.resolve_base_url("https://seed.example.com") == "https://seed.example.com"


def test_a_card_with_no_usable_interface_falls_back():
    with serving({"name": "x", "skills": []}):
        assert disc.resolve_base_url("https://seed.example.com") == "https://seed.example.com"


def test_no_seed_means_no_request_at_all():
    with mock.patch.object(disc.urllib.request, "urlopen") as urlopen:
        assert disc.resolve_base_url("") == ""
    assert not urlopen.called


# ── caching ────────────────────────────────────────────────────────────────

def test_the_card_is_fetched_once_per_container():
    # A cold fetch per turn would add a round trip to every agent step for a
    # document that changes at deploy time.
    with serving(CARD) as urlopen:
        disc.resolve_base_url("https://seed.example.com")
        disc.resolve_base_url("https://seed.example.com")
    assert urlopen.call_count == 1


def test_a_sibling_with_no_card_is_not_re_asked_every_turn():
    error = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    with mock.patch.object(disc.urllib.request, "urlopen", side_effect=error) as urlopen:
        disc.resolve_base_url("https://seed.example.com")
        disc.resolve_base_url("https://seed.example.com")
    assert urlopen.call_count == 1, "a negative result must be cached too"


def test_the_cache_expires_so_a_redeployed_sibling_is_picked_up():
    import time as real_time

    with serving(CARD) as urlopen:
        disc.resolve_base_url("https://seed.example.com")
        # Relative to now, not an absolute constant: 10**9 is the year 2001,
        # which is *earlier* than the entry and expired nothing.
        later = real_time.time() + disc._CACHE_TTL_SECONDS + 1
        with mock.patch.object(disc.time, "time", return_value=later):
            disc.resolve_base_url("https://seed.example.com")
    assert urlopen.call_count == 2


# ── the client that uses it ────────────────────────────────────────────────

def test_the_contextweave_client_uses_the_discovered_url(monkeypatch):
    from src.orchestrator import contextweave_client as cw

    monkeypatch.setenv("CONTEXTWEAVE_URL", "https://seed.example.com")
    monkeypatch.delenv("A2A_DISCOVERY", raising=False)
    with serving(CARD):
        assert cw.base_url() == "https://cw.example.com/prod"


def test_discovery_can_be_switched_off(monkeypatch):
    from src.orchestrator import contextweave_client as cw

    monkeypatch.setenv("CONTEXTWEAVE_URL", "https://seed.example.com")
    monkeypatch.setenv("A2A_DISCOVERY", "0")
    with mock.patch.object(disc.urllib.request, "urlopen") as urlopen:
        assert cw.base_url() == "https://seed.example.com"
    assert not urlopen.called, "the opt-out must not even attempt a fetch"


def test_discovery_blowing_up_never_breaks_a_run(monkeypatch):
    # A knowledge layer degrading to no context is designed behaviour here; a
    # discovery layer taking the run down with it would not be.
    from src.orchestrator import contextweave_client as cw

    monkeypatch.setenv("CONTEXTWEAVE_URL", "https://seed.example.com")
    monkeypatch.delenv("A2A_DISCOVERY", raising=False)
    with mock.patch.object(disc, "resolve_base_url", side_effect=RuntimeError("boom")):
        assert cw.base_url() == "https://seed.example.com"


def test_an_unset_url_stays_unset(monkeypatch):
    from src.orchestrator import contextweave_client as cw

    monkeypatch.delenv("CONTEXTWEAVE_URL", raising=False)
    assert cw.base_url() == ""
    assert not cw.is_configured()
