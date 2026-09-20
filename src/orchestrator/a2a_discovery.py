"""Find a sibling service by its A2A Agent Card instead of guessing its paths.

`contextweave_client` built every request as `{CONTEXTWEAVE_URL}` + a path
constant held in *this* repository. That works right up until ContextWeave
moves a route, at which point the break surfaces as a 404 that the RAG layer
degrades past silently -- the run just loses its grounding and nobody is told.

A2A's answer is a document. `GET /.well-known/agent-card.json` returns the
interface URL and the skills the service actually serves, so the caller reads
where to go rather than assuming it.

Deliberately additive. The env var is still the seed -- discovery needs a
first address -- and when no card is served, or it is unreadable, or it names
no usable interface, the caller falls back to exactly the behaviour it had
before. A knowledge layer that degrades to no context is the designed
behaviour of every RAG mode here; a discovery layer that hard-failed would be
strictly worse than the hardcoding it replaces.

The card is fetched once per Lambda container and cached, because a cold call
per turn would add a round trip to every agent step for a document that
changes at deploy time.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .logger import get_logger

log = get_logger("a2a_discovery")

AGENT_CARD_PATH = "/.well-known/agent-card.json"
_TIMEOUT_SECONDS = 5
# Long enough that no run pays for discovery twice, short enough that a
# redeployed sibling is picked up without recycling this container.
_CACHE_TTL_SECONDS = 300

# A card is a document from another service. It is read for the one field
# needed and never trusted to be well-formed.
_cache: Dict[str, Any] = {}


# "no entry / expired" and "cached a negative result" are different answers,
# and None cannot express both: an expired entry that returned None would be
# read as "this sibling serves no card" and never re-fetched.
_MISS = object()


def _cache_get(base: str) -> Any:
    entry = _cache.get(base)
    if not entry:
        return _MISS
    if time.time() - entry["at"] > _CACHE_TTL_SECONDS:
        _cache.pop(base, None)
        return _MISS
    return entry["card"]


def _cache_put(base: str, card: Optional[Dict[str, Any]]) -> None:
    # A miss is cached too: a sibling that serves no card should not be asked
    # again on every single turn.
    _cache[base] = {"at": time.time(), "card": card}


def clear_cache() -> None:
    _cache.clear()


def fetch_agent_card(base_url: str, *, timeout: int = _TIMEOUT_SECONDS) -> Optional[Dict[str, Any]]:
    """The sibling's card, or None if it does not serve one."""
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return None
    cached = _cache_get(base)
    if cached is not _MISS:
        return cached

    try:
        req = urllib.request.Request(base + AGENT_CARD_PATH, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            card = json.loads(resp.read().decode("utf-8"))
        if not isinstance(card, dict):
            raise ValueError("card is not an object")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
        # Not an error. A sibling that predates A2A serves no card, and the
        # caller's existing behaviour is the correct fallback.
        log.info("a2a_card_unavailable", extra={"base_url": base, "reason": str(exc)[:200]})
        _cache_put(base, None)
        return None

    # `name` is a reserved LogRecord attribute: passing it in `extra` raises
    # KeyError inside logging, which would have turned every *successful*
    # discovery into an exception -- the one path that must never fail.
    log.info("a2a_card_discovered",
             extra={"base_url": base, "agent_name": str(card.get("name"))[:80],
                    "skill_count": len(card.get("skills") or [])})
    _cache_put(base, card)
    return card


def interface_url(card: Optional[Dict[str, Any]], *, binding: str = "HTTP+JSON") -> str:
    """The URL for the first interface with this binding.

    A2A orders `supportedInterfaces` by preference and says a client takes the
    first it supports. Taking the first *entry* regardless would send HTTP to
    a gRPC endpoint on any sibling that lists more than one.
    """
    interfaces = (card or {}).get("supportedInterfaces")
    if not isinstance(interfaces, list):
        return ""
    for entry in interfaces:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("protocolBinding") or "") != binding:
            continue
        url = str(entry.get("url") or "").strip().rstrip("/")
        if url:
            return url
    return ""


def skill_ids(card: Optional[Dict[str, Any]]) -> List[str]:
    """Every skill the sibling advertises."""
    skills = (card or {}).get("skills")
    if not isinstance(skills, list):
        return []
    return [str(s.get("id")) for s in skills if isinstance(s, dict) and s.get("id")]


def offers_skill(card: Optional[Dict[str, Any]], skill_id: str) -> bool:
    return skill_id in skill_ids(card)


def resolve_base_url(configured: str, *, binding: str = "HTTP+JSON") -> str:
    """Where to send requests: what the card says, else what was configured.

    The env var remains the seed, because discovery needs a first address.
    What changes is that the *serving* URL now comes from the sibling rather
    than being assumed to equal the seed.
    """
    seed = str(configured or "").strip().rstrip("/")
    if not seed:
        return ""
    discovered = interface_url(fetch_agent_card(seed), binding=binding)
    if discovered and discovered != seed:
        log.info("a2a_interface_differs_from_seed",
                 extra={"seed": seed, "discovered": discovered})
    return discovered or seed
