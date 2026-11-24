"""Unit tests for TeamWeave SQLitePeerRegistry and PeerRouter failover."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.orchestrator.local_policy import LocalTeamPolicy, PolicyDeniedError
from src.teamweave.peer_registry import (
    PeerFailoverExhaustedError,
    PeerRecord,
    PeerRouter,
    SQLitePeerRegistry,
)


def test_register_list_capability_filter_and_latency_order(tmp_path: Path) -> None:
    with SQLitePeerRegistry(tmp_path / "peers.db") as registry:
        registry.register_peer(
            PeerRecord(
                peer_id="node-slow",
                base_url="http://192.168.1.20:9000",
                agent_card={"name": "Slow Node", "skills": [{"id": "research"}, {"id": "summarize"}]},
                healthy=True,
                latency_ms=120.0,
            )
        )
        registry.register_peer(
            "node-fast",
            "http://192.168.1.10:9000",
            agent_card={"name": "Fast Node"},
            capabilities=["research"],
            healthy=True,
            latency_ms=18.5,
        )
        registry.register_peer(
            "node-down",
            "http://192.168.1.30:9000",
            capabilities=["research"],
            healthy=False,
            latency_ms=5.0,
        )

        healthy_research = registry.list_peers(capability="research", healthy_only=True)
        assert [p.peer_id for p in healthy_research] == ["node-fast", "node-slow"]

        all_research = registry.list_peers(capability="research", healthy_only=False)
        assert [p.peer_id for p in all_research] == ["node-down", "node-fast", "node-slow"]

        summarize_peers = registry.list_peers(capability="summarize", healthy_only=True)
        assert [p.peer_id for p in summarize_peers] == ["node-slow"]

        router = PeerRouter(registry)
        selected = router.select_peer("research")
        assert selected is not None
        assert selected.peer_id == "node-fast"

        selected_excl = router.select_peer("research", exclude_peer_ids={"node-fast"})
        assert selected_excl is not None
        assert selected_excl.peer_id == "node-slow"


def test_automatic_failover_marks_primary_unhealthy_and_succeeds_on_secondary(tmp_path: Path) -> None:
    with SQLitePeerRegistry(tmp_path / "peers.db") as registry:
        registry.register_peer(
            "peer-primary",
            "http://10.0.0.1:9000",
            capabilities=["vision"],
            healthy=True,
            latency_ms=12.0,
        )
        registry.register_peer(
            "peer-secondary",
            "http://10.0.0.2:9000",
            capabilities=["vision"],
            healthy=True,
            latency_ms=45.0,
        )

        called: list[str] = []

        def _dispatch(peer: PeerRecord, payload: dict[str, Any]) -> dict[str, Any]:
            called.append(peer.peer_id)
            if peer.peer_id == "peer-primary":
                raise ConnectionError("ECONNREFUSED")
            return {"handled_by": peer.peer_id, "echo": payload["task"]}

        router = PeerRouter(registry)
        outcome = router.delegate_with_failover("vision", {"task": "inspect"}, _dispatch)

        assert called == ["peer-primary", "peer-secondary"]
        assert outcome["status"] == "succeeded"
        assert outcome["peer_id"] == "peer-secondary"
        assert outcome["attempts"] == 2
        assert len(outcome["failed_peers"]) == 1
        assert outcome["failed_peers"][0]["peer_id"] == "peer-primary"
        assert outcome["result"] == {"handled_by": "peer-secondary", "echo": "inspect"}

        # Primary peer is now marked unhealthy in SQLite
        primary = registry.get_peer("peer-primary")
        assert primary is not None
        assert primary.healthy is False

        # Secondary remains healthy
        secondary = registry.get_peer("peer-secondary")
        assert secondary is not None
        assert secondary.healthy is True


def test_failover_exhausted_raises_error(tmp_path: Path) -> None:
    with SQLitePeerRegistry(tmp_path / "peers.db") as registry:
        registry.register_peer(
            "peer-1",
            "http://10.0.0.1:9000",
            capabilities=["code"],
            healthy=True,
            latency_ms=10.0,
        )

        def _always_fail(peer: PeerRecord, payload: dict[str, Any]) -> Any:
            _ = (peer, payload)
            raise TimeoutError("timed out")

        router = PeerRouter(registry)
        with pytest.raises(PeerFailoverExhaustedError) as exc_info:
            router.delegate_with_failover("code", {"prompt": "hi"}, _always_fail)

        assert len(exc_info.value.failed_peers) == 1
        assert exc_info.value.failed_peers[0]["peer_id"] == "peer-1"


def test_local_team_policy_enforces_max_steps_and_allowed_roles(tmp_path: Path) -> None:
    with SQLitePeerRegistry(tmp_path / "peers.db") as registry:
        registry.register_peer("p1", "http://10.0.0.1:9000", capabilities=["research"], latency_ms=10.0)
        registry.register_peer("p2", "http://10.0.0.2:9000", capabilities=["research"], latency_ms=20.0)

        # Max 1 delegation attempt: p1 fails, attempting p2 is blocked by max_steps=1
        policy_max_1 = LocalTeamPolicy(max_steps=1, allowed_roles={"research"})
        router_max_1 = PeerRouter(registry, policy=policy_max_1)

        def _fail_p1(peer: PeerRecord, payload: dict[str, Any]) -> dict[str, Any]:
            _ = payload
            if peer.peer_id == "p1":
                raise RuntimeError("p1 down")
            return {"ok": True}

        with pytest.raises(PolicyDeniedError, match="max_steps"):
            router_max_1.delegate_with_failover("research", {"q": "test"}, _fail_p1)

        # Disallowed role is blocked immediately before dispatching
        registry.mark_health("p1", True, latency_ms=10.0)
        policy_roles = LocalTeamPolicy(max_steps=5, allowed_roles={"planner"})
        router_roles = PeerRouter(registry, policy=policy_roles)

        with pytest.raises(PolicyDeniedError, match="allowed_roles"):
            router_roles.delegate_with_failover("research", {"q": "test"}, lambda p, d: d)
