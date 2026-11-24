"""Multi-node A2A peer registry and capability failover router for TeamWeave.

Provides:
- :class:`PeerRecord`: metadata, A2A agent card, capabilities, health, and latency for a peer node.
- :class:`SQLitePeerRegistry`: SQLite WAL-backed peer registry for standalone and edge nodes.
- :class:`PeerRouter`: latency-ordered peer selection and automatic failover delegation
  enforcing :class:`LocalTeamPolicy` delegation and duration guardrails.

Zero ``weaveos`` imports (standalone dual-mode).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.orchestrator.local_policy import LocalTeamPolicy, PolicyDeniedError


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class PeerFailoverExhaustedError(RuntimeError):
    """Raised when no healthy capable peer succeeds during failover delegation."""

    def __init__(self, message: str, failed_peers: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.failed_peers: list[dict[str, Any]] = list(failed_peers or [])


def _extract_capabilities_from_card(agent_card: dict[str, Any]) -> list[str]:
    caps: list[str] = []
    seen: set[str] = set()

    raw_caps = agent_card.get("capabilities")
    if isinstance(raw_caps, list):
        for item in raw_caps:
            val = str(item).strip()
            if val and val not in seen:
                seen.add(val)
                caps.append(val)

    raw_skills = agent_card.get("skills")
    if isinstance(raw_skills, list):
        for skill in raw_skills:
            if isinstance(skill, dict):
                val = str(skill.get("id") or skill.get("name") or "").strip()
            else:
                val = str(skill).strip()
            if val and val not in seen:
                seen.add(val)
                caps.append(val)

    return caps


@dataclass
class PeerRecord:
    """Registered A2A peer agent node."""

    peer_id: str
    base_url: str
    agent_card: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    healthy: bool = True
    latency_ms: float = 0.0
    last_seen_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.peer_id = str(self.peer_id or "").strip()
        if not self.peer_id:
            raise ValueError("PeerRecord.peer_id must be a non-empty string")
        self.base_url = str(self.base_url or "").strip()
        if not self.base_url:
            raise ValueError("PeerRecord.base_url must be a non-empty string")
        if not isinstance(self.agent_card, dict):
            raise TypeError("PeerRecord.agent_card must be a dict")
        if not self.capabilities and self.agent_card:
            self.capabilities = _extract_capabilities_from_card(self.agent_card)
        else:
            self.capabilities = [str(c).strip() for c in self.capabilities if str(c).strip()]
        self.healthy = bool(self.healthy)
        self.latency_ms = max(0.0, float(self.latency_ms))
        if not isinstance(self.last_seen_at, str) or not self.last_seen_at.strip():
            self.last_seen_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "base_url": self.base_url,
            "agent_card": dict(self.agent_card),
            "capabilities": list(self.capabilities),
            "healthy": self.healthy,
            "latency_ms": self.latency_ms,
            "last_seen_at": self.last_seen_at,
        }


class SQLitePeerRegistry:
    """SQLite WAL-backed A2A peer registry."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS a2a_peers (
                peer_id TEXT PRIMARY KEY,
                base_url TEXT NOT NULL,
                agent_card_json TEXT NOT NULL,
                capabilities_json TEXT NOT NULL,
                healthy INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLitePeerRegistry:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> PeerRecord:
        return PeerRecord(
            peer_id=row["peer_id"],
            base_url=row["base_url"],
            agent_card=json.loads(row["agent_card_json"]),
            capabilities=json.loads(row["capabilities_json"]),
            healthy=bool(row["healthy"]),
            latency_ms=float(row["latency_ms"]),
            last_seen_at=row["last_seen_at"],
        )

    def register_peer(
        self,
        peer: PeerRecord | str,
        base_url: str | None = None,
        *,
        agent_card: dict[str, Any] | None = None,
        capabilities: list[str] | None = None,
        healthy: bool = True,
        latency_ms: float = 0.0,
        last_seen_at: str | None = None,
    ) -> PeerRecord:
        """Register or update an A2A peer in SQLite."""
        if isinstance(peer, PeerRecord):
            record = peer
        else:
            if base_url is None:
                raise ValueError("base_url is required when registering a peer by peer_id")
            record = PeerRecord(
                peer_id=peer,
                base_url=base_url,
                agent_card=dict(agent_card or {}),
                capabilities=list(capabilities) if capabilities is not None else [],
                healthy=healthy,
                latency_ms=latency_ms,
                last_seen_at=last_seen_at or _utc_now(),
            )

        self._conn.execute(
            """
            INSERT INTO a2a_peers (
                peer_id, base_url, agent_card_json, capabilities_json, healthy, latency_ms, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(peer_id) DO UPDATE SET
                base_url = excluded.base_url,
                agent_card_json = excluded.agent_card_json,
                capabilities_json = excluded.capabilities_json,
                healthy = excluded.healthy,
                latency_ms = excluded.latency_ms,
                last_seen_at = excluded.last_seen_at;
            """,
            (
                record.peer_id,
                record.base_url,
                json.dumps(record.agent_card, sort_keys=True),
                json.dumps(record.capabilities),
                1 if record.healthy else 0,
                record.latency_ms,
                record.last_seen_at,
            ),
        )
        self._conn.commit()
        return record

    def get_peer(self, peer_id: str) -> PeerRecord | None:
        row = self._conn.execute(
            "SELECT * FROM a2a_peers WHERE peer_id = ?;",
            (str(peer_id).strip(),),
        ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def mark_health(
        self,
        peer_id: str,
        healthy: bool,
        latency_ms: float | None = None,
    ) -> PeerRecord | None:
        """Update health status, optional latency, and ``last_seen_at`` for a peer."""
        existing = self.get_peer(peer_id)
        if existing is None:
            return None
        new_latency = existing.latency_ms if latency_ms is None else max(0.0, float(latency_ms))
        now_iso = _utc_now()
        self._conn.execute(
            """
            UPDATE a2a_peers
            SET healthy = ?, latency_ms = ?, last_seen_at = ?
            WHERE peer_id = ?;
            """,
            (1 if healthy else 0, new_latency, now_iso, existing.peer_id),
        )
        self._conn.commit()
        return self.get_peer(existing.peer_id)

    def list_peers(
        self,
        capability: str | None = None,
        healthy_only: bool = True,
    ) -> list[PeerRecord]:
        """List registered peers ordered by lowest ``latency_ms`` first."""
        if healthy_only:
            rows = self._conn.execute(
                "SELECT * FROM a2a_peers WHERE healthy = 1 ORDER BY latency_ms ASC, peer_id ASC;"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM a2a_peers ORDER BY latency_ms ASC, peer_id ASC;"
            ).fetchall()

        records = [self._row_to_record(r) for r in rows]
        if capability is not None:
            cap_norm = str(capability).strip()
            records = [r for r in records if cap_norm in r.capabilities]
        return records

    def remove_peer(self, peer_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM a2a_peers WHERE peer_id = ?;",
            (str(peer_id).strip(),),
        )
        self._conn.commit()
        return cur.rowcount > 0


class PeerRouter:
    """Selects lowest-latency healthy A2A peers and delegates with automatic failover."""

    def __init__(
        self,
        registry: SQLitePeerRegistry,
        policy: LocalTeamPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy

    def select_peer(
        self,
        capability: str,
        *,
        exclude_peer_ids: set[str] | None = None,
    ) -> PeerRecord | None:
        """Pick the lowest-latency healthy peer supporting ``capability``."""
        excluded = set(exclude_peer_ids or ())
        candidates = self.registry.list_peers(capability=capability, healthy_only=True)
        for peer in candidates:
            if peer.peer_id not in excluded:
                return peer
        return None

    def delegate_with_failover(
        self,
        capability: str,
        payload: dict[str, Any],
        dispatch_fn: Callable[[PeerRecord, dict[str, Any]], Any],
        *,
        role: str | None = None,
    ) -> dict[str, Any]:
        """Delegate ``payload`` to capable peers in latency order with automatic failover.

        Marks failing peers unhealthy in ``self.registry`` and enforces
        ``self.policy`` (max steps / allowed roles / max duration) before each attempt.
        """
        started = time.monotonic()
        tried: set[str] = set()
        failed_peers: list[dict[str, Any]] = []
        attempt_index = 0
        policy_role = role if role is not None else capability

        while True:
            peer = self.select_peer(capability, exclude_peer_ids=tried)
            if peer is None:
                break

            if self.policy is not None:
                elapsed = time.monotonic() - started
                allowed, reason = self.policy.validate_step(
                    step_index=attempt_index,
                    role=policy_role,
                    elapsed_seconds=elapsed,
                )
                if not allowed:
                    raise PolicyDeniedError(reason or "Delegation denied by LocalTeamPolicy")

            tried.add(peer.peer_id)
            attempt_index += 1
            call_start = time.monotonic()
            try:
                response = dispatch_fn(peer, payload)
                measured_ms = round((time.monotonic() - call_start) * 1000.0, 3)
                self.registry.mark_health(peer.peer_id, True, latency_ms=measured_ms)
                return {
                    "status": "succeeded",
                    "peer_id": peer.peer_id,
                    "base_url": peer.base_url,
                    "capability": capability,
                    "attempts": attempt_index,
                    "failed_peers": failed_peers,
                    "result": response,
                }
            except Exception as exc:
                measured_ms = round((time.monotonic() - call_start) * 1000.0, 3)
                self.registry.mark_health(peer.peer_id, False, latency_ms=measured_ms)
                failed_peers.append(
                    {
                        "peer_id": peer.peer_id,
                        "base_url": peer.base_url,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )

        raise PeerFailoverExhaustedError(
            f"All healthy peers exhausted for capability={capability!r} "
            f"(attempted={len(failed_peers)})",
            failed_peers=failed_peers,
        )
