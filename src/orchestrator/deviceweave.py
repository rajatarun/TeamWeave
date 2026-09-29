"""DeviceWeave HTTP client — list devices, read status, send a command.

DeviceWeave (github.com/rajatarun/DeviceWeave) publishes an API Gateway HTTP
API. The operations this module calls are the ones that API actually serves:

    GET  {DEVICEWEAVE_URL}/devices
    GET  {DEVICEWEAVE_URL}/devices/{device_id}
    POST {DEVICEWEAVE_URL}/execute     {"command": "<natural language>"}

``POST /execute`` without ``session_id`` is the one-shot path. DeviceWeave's
intent parser maps a command onto an action (``turn_on``, ``turn_off``,
``get_status``, …) and runs it through the policy engine and the provider
adapter. There is no separate status route: live state is ``get_status``,
reached by a command whose text contains "status". ``GET /devices/{id}``
returns the registry record (id, name, type, capabilities) and not the live
power state.

Auth: that HTTP API has no authorizer. Calls go out with no credential.
``DEVICEWEAVE_SECRET_ARN`` is optional. When it is set, the secret is read
from Secrets Manager and sent as ``Authorization: Bearer``. DeviceWeave does
not check that header today; the hook is for a deployment that has put a
bearer in front of the API. The secret value is ``{"key": "..."}``,
``{"token": "..."}``, or the raw token. Nothing here embeds a key.

The stack that publishes the URL is ``deviceweave-prod`` (DeviceWeave's
deploy workflow: ``deviceweave-${{ inputs.stage || 'prod' }}``), output
``ApiBaseUrl``. The worker reads it as ``DEVICEWEAVE_URL``.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from .deadline import budget_for_call
from .logger import get_logger

log = get_logger("deviceweave")

Transport = Callable[[str, str, Optional[Dict[str, Any]], Dict[str, str]], Tuple[int, Any]]

# Security devices from DeviceWeave's provider adapters. Powering one of
# these off is the "critical device" case. Lights, plugs, bulbs, and fans
# are not in this set: turning those off is an ordinary command.
CRITICAL_DEVICE_TYPES = frozenset({
    "WyzeLock",
    "RingDoorbell",
    "RingCamera",
    "WyzeCamera",
    "WyzeMotionSensor",
    "WyzeContactSensor",
    "MyQGarageDoor",
})

_UNLOCK = re.compile(r"\bunlock\b", re.I)
_FACTORY = re.compile(r"\bfactory[\s_-]*reset\b", re.I)
_POWER_OFF = re.compile(
    r"\b(turn\s+off|switch\s+off|power\s+off|shut\s+off|shut\s+down|power\s+down)\b",
    re.I,
)
_DISABLE = re.compile(r"\b(disable|disarm)\b", re.I)
_CRITICAL_WORD = re.compile(
    r"\b(locks?|alarms?|security|cameras?|doorbells?|garages?|sensors?)\b",
    re.I,
)
_OPEN_GARAGE = re.compile(r"\bopen\b.{0,40}\bgarages?\b|\bgarages?\b.{0,40}\bopen\b", re.I)
_CONFIRM = frozenset({"1", "true", "yes", "confirm"})


def base_url(env: Optional[Dict[str, str]] = None) -> str:
    source = os.environ if env is None else env
    return str(source.get("DEVICEWEAVE_URL") or "").strip().rstrip("/")


def is_confirmed(value: Any) -> bool:
    """True only for an explicit confirm flag. Absence is not confirmation."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in _CONFIRM


def _read_secret(arn: str) -> str:
    import boto3

    resp = boto3.client("secretsmanager").get_secret_value(SecretId=arn)
    raw = resp.get("SecretString") or ""
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return raw.strip()
    if isinstance(parsed, dict):
        return str(parsed.get("token") or parsed.get("key") or parsed.get("value") or "").strip()
    return raw.strip()


def headers(env: Optional[Dict[str, str]] = None,
            secret_reader: Optional[Callable[[str], str]] = None) -> Dict[str, str]:
    """Headers for one call. No Authorization unless a secret ARN is set.

    A set ARN that does not yield a token is an error: calling anyway would
    drop the credential the operator configured and look like success.
    """
    source = os.environ if env is None else env
    built = {"Content-Type": "application/json", "Accept": "application/json"}
    arn = str(source.get("DEVICEWEAVE_SECRET_ARN") or "").strip()
    if not arn:
        return built
    token = (secret_reader or _read_secret)(arn)
    if not token:
        raise RuntimeError(
            "DEVICEWEAVE_SECRET_ARN is set but the secret has no token"
        )
    built["Authorization"] = f"Bearer {token}"
    return built


def _http(method: str, url: str, body: Optional[Dict[str, Any]],
          hdrs: Dict[str, str]) -> Tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(request, timeout=budget_for_call()) as response:
            raw = response.read().decode()
            status = getattr(response, "status", 200)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        status = exc.code
    if not raw:
        return status, {}
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, {"error": raw[:500]}


def _result_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, sort_keys=True)


def _state_text(value: Any) -> str:
    """A short state. DeviceWeave returns ``{"state": "on", ...}``; the action log stores that as JSON."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        value = parsed
    if isinstance(value, dict) and value.get("state") not in (None, ""):
        return str(value["state"])
    return _result_text(value)


def _named_devices(command: str, devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text = command.lower()
    found = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        name = str(device.get("name") or "").strip()
        if len(name) >= 3 and name.lower() in text:
            found.append(device)
    return found


def _critical(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        d for d in devices
        if isinstance(d, dict) and str(d.get("device_type") or "") in CRITICAL_DEVICE_TYPES
    ]


def _target_label(command: str, matched: List[Dict[str, Any]]) -> str:
    if matched:
        return str(matched[0].get("name") or matched[0].get("id") or "device")
    word = _CRITICAL_WORD.search(command)
    if word:
        return word.group(0).lower()
    return "device"


def risky_reasons(command: str, devices: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """Why this instruction must not be sent without an explicit confirm flag.

    The four cases the team is held to: unlock, disable security, factory
    reset, and powering off a critical device. Opening a garage door is
    included with unlock — MyQ's action for ``MyQGarageDoor`` is ``open``.
    """
    text = command or ""
    catalog = devices or []
    matched = _critical(_named_devices(text, catalog))
    reasons: List[str] = []
    if _UNLOCK.search(text) or _OPEN_GARAGE.search(text):
        reasons.append("unlock")
    if _FACTORY.search(text):
        reasons.append("factory reset")
    security_target = bool(_CRITICAL_WORD.search(text)) or bool(matched)
    if _DISABLE.search(text) and security_target:
        reasons.append("disable security")
    if _POWER_OFF.search(text) and security_target:
        reasons.append("power off critical device")
    return reasons


def _proposal(command: str, devices: List[Dict[str, Any]]) -> Dict[str, Any]:
    matched = _critical(_named_devices(command, devices))
    return {
        "device": _target_label(command, matched),
        "command": command.strip(),
        "reason": "; ".join(risky_reasons(command, devices)),
    }


def _envelope(*, executed: bool = False, devices: Optional[List[Dict[str, Any]]] = None,
              actions: Optional[List[Dict[str, str]]] = None,
              state: Optional[List[Dict[str, str]]] = None,
              needs: Optional[List[Dict[str, str]]] = None,
              assumptions: Optional[List[str]] = None,
              errors: Optional[List[str]] = None,
              error: str = "") -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "devices": devices or [],
        "executed": executed,
        "actions_taken": actions or [],
        "current_state": state or [],
        "needs_confirmation": needs or [],
        "assumptions": assumptions or [],
        "errors": list(errors or []),
    }
    if error:
        body["error"] = error
        if error not in body["errors"]:
            body["errors"].append(error)
    return body


def _request(method: str, path: str, body: Optional[Dict[str, Any]] = None, *,
             transport: Optional[Transport] = None,
             env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    root = base_url(env)
    if not root:
        return {
            "ok": False,
            "status": 0,
            "error": (
                "DeviceWeave is not configured: DEVICEWEAVE_URL is empty. "
                "Set DeviceWeaveApiBaseUrl from the deviceweave-prod stack output ApiBaseUrl."
            ),
        }
    if not root.startswith("https://") and not root.startswith("http://"):
        return {"ok": False, "status": 0, "error": "DEVICEWEAVE_URL is not an http(s) URL"}
    try:
        hdrs = headers(env)
    except RuntimeError as exc:
        return {"ok": False, "status": 0, "error": str(exc)}
    url = root + path
    caller = transport or _http
    try:
        status, payload = caller(method, url, body, hdrs)
    except Exception as exc:  # noqa: BLE001 — a down sibling is an error result, not a lost run
        log.warning("deviceweave_transport_failed", extra={"method": method, "path": path})
        return {"ok": False, "status": 0, "error": f"DeviceWeave request failed: {exc}"}
    if not isinstance(payload, dict):
        payload = {"error": _result_text(payload)}
    if status < 200 or status >= 300:
        message = str(payload.get("error") or f"DeviceWeave returned HTTP {status}")
        return {"ok": False, "status": status, "error": message, "body": payload}
    return {"ok": True, "status": status, "body": payload}


def list_devices(*, transport: Optional[Transport] = None,
                 env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """GET /devices — the active registry, public view (no IP addresses)."""
    got = _request("GET", "/devices", transport=transport, env=env)
    if not got["ok"]:
        return _envelope(error=got["error"])
    devices = got["body"].get("devices") or []
    if not isinstance(devices, list):
        return _envelope(error="DeviceWeave GET /devices did not return a devices list")
    return _envelope(devices=devices)


def get_device(device_id: str, *, transport: Optional[Transport] = None,
               env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """GET /devices/{id} — one registry record."""
    quoted = urllib.parse.quote(str(device_id or "").strip(), safe="")
    if not quoted:
        return _envelope(error="device_status requires a device_id")
    got = _request("GET", f"/devices/{quoted}", transport=transport, env=env)
    if not got["ok"]:
        return _envelope(error=got["error"])
    return _envelope(devices=[got["body"]])


def _actions_from(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    kind = payload.get("type")
    if kind == "device":
        command = str(payload.get("action") or "")
        if not command:
            return []
        return [{
            "device": str(payload.get("device_name") or payload.get("device_id") or "device"),
            "command": command,
            "result": _result_text(payload.get("result")) or "ok",
        }]
    if kind == "scene":
        actions = []
        for step in payload.get("results") or []:
            if not isinstance(step, dict):
                continue
            command = str(step.get("action") or "")
            if not command:
                continue
            result = step.get("result") if step.get("success", True) else step.get("error")
            actions.append({
                "device": str(step.get("device_name") or step.get("device_id") or "device"),
                "command": command,
                "result": _result_text(result) or ("ok" if step.get("success", True) else "failed"),
            })
        return actions
    return []


def send_command(command: str, *, confirm: Any = None,
                 devices: Optional[List[Dict[str, Any]]] = None,
                 transport: Optional[Transport] = None,
                 env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """POST /execute, unless the command is risky and confirm is not set.

    ``devices`` is the registry from ``list_devices``. When it is omitted the
    risk check still sees unlock / factory reset / the critical words in the
    text; it cannot match a device name it has not been shown.
    """
    text = str(command or "").strip()
    catalog = list(devices or [])
    if not text:
        return _envelope(devices=catalog, error="no instruction to execute")
    reasons = risky_reasons(text, catalog)
    if reasons and not is_confirmed(confirm):
        return _envelope(
            devices=catalog,
            needs=[_proposal(text, catalog)],
            assumptions=["The instruction was not sent. Set confirm to yes to execute it."],
        )
    assumptions = []
    if reasons:
        assumptions.append("confirm was set, so the risky command was sent to DeviceWeave.")
    got = _request("POST", "/execute", {"command": text}, transport=transport, env=env)
    if not got["ok"]:
        return _envelope(devices=catalog, assumptions=assumptions, error=got["error"])
    actions = _actions_from(got["body"])
    if not actions:
        return _envelope(
            devices=catalog,
            assumptions=assumptions,
            error="DeviceWeave accepted the call but returned no device action",
        )
    return _envelope(executed=True, devices=catalog, actions=actions, assumptions=assumptions)


def _verify(actions: List[Dict[str, str]], *, transport: Optional[Transport],
            env: Optional[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[str]]:
    """Re-read live state with a get_status command. Skips an action that was already one."""
    state: List[Dict[str, str]] = []
    errors: List[str] = []
    seen = set()
    for action in actions:
        device = action["device"]
        if action["command"] == "get_status":
            if device not in seen:
                seen.add(device)
                state.append({"device": device, "state": _state_text(action["result"])})
            continue
        if device in seen:
            continue
        seen.add(device)
        got = _request(
            "POST", "/execute", {"command": f"status of {device}"},
            transport=transport, env=env,
        )
        if not got["ok"]:
            errors.append(f"could not re-read status of {device}: {got['error']}")
            fallback = _state_text(action["result"])
            if fallback:
                state.append({"device": device, "state": fallback})
            continue
        verified = _actions_from(got["body"])
        if verified:
            state.append({
                "device": verified[0]["device"],
                "state": _state_text(verified[0]["result"]),
            })
        else:
            errors.append(f"status of {device} returned no state")
    return state, errors


def act(command: str, confirm: Any = None, *,
        transport: Optional[Transport] = None,
        env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Look up the registry, send the command when it is allowed, then re-read state.

    A risky command with no confirm flag returns ``needs_confirmation`` and
    does not call ``POST /execute``. A list failure does not fall through to
    a command: the lookup did not happen, which is different from an empty
    registry.
    """
    listed = list_devices(transport=transport, env=env)
    if listed.get("error"):
        return listed
    devices = listed["devices"]
    sent = send_command(
        command, confirm=confirm, devices=devices, transport=transport, env=env,
    )
    if not sent.get("executed"):
        sent["devices"] = devices
        return sent
    state, verify_errors = _verify(sent["actions_taken"], transport=transport, env=env)
    sent["current_state"] = state
    for message in verify_errors:
        if message not in sent["errors"]:
            sent["errors"].append(message)
    return sent


def run(operation: str = "act", *, command: str = "", confirm: Any = None,
        device_id: str = "", name: str = "",
        transport: Optional[Transport] = None,
        env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Dispatch one DeviceWeave operation: list, status, command, or act."""
    op = (operation or "act").strip().lower()
    if op == "list":
        return list_devices(transport=transport, env=env)
    if op == "status":
        record = get_device(device_id, transport=transport, env=env) if device_id else None
        if record is not None and record.get("error"):
            return record
        label = name
        if record and record.get("devices"):
            label = str(record["devices"][0].get("name") or name or device_id)
        if not label:
            return _envelope(error="device_status requires a device_id or a name")
        # Live state is a get_status command. It is built here, not taken from
        # the user, so it does not go through the confirm gate.
        phrase = f"status of {label}"
        got = _request("POST", "/execute", {"command": phrase}, transport=transport, env=env)
        catalog = (record or {}).get("devices") or []
        if not got["ok"]:
            return _envelope(devices=catalog, error=got["error"])
        actions = _actions_from(got["body"])
        state = [
            {"device": item["device"], "state": _state_text(item["result"])}
            for item in actions
        ]
        if not actions:
            return _envelope(
                devices=catalog,
                error="DeviceWeave accepted the status call but returned no device action",
            )
        return _envelope(executed=True, devices=catalog, actions=actions, state=state)
    if op == "command":
        return send_command(command, confirm=confirm, transport=transport, env=env)
    if op == "act":
        return act(command, confirm=confirm, transport=transport, env=env)
    return _envelope(error=f"unknown DeviceWeave operation {operation!r}")
