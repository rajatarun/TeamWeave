"""CipherWeave access tokens for the calls TeamWeave makes to its siblings.

ScreenWeave, ToolWeave, CipherWeave's own MCP server and DeviceWeave all
verify a CipherWeave token (an ES256 JWT, `aud` = the sibling) and, until they
are switched to enforce, also take a call that carries none. They cannot be
switched while their one regular caller -- this worker -- sends nothing, so
this module is what unblocks that switch.

**One client per sibling.** The issuer's `audit-clients` treats a standing
client with more than one audience as a master key, so TeamWeave holds four:
`teamweave-<sibling>`, each in its own Secrets Manager secret
(named `<CIPHERWEAVE_SECRET_PREFIX><sibling>`). That is also the shape `issuer_admin.py
rotate-secret --secret-arn` writes, so a secret rotated there, or by
`rotate-due`, is picked up here within `SECRET_TTL_S` with no deploy.

**The secret's value** is what `rotate-secret` writes: the raw client secret.
A JSON object `{"client_id": ..., "client_secret": ...}` is accepted too, for
a client named something other than `teamweave-<sibling>`. The stack creates
each secret as `{}`, which means *not configured yet*.

**Never raises, never blocks a run.** No issuer URL, no secret, an unfilled
secret, or an issuer that refuses all return no header, and the call goes out
as it does today. While the sibling is in audit mode that call still works;
once it enforces, the sibling's 401 is the error the step reports. A token
outage must not take the platform down before the siblings even require one.
Every outcome other than "not configured" is logged, without the secret or
the token.

**Tokens are cached** per sibling until 60 s before expiry. A 401 from the
sibling drops that cached token (`forget`), and an `invalid_client` from the
issuer re-reads the secret once, which is how a rotation without a grace
window is survived.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, Optional, Tuple

from .logger import get_logger

log = get_logger("cipherweave_tokens")

# Siblings that verify CipherWeave tokens. DataDictionary does not: it is not
# an audience the issuer knows, so asking would be refused on every call.
AUDIENCES = ("screenweave", "toolweave", "cipherweave", "deviceweave")
TOKEN_URL_ENV = "CIPHERWEAVE_TOKEN_URL"
SECRET_PREFIX_ENV = "CIPHERWEAVE_SECRET_PREFIX"
CLIENT_PREFIX = "teamweave-"
SECRET_TTL_S = 300.0
REFRESH_MARGIN_S = 60.0
TIMEOUT_S = 5.0

_lock = threading.Lock()
_tokens: Dict[str, Tuple[str, float]] = {}
_secrets: Dict[str, Tuple[Optional[Tuple[str, str]], float]] = {}


def secret_id(audience: str, env: Dict[str, str]) -> str:
    """Secrets Manager name of `audience`'s client secret, or "" if unconfigured.

    One prefix variable instead of one ARN per sibling: a Lambda's environment
    is capped at 4 KB in total, and four ARNs (about 570 bytes) took the
    worker from under the cap to 4472 bytes, which failed the deploy.
    """
    prefix = str(env.get(SECRET_PREFIX_ENV) or "").strip()
    return prefix + audience if prefix else ""


def _read_secret(arn: str) -> str:
    import boto3

    return boto3.client("secretsmanager").get_secret_value(SecretId=arn).get("SecretString") or ""


def parse_credential(raw: str, audience: str) -> Optional[Tuple[str, str]]:
    """(client_id, client_secret), or None for an unfilled secret."""
    raw = (raw or "").strip()
    if not raw:
        return None
    # A raw secret (secrets.token_urlsafe) never starts with either.
    if raw[0] in "{[":
        try:
            parsed = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(parsed, dict):
            return None
        secret = str(parsed.get("client_secret") or "").strip()
        if not secret:
            return None
        return str(parsed.get("client_id") or CLIENT_PREFIX + audience).strip(), secret
    return CLIENT_PREFIX + audience, raw


def _credential(arn: str, audience: str, reader: Callable[[str], str],
                clock: Callable[[], float], *, reread: bool = False) -> Optional[Tuple[str, str]]:
    with _lock:
        cached = _secrets.get(arn)
    if cached and not reread and clock() - cached[1] < SECRET_TTL_S:
        return cached[0]
    credential = parse_credential(reader(arn), audience)
    with _lock:
        _secrets[arn] = (credential, clock())
    return credential


class _Refused(Exception):
    def __init__(self, status: int, error: str) -> None:
        super().__init__(f"{status} {error}")
        self.status = status
        self.error = error


def _mint(token_url: str, credential: Tuple[str, str], audience: str, opener,
          clock: Callable[[], float]) -> Tuple[str, float]:
    client_id, client_secret = credential
    pair = f"{urllib.parse.quote_plus(client_id)}:{urllib.parse.quote_plus(client_secret)}"
    request = urllib.request.Request(
        token_url,
        data=json.dumps({"grant_type": "client_credentials", "audience": audience}).encode(),
        headers={"Authorization": "Basic " + base64.b64encode(pair.encode()).decode(),
                 "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=TIMEOUT_S) as response:
            body = json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            error = str(json.loads(exc.read() or b"{}").get("error") or "")
        except ValueError:
            error = ""
        raise _Refused(exc.code, error or "refused") from None
    access = body.get("access_token") if isinstance(body, dict) else None
    if not access:
        raise _Refused(200, "no access_token in the response")
    # A DPoP-bound token sent as Bearer is refused (`dpop_required`), so one
    # is not used rather than sent wrong. These clients are not DPoP clients.
    if str(body.get("token_type") or "Bearer").lower() != "bearer":
        raise _Refused(200, f"token_type {body.get('token_type')} is not supported here")
    return access, clock() + float(body.get("expires_in") or 0)


def authorization(audience: str, env: Optional[Dict[str, str]] = None, *,
                  opener=None, secret_reader: Optional[Callable[[str], str]] = None,
                  clock: Callable[[], float] = time.time) -> Dict[str, str]:
    """`{"Authorization": "Bearer <token>"}` for one call to `audience`, or `{}`."""
    source = os.environ if env is None else env
    if audience not in AUDIENCES:
        return {}
    token_url = str(source.get(TOKEN_URL_ENV) or "").strip()
    arn = secret_id(audience, source)
    if not token_url or not arn:
        return {}
    if not token_url.startswith("https://"):
        # A client secret sent over anything else is handed to whoever listens.
        log.warning("cipherweave_token_skipped audience=%s reason=token_url_not_https", audience)
        return {}

    with _lock:
        cached = _tokens.get(audience)
    if cached and cached[1] - clock() > REFRESH_MARGIN_S:
        return {"Authorization": f"Bearer {cached[0]}"}

    if opener is None:
        opener = urllib.request.urlopen
    reader = secret_reader or _read_secret
    try:
        credential = _credential(arn, audience, reader, clock)
        if credential is None:
            return {}
        try:
            token = _mint(token_url, credential, audience, opener, clock)
        except _Refused as exc:
            if exc.error != "invalid_client":
                raise
            # The secret may have been rotated with no grace window since it
            # was cached. One fresh read, then the refusal stands.
            fresh = _credential(arn, audience, reader, clock, reread=True)
            if fresh is None or fresh == credential:
                raise
            token = _mint(token_url, fresh, audience, opener, clock)
    except _Refused as exc:
        log.warning("cipherweave_token_refused audience=%s status=%s error=%s",
                    audience, exc.status, exc.error)
        return {}
    except Exception as exc:
        log.warning("cipherweave_token_unavailable audience=%s err=%s",
                    audience, type(exc).__name__)
        return {}

    with _lock:
        _tokens[audience] = token
    return {"Authorization": f"Bearer {token[0]}"}


def forget(audience: str) -> None:
    """Drop the cached token: the sibling refused it (revoked, or a key rotated)."""
    with _lock:
        _tokens.pop(audience, None)


def reset() -> None:
    with _lock:
        _tokens.clear()
        _secrets.clear()
