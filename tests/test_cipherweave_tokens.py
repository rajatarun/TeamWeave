"""CipherWeave tokens on the calls TeamWeave makes to its siblings.

The siblings cannot switch their auth from audit to enforce while this worker
sends nothing, so these hold both halves: a configured sibling gets a token on
every call, and an unconfigured, unfilled or failing one gets the call it got
before -- a token outage must not take runs down before anyone requires one.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import pathlib
import urllib.error

import pytest
import yaml

from src.orchestrator import cipherweave_tokens as cwt
from src.orchestrator import deviceweave, mcp_client

REPO = pathlib.Path(__file__).resolve().parents[1]
TOKEN_URL = "https://issuer.test/prod/token"
PREFIX = "teamweave/cipherweave-client/"
ENV = {cwt.TOKEN_URL_ENV: TOKEN_URL, cwt.SECRET_PREFIX_ENV: PREFIX}
SECRET = "s3cr3t-value-never-logged"


@pytest.fixture(autouse=True)
def fresh():
    cwt.reset()
    yield
    cwt.reset()


class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *_e): self.close(); return False


class Issuer:
    """The token endpoint as the issuer answers it: one secret per client id."""

    def __init__(self, secrets=None, expires_in=900, token_type="Bearer", fail=None):
        self.secrets = secrets if secrets is not None else {"teamweave-screenweave": SECRET}
        self.expires_in = expires_in
        self.token_type = token_type
        self.fail = fail
        self.requests = []

    def __call__(self, request, timeout=None):
        body = json.loads(request.data)
        basic = request.get_header("Authorization").split(" ", 1)[1]
        client_id, secret = base64.b64decode(basic).decode().split(":", 1)
        # Recorded before anything can raise: the code under test swallows
        # errors, so a refusal here must still be visible to the assertion.
        self.requests.append({"client_id": client_id, "body": body, "url": request.full_url})
        if request.full_url != TOKEN_URL:
            raise AssertionError(f"sent to {request.full_url}")
        if self.fail:
            raise self.fail
        if self.secrets.get(client_id) != secret:
            raise urllib.error.HTTPError(TOKEN_URL, 401, "Unauthorized", {},
                                         io.BytesIO(b'{"error": "invalid_client"}'))
        token = f"tok-{len(self.requests)}-{body['audience']}"
        return Response(json.dumps({"access_token": token, "token_type": self.token_type,
                                    "expires_in": self.expires_in}).encode())


class Clock:
    def __init__(self): self.now = 1_000_000.0
    def __call__(self): return self.now


def auth(issuer, secret=SECRET, env=ENV, audience="screenweave", clock=None, reads=None):
    def reader(arn):
        if reads is not None:
            reads.append(arn)
        return secret() if callable(secret) else secret
    return cwt.authorization(audience, env, opener=issuer, secret_reader=reader,
                             clock=clock or Clock())


# ── configured: a token per sibling ─────────────────────────────────────────

def test_a_raw_secret_mints_for_its_own_client_and_audience():
    issuer = Issuer()
    assert auth(issuer) == {"Authorization": "Bearer tok-1-screenweave"}
    assert issuer.requests == [{"client_id": "teamweave-screenweave", "url": TOKEN_URL,
                                "body": {"grant_type": "client_credentials",
                                         "audience": "screenweave"}}]


def test_a_json_secret_can_name_its_client():
    issuer = Issuer(secrets={"tw-shots": "x"})
    got = auth(issuer, secret=json.dumps({"client_id": "tw-shots", "client_secret": "x"}))
    assert got["Authorization"].startswith("Bearer tok-")
    assert issuer.requests[0]["client_id"] == "tw-shots"


def test_the_token_is_cached_until_shortly_before_it_expires():
    issuer, clock = Issuer(expires_in=900), Clock()
    assert auth(issuer, clock=clock) == auth(issuer, clock=clock)
    assert len(issuer.requests) == 1
    clock.now += 900 - cwt.REFRESH_MARGIN_S + 1
    assert auth(issuer, clock=clock) == {"Authorization": "Bearer tok-2-screenweave"}


def test_a_token_the_sibling_refused_is_not_resent():
    issuer = Issuer()
    auth(issuer)
    cwt.forget("screenweave")
    assert auth(issuer) == {"Authorization": "Bearer tok-2-screenweave"}


def test_a_rotated_secret_is_read_again_once_on_invalid_client():
    issuer, clock = Issuer(), Clock()
    current = {"value": SECRET}
    reads = []
    auth(issuer, secret=lambda: current["value"], clock=clock, reads=reads)
    # rotate-secret with no grace: the old secret is refused at once.
    issuer.secrets["teamweave-screenweave"] = current["value"] = "rotated"
    cwt.forget("screenweave")
    got = auth(issuer, secret=lambda: current["value"], clock=clock, reads=reads)
    assert got == {"Authorization": "Bearer tok-3-screenweave"}
    assert len(reads) == 2, "cached, then one fresh read on invalid_client"


def test_the_secret_is_re_read_after_its_ttl():
    issuer, clock = Issuer(expires_in=10_000), Clock()
    reads = []
    auth(issuer, clock=clock, reads=reads)
    cwt.forget("screenweave")
    clock.now += cwt.SECRET_TTL_S + 1
    auth(issuer, clock=clock, reads=reads)
    assert len(reads) == 2


# ── not configured, or failing: the call goes out as before ─────────────────

@pytest.mark.parametrize("env", [
    {}, {cwt.TOKEN_URL_ENV: TOKEN_URL}, {cwt.SECRET_PREFIX_ENV: PREFIX},
], ids=["nothing", "no-secret", "no-issuer"])
def test_unconfigured_sends_nothing_and_asks_nobody(env):
    issuer = Issuer()
    assert auth(issuer, env=env) == {}
    assert issuer.requests == []


@pytest.mark.parametrize("value", ["", "{}", "  ", '{"client_id": "x"}', "[1]"])
def test_an_unfilled_secret_sends_nothing(value):
    issuer = Issuer()
    assert auth(issuer, secret=value) == {}
    assert issuer.requests == []


def test_a_sibling_the_issuer_does_not_know_gets_nothing():
    env = ENV
    issuer = Issuer()
    assert auth(issuer, env=env, audience="datadictionary") == {}
    assert issuer.requests == []


def test_a_plain_http_issuer_is_never_sent_the_secret():
    issuer = Issuer()
    assert auth(issuer, env={**ENV, cwt.TOKEN_URL_ENV: "http://issuer.test/token"}) == {}
    assert issuer.requests == []


@pytest.mark.parametrize("fail", [
    urllib.error.URLError("timed out"),
    urllib.error.HTTPError(TOKEN_URL, 503, "Unavailable", {}, io.BytesIO(b"oops")),
    urllib.error.HTTPError(TOKEN_URL, 400, "Bad", {}, io.BytesIO(b'{"error": "invalid_target"}')),
], ids=["network", "issuer-down", "audience-refused"])
def test_an_issuer_failure_degrades_and_never_logs_the_secret(fail, caplog):
    caplog.set_level(logging.DEBUG)
    assert auth(Issuer(fail=fail)) == {}
    assert SECRET not in caplog.text
    assert "cipherweave_token_" in caplog.text, "a failure is logged, not silent"


def test_a_wrong_secret_is_refused_once_not_in_a_loop(caplog):
    issuer = Issuer(secrets={"teamweave-screenweave": "something-else"})
    assert auth(issuer) == {}
    assert len(issuer.requests) == 1, "the same secret read again is not retried"
    assert "invalid_client" in caplog.text and SECRET not in caplog.text


def test_a_dpop_bound_token_is_not_sent_as_bearer():
    assert auth(Issuer(token_type="DPoP")) == {}


def test_a_failing_secrets_manager_degrades():
    def boom(_arn):
        raise RuntimeError("AccessDenied")
    assert cwt.authorization("screenweave", ENV, opener=Issuer(), secret_reader=boom) == {}


# ── the two call paths ──────────────────────────────────────────────────────

HELLO = json.dumps({"jsonrpc": "2.0", "id": 1,
                    "result": {"protocolVersion": mcp_client.PROTOCOL_VERSION}})
RESULT = json.dumps({"jsonrpc": "2.0", "id": 2,
                     "result": {"content": [{"type": "text", "text": '{"ok": true}'}]}})
MCP_ENV = {**ENV, "SCREENWEAVE_MCP_URL": "https://screenweave.test/mcp"}


def _with_token(monkeypatch, issuer):
    real = cwt.authorization
    monkeypatch.setattr(cwt, "authorization", lambda audience, env=None: real(
        audience, env, opener=issuer, secret_reader=lambda _arn: SECRET))


def test_mcp_calls_carry_the_token_on_both_requests(monkeypatch):
    _with_token(monkeypatch, Issuer())
    sent = []
    bodies = [HELLO, RESULT]

    def sibling(request, timeout=None):
        sent.append(request.get_header("Authorization"))
        return Response(bodies.pop(0).encode())

    out = mcp_client.call_tool("screenweave", "capture", {}, env=MCP_ENV, opener=sibling)
    assert out["data"] == {"ok": True}
    assert sent == ["Bearer tok-1-screenweave"] * 2


def test_mcp_calls_without_configuration_send_no_authorization():
    sent = []
    bodies = [HELLO, RESULT]

    def sibling(request, timeout=None):
        sent.append(request.get_header("Authorization"))
        return Response(bodies.pop(0).encode())

    env = {"SCREENWEAVE_MCP_URL": "https://screenweave.test/mcp"}
    assert "error" not in mcp_client.call_tool("screenweave", "capture", {}, env=env, opener=sibling)
    assert sent == [None, None]


def test_a_sibling_401_drops_the_cached_token(monkeypatch):
    issuer = Issuer()
    _with_token(monkeypatch, issuer)

    def refuse(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b""))

    out = mcp_client.call_tool("screenweave", "capture", {}, env=MCP_ENV, opener=refuse)
    assert out["error"] == "HTTP 401 from screenweave"
    cwt.authorization("screenweave", MCP_ENV)
    assert len(issuer.requests) == 2, "minted again, not reused"


DW_ENV = {cwt.TOKEN_URL_ENV: TOKEN_URL, cwt.SECRET_PREFIX_ENV: PREFIX,
          "DEVICEWEAVE_SECRET_ARN": "arn:static"}


def test_deviceweave_prefers_the_cipherweave_token_over_the_static_secret():
    built = deviceweave.headers(DW_ENV, secret_reader=lambda _a: "static-token",
                                token_auth=lambda aud, env: {"Authorization": f"Bearer cw-{aud}"})
    assert built["Authorization"] == "Bearer cw-deviceweave"


def test_deviceweave_falls_back_to_the_static_secret_when_no_token():
    built = deviceweave.headers(DW_ENV, secret_reader=lambda _a: "static-token",
                                token_auth=lambda aud, env: {})
    assert built["Authorization"] == "Bearer static-token"


def test_deviceweave_asks_for_its_own_audience():
    issuer = Issuer(secrets={"teamweave-deviceweave": SECRET})
    env = {k: v for k, v in DW_ENV.items() if k != "DEVICEWEAVE_SECRET_ARN"}
    built = deviceweave.headers(env, token_auth=lambda aud, e: cwt.authorization(
        aud, e, opener=issuer, secret_reader=lambda _a: SECRET))
    assert built["Authorization"] == "Bearer tok-1-deviceweave"


# ── the deployment: one secret per sibling, readable by the worker alone ────

class _Loader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    name = "Ref" if suffix == "Ref" else f"Fn::{suffix}"
    return {name: value}


_Loader.add_multi_constructor("!", _tag)
TEMPLATE = yaml.load((REPO / "infra" / "template.yaml").read_text(), Loader=_Loader)
RESOURCES = TEMPLATE["Resources"]
SECRETS = {aud: f"CipherWeaveClientSecret{aud.capitalize()}" for aud in cwt.AUDIENCES}


def test_each_sibling_has_its_own_unfilled_secret():
    for audience, logical in SECRETS.items():
        secret = RESOURCES[logical]
        assert secret["Type"] == "AWS::SecretsManager::Secret"
        # Unfilled means "not configured": runs keep working until filled.
        assert cwt.parse_credential(secret["Properties"]["SecretString"], audience) is None


def test_the_worker_is_given_the_issuer_and_the_secret_prefix():
    env = RESOURCES["WorkerFunction"]["Properties"]["Environment"]["Variables"]
    assert env[cwt.TOKEN_URL_ENV] == {"Ref": "CipherWeaveTokenEndpoint"}
    assert env[cwt.SECRET_PREFIX_ENV] == {"Fn::Sub": "${AWS::StackName}/cipherweave-client/"}


def test_each_secret_is_named_prefix_plus_sibling():
    """The worker builds the name from the prefix, so a rename here silently unconfigures it."""
    for audience, logical in SECRETS.items():
        name = RESOURCES[logical]["Properties"]["Name"]
        assert name == {"Fn::Sub": "${AWS::StackName}/cipherweave-client/" + audience}


def _statements(role):
    for policy in role["Properties"].get("Policies", []):
        for statement in policy["PolicyDocument"]["Statement"]:
            yield statement


def test_only_the_worker_can_read_the_client_secrets():
    wanted = [{"Ref": logical} for logical in SECRETS.values()]
    worker = [s for s in _statements(RESOURCES["WorkerRole"])
              if isinstance(s, dict) and s.get("Resource") == wanted]
    assert worker and worker[0]["Action"] == ["secretsmanager:GetSecretValue"]
    for name, resource in RESOURCES.items():
        if resource["Type"] != "AWS::IAM::Role" or name == "WorkerRole":
            continue
        text = json.dumps(resource)
        for logical in SECRETS.values():
            assert logical not in text, f"{name} can read {logical}"


def test_the_deploy_reads_the_token_endpoint_from_the_issuer_stack():
    workflow = (REPO / ".github" / "workflows" / "deploy.yml").read_text()
    assert "IssuerTokenEndpoint" in workflow
    assert "CipherWeaveTokenEndpoint=" in workflow
