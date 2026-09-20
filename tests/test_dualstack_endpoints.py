"""IPv6 where AWS serves it, IPv4 pins where it does not — and nothing unpinned.

`AWS_USE_DUALSTACK_ENDPOINT=true` is a blanket instruction: every boto3 client
builds a dual-stack hostname unless an `AWS_ENDPOINT_URL_*` variable overrides
it. Where AWS publishes no dual-stack endpoint that hostname does not exist and
the call dies resolving DNS — **inside the VPC, at run time**. Not at deploy,
not in CI, because the runner has no dual-stack setting of its own.

That is how `bedrock-agentcore` went unpinned: it is the substrate every agent
turn runs on, its dual-stack hostname does not resolve, the smoke test passes
because it runs on the GitHub runner, and nothing else would have noticed until
a real pipeline step failed. Secrets Manager was the same, and worse: `rag.py`
swallows the failure and degrades to no RAG context, so the run just gets
quietly worse.

These tests do not query DNS — `scripts/check_dualstack_pins.py` does that, and
a test that depends on the network is a test that fails for reasons unrelated
to the change. What they pin is the part that is knowable offline and was
actually got wrong: the variable names, and that no service is left to chance.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import boto3
import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO / "infra" / "template.yaml").read_text()
SRC = REPO / "src"

# Pinned to IPv4 despite having a dual-stack endpoint: their Gateway VPC
# endpoints are free and private, and a dual-stack hostname would route that
# traffic out through the Egress-Only Internet Gateway instead. IPv6 is the
# goal where it replaces NAT, not where it replaces a private path.
DELIBERATE_IPV4 = {"s3", "dynamodb"}

# Services whose dual-stack hostname does not resolve today. Kept as a list so
# a removal is a deliberate edit, and cross-checked against the template.
NO_DUALSTACK = {
    "bedrock", "bedrock-runtime", "bedrock-agent", "bedrock-agent-runtime",
    "bedrock-agentcore", "bedrock-agentcore-control", "secretsmanager",
}


def endpoint_env_var(service: str) -> str:
    client = boto3.client(service, region_name="us-east-1",
                          aws_access_key_id="x", aws_secret_access_key="y")
    service_id = str(client.meta.service_model.service_id)
    return "AWS_ENDPOINT_URL_" + service_id.upper().replace(" ", "_").replace("-", "_")


def services_called() -> set[str]:
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        found.update(re.findall(r'boto3\.(?:client|resource)\(\s*["\']([a-z0-9-]+)["\']',
                                path.read_text()))
    return found


def pins() -> set[str]:
    return set(re.findall(r"^\s*(AWS_ENDPOINT_URL_[A-Z0-9_]+):", TEMPLATE, re.M))


def test_the_blanket_dualstack_setting_is_on():
    # Without it nothing uses IPv6 and every pin below is pointless.
    assert re.search(r'AWS_USE_DUALSTACK_ENDPOINT:\s*"true"', TEMPLATE)


@pytest.mark.parametrize("service", sorted(NO_DUALSTACK | DELIBERATE_IPV4))
def test_every_service_without_ipv6_is_pinned(service):
    assert endpoint_env_var(service) in pins(), (
        f"{service} has no usable dual-stack endpoint, so the blanket setting "
        f"builds a hostname that does not resolve and every call fails in the VPC"
    )


def test_the_substrate_itself_is_pinned():
    # The one that was missing. Every agent turn goes through it.
    assert "AWS_ENDPOINT_URL_BEDROCK_AGENTCORE:" in TEMPLATE


@pytest.mark.parametrize("service,expected", [
    ("secretsmanager", "AWS_ENDPOINT_URL_SECRETS_MANAGER"),
    ("bedrock-agentcore", "AWS_ENDPOINT_URL_BEDROCK_AGENTCORE"),
    ("bedrock-agent-runtime", "AWS_ENDPOINT_URL_BEDROCK_AGENT_RUNTIME"),
    ("stepfunctions", "AWS_ENDPOINT_URL_SFN"),
])
def test_the_variable_name_comes_from_the_service_id(service, expected):
    """botocore derives it from serviceId, not from the client name.

    Secrets Manager's serviceId is "Secrets Manager", so the variable is
    SECRETS_MANAGER. SECRETSMANAGER is a variable nothing reads and a pin that
    silently does nothing — which looks identical to a pin that works.
    """
    assert endpoint_env_var(service) == expected


@pytest.mark.parametrize("var", sorted(pins()))
def test_no_pin_points_at_a_dualstack_hostname(var):
    # A pin exists to force IPv4. One naming api.aws re-introduces exactly the
    # hostname it was added to avoid.
    line = next(l for l in TEMPLATE.splitlines() if l.strip().startswith(f"{var}:"))
    assert "api.aws" not in line
    assert "amazonaws.com" in line


@pytest.mark.parametrize("var", sorted(pins()))
def test_every_pin_names_a_real_service(var):
    # A typo'd variable is read by nothing and pins nothing, while looking
    # exactly like a pin that works.
    known = {endpoint_env_var(s) for s in services_called() | NO_DUALSTACK | DELIBERATE_IPV4}
    assert var in known, f"{var} matches no service this code calls"


def test_no_service_the_code_calls_is_left_unpinned_and_unknown():
    """Every client is either known to have IPv6, or pinned.

    This is the check that would have caught bedrock-agentcore: a new
    boto3.client(...) for a service with no dual-stack endpoint is broken in
    the VPC and nowhere else, so nothing else would report it.
    """
    known_dualstack = {"stepfunctions", "lambda", "logs", "sts", "xray", "events"}
    unaccounted = services_called() - NO_DUALSTACK - DELIBERATE_IPV4 - known_dualstack
    assert not unaccounted, (
        f"these services are called but neither known-dual-stack nor pinned: "
        f"{sorted(unaccounted)} — run scripts/check_dualstack_pins.py"
    )


# ── The audit script itself ───────────────────────────────────────────────────
# It is the thing that decides what the template should pin, so grepping it for
# a few strings is not enough: the two bugs it was written to avoid — deriving
# the variable from the client name, and building the dual-stack hostname by
# hand — are both invisible to a grep. These drive it instead, with DNS stubbed
# so the classification is tested and the network is not.


def load_audit():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_dualstack_pins", REPO / "scripts" / "check_dualstack_pins.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_audit(monkeypatch, capsys, *, called, pinned, dualstack):
    """Run main() against a fabricated world. Returns (exit_code, output)."""
    audit = load_audit()
    monkeypatch.setattr(audit, "services_the_code_calls", lambda: set(called))
    monkeypatch.setattr(audit, "pinned_services", lambda: {v: "https://x" for v in pinned})
    monkeypatch.setattr(audit, "has_dualstack_endpoint",
                        lambda service, region: service in dualstack)
    monkeypatch.setattr("sys.argv", ["check_dualstack_pins.py", "--region", "us-east-1"])
    code = audit.main()
    return code, capsys.readouterr().out


def test_audit_fails_on_a_service_with_no_dualstack_and_no_pin(monkeypatch, capsys):
    # The bedrock-agentcore case exactly: reachable in CI, dead in the VPC.
    code, out = run_audit(monkeypatch, capsys,
                          called={"bedrock-agentcore"}, pinned=set(), dualstack=set())
    assert code == 1
    assert "NO PIN" in out


def test_audit_passes_once_that_service_is_pinned(monkeypatch, capsys):
    code, out = run_audit(monkeypatch, capsys,
                          called={"bedrock-agentcore"},
                          pinned={"AWS_ENDPOINT_URL_BEDROCK_AGENTCORE"},
                          dualstack=set())
    assert code == 0
    assert "NO PIN" not in out


def test_audit_calls_a_stale_pin_an_opportunity_not_a_failure(monkeypatch, capsys):
    # A service that has since gained IPv6: worth un-pinning, but the build
    # must not go red over it or the check gets ignored.
    code, out = run_audit(monkeypatch, capsys,
                          called={"lambda"},
                          pinned={"AWS_ENDPOINT_URL_LAMBDA"},
                          dualstack={"lambda"})
    assert code == 0
    assert "PIN IS STALE" in out                      # the per-service line
    assert "can be removed" in out                    # and the summary notice
    assert "AWS_ENDPOINT_URL_LAMBDA" in out           # naming the variable to delete


def test_audit_does_not_call_a_deliberate_ipv4_pin_stale(monkeypatch, capsys):
    # S3 and DynamoDB are pinned despite having IPv6, to keep their traffic on
    # the free Gateway endpoint. Reporting those as removable every run would
    # train everyone to ignore the notice.
    code, out = run_audit(monkeypatch, capsys,
                          called={"s3", "dynamodb"},
                          pinned={"AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL_DYNAMODB"},
                          dualstack={"s3", "dynamodb"})
    assert code == 0
    assert "STALE" not in out
    assert "on purpose" in out


def test_audit_scans_the_whole_source_tree_not_just_its_top_level():
    """Every real client lives in `src/orchestrator/`, one level down.

    A non-recursive scan finds nothing, and an audit that finds no clients
    reports no problems — the most dangerous way for this check to break,
    because it goes green.
    """
    audit = load_audit()
    found = audit.services_the_code_calls()
    assert "bedrock-agentcore" in found                # src/orchestrator/agent_runtime.py
    assert "secretsmanager" in found                   # src/orchestrator/rag.py
    assert len(found) >= 5, f"suspiciously few clients found: {sorted(found)}"


def test_audit_reads_the_pins_out_of_the_template():
    # The other half of the same failure: a pin parser that matches nothing
    # reports every service as unpinned, or (worse) a scan that matches
    # nothing reports every pin as fine.
    audit = load_audit()
    assert "AWS_ENDPOINT_URL_BEDROCK_AGENTCORE" in audit.pinned_services()


def test_audit_derives_the_variable_from_the_service_id(monkeypatch, capsys):
    # Not from the client name. "secretsmanager" would give SECRETSMANAGER,
    # a variable botocore never reads, and the audit would then bless a pin
    # that does nothing.
    audit = load_audit()
    assert audit.endpoint_env_var("secretsmanager") == "AWS_ENDPOINT_URL_SECRETS_MANAGER"
    assert audit.endpoint_env_var("stepfunctions") == "AWS_ENDPOINT_URL_SFN"


def test_audit_asks_botocore_for_the_dualstack_hostname():
    """Constructing `{client}.{region}.api.aws` by hand is wrong twice over.

    Step Functions' endpoint prefix is `states`, not its client name, and S3
    uses a `s3.dualstack.{region}.amazonaws.com` shape entirely of its own.
    Building the name by hand reported both as having no dual-stack endpoint.
    """
    audit = load_audit()
    assert audit.dualstack_host("stepfunctions", "us-east-1").startswith("states.")
    assert "dualstack" in audit.dualstack_host("s3", "us-east-1")
