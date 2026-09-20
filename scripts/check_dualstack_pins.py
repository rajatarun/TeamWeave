#!/usr/bin/env python3
"""Which IPv4 pins are still needed, and which services are missing one.

`AWS_USE_DUALSTACK_ENDPOINT=true` is a blanket instruction: every boto3 client
builds `{service}.{region}.api.aws` unless an `AWS_ENDPOINT_URL_*` variable
overrides it. Where AWS publishes no dual-stack endpoint that hostname does not
exist, and the call dies resolving DNS — inside the VPC, at run time. Not at
deploy, and not in CI, which has no dual-stack setting of its own. That is how
AgentCore, the substrate every agent turn runs on, went unpinned.

So this checks both directions:

  * a service the code calls, with no dual-stack endpoint and no pin, is
    **broken** — it will fail in the VPC and nowhere else;
  * a pin on a service that has since gained a dual-stack endpoint is
    **stale** — removing it moves that traffic onto IPv6 and off NAT.

Existence of `{service}.{region}.api.aws` is the test. AWS publishes that name
only where it serves dual-stack, so a name that does not resolve is a service
that has none. Run it to decide what the template should pin:

    python3 scripts/check_dualstack_pins.py --region us-east-1
"""
from __future__ import annotations

import argparse
import re
import socket
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "infra" / "template.yaml"
SRC = REPO / "src"

# Pinned to IPv4 on purpose rather than for lack of an endpoint: their Gateway
# VPC endpoints are free and private, and the dual-stack hostnames would route
# that traffic out through the Egress-Only Internet Gateway instead.
DELIBERATE_IPV4 = {"s3", "dynamodb"}


def services_the_code_calls() -> Set[str]:
    """Every service a boto3 client is created for, from the source itself."""
    found: Set[str] = set()
    for path in SRC.rglob("*.py"):
        for match in re.finditer(r'boto3\.(?:client|resource)\(\s*["\']([a-z0-9-]+)["\']', path.read_text()):
            found.add(match.group(1))
    return found


def pinned_services() -> Dict[str, str]:
    """serviceId-derived env var -> hostname, as the template sets them."""
    out: Dict[str, str] = {}
    for line in TEMPLATE.read_text().splitlines():
        m = re.match(r'\s*(AWS_ENDPOINT_URL_[A-Z0-9_]+):\s*!Sub\s*"([^"]+)"', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def endpoint_env_var(service: str) -> str:
    """The variable botocore reads, derived from the service's own serviceId.

    Not from the client name: Secrets Manager's serviceId is "Secrets Manager",
    so the variable is AWS_ENDPOINT_URL_SECRETS_MANAGER. Guessing
    SECRETSMANAGER gives a variable nothing reads and a pin that does nothing.
    """
    import boto3

    client = boto3.client(service, region_name="us-east-1",
                          aws_access_key_id="x", aws_secret_access_key="y")
    service_id = str(client.meta.service_model.service_id)
    return "AWS_ENDPOINT_URL_" + service_id.upper().replace(" ", "_").replace("-", "_")


def dualstack_host(service: str, region: str) -> str:
    """The hostname botocore would actually use with dual-stack enabled.

    Not `{client_name}.{region}.api.aws`. The endpoint prefix is often not the
    client name -- Step Functions' client is `stepfunctions` and its endpoint
    is `states` -- and S3 uses `s3.dualstack.{region}.amazonaws.com`, a
    different pattern entirely. Constructing the name by hand reported both as
    having no dual-stack endpoint, which was wrong in both directions.
    """
    import boto3
    from botocore.config import Config
    from urllib.parse import urlparse

    client = boto3.client(service, region_name=region,
                          config=Config(use_dualstack_endpoint=True),
                          aws_access_key_id="x", aws_secret_access_key="y")
    return urlparse(client.meta.endpoint_url).hostname or ""


def has_dualstack_endpoint(service: str, region: str) -> bool:
    """Whether that hostname exists. AWS publishes it only where it serves."""
    host = dualstack_host(service, region)
    if not host:
        return False
    try:
        socket.getaddrinfo(host, 443)
        return True
    except socket.gaierror:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    called = sorted(services_the_code_calls())
    pins = pinned_services()
    broken: List[str] = []
    stale: List[Tuple[str, str]] = []

    print(f"Checking {len(called)} service(s) the code calls, in {args.region}\n")
    for service in called:
        try:
            env = endpoint_env_var(service)
        except Exception as exc:  # noqa: BLE001 - an unknown service is not fatal here
            print(f"  {service:26} could not resolve serviceId ({str(exc)[:60]})")
            continue
        dual = has_dualstack_endpoint(service, args.region)
        pinned = env in pins
        if dual and pinned and service not in DELIBERATE_IPV4:
            stale.append((service, env))
            state = "dual-stack available — PIN IS STALE"
        elif dual and pinned:
            state = "dual-stack available — pinned on purpose (free gateway endpoint)"
        elif dual:
            state = "dual-stack — using IPv6"
        elif pinned:
            state = "no dual-stack — pinned (correct)"
        else:
            broken.append(service)
            state = "no dual-stack and NO PIN — will fail DNS inside the VPC"
        print(f"  {service:26} {state}")

    print()
    if broken:
        print("::error::These services have no dual-stack endpoint and no IPv4 pin, so every "
              f"call fails inside the VPC: {', '.join(broken)}", flush=True)
    if stale:
        print("::notice::These pins can be removed, moving their traffic to IPv6: "
              + ", ".join(f"{s} ({e})" for s, e in stale), flush=True)
    if not broken and not stale:
        print("::notice::Every service is either on IPv6 or pinned for a reason.", flush=True)

    # A stale pin is an opportunity, not a failure. A missing one is a fault.
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
