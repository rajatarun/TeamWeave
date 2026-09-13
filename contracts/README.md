# Vendored cross-repository contracts

These files are **copies**. None of them is authored here. TeamWeave sits on the
consumer side of every one of them — it writes the shared metrics table, reads it
back in two dashboard handlers, and calls ContextWeave over HTTP — so its test
suite validates itself against the producers' definitions rather than against its
own assumptions about them.

| File | Canonical home repo | What it pins | How TeamWeave is bound to it |
|------|--------------------|--------------|------------------------------|
| `observatory_metrics_item.json` | **mcp-observatory** (`contracts/observatory_metrics_item.json`) | Item shape, pk/sk grammar and namespace registry of the shared `OBSERVATORY_METRICS` DynamoDB table | `src/orchestrator/mcp_observatory.py` writes items; `src/orchestrator/agent_metrics_handler.py` and `src/orchestrator/unified_observability_handler.py` read them |
| `conformance.py` | **mcp-observatory** (`contracts/conformance.py`) | Dependency-free checker (`load_contract`, `check_item`, `readers_for`) every consumer runs against the file above | `tests/test_shared_table_contract.py` |
| `contextweave_http_api.json` | **ContextWeave** (`contracts/contextweave_http_api.json`) | The HTTP surface ContextWeave produces and TeamWeave consumes, with a realistic `sample` response per endpoint | `src/orchestrator/contextweave_client.py`, `src/orchestrator/unified_observability_handler.py`; tests in `tests/test_contextweave_contract.py` and `tests/test_unified_observability_handler.py` |

## Rules for these copies

1. **Do not edit them here.** A local edit makes this repository's tests pass
   against a contract nobody else holds, which is the exact failure the files
   exist to prevent. Change the canonical copy in its home repo first.
2. **Copies move together on a version bump.** Each file carries a `version`
   string. When a producer bumps it, every vendored copy in every consumer
   repository must be re-copied in the same change; a consumer left on 1.0.0
   while the producer ships 1.1.0 is testing against a contract that no longer
   describes production.
3. **A disagreement between a contract and this repo's code is a finding.** If a
   conformance test fails, the answer is to report which side is wrong — not to
   edit the contract until the test goes green.

## What the tests here deliberately do *not* fix

`observatory_metrics_item.json` records namespaces with `status: "unread"` —
`SPAN` (written by the shared `mcp_observatory.aws.DynamoDBSpanExporter`),
`WRAPPER` and `INVOCATION` (toolweave). TeamWeave's dashboards query only
`OBSERVATORY#{operation}`, so rows in those namespaces are written, billed and
never displayed. `tests/test_shared_table_contract.py` pins that gap as a
recorded fact rather than papering over it; resolving it is a platform decision
in mcp-observatory's `docs/integration-audit.md`, not a change a consumer
repository can make alone.
