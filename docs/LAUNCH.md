# Launch materials (draft)

Do not publish from this file without maintainer review. No user counts or
benchmarks are claimed unless measured and labeled.

---

## Draft launch post

**Title:** TeamWeave — multi-agent workflows as JSON on AWS

**Problem:** Teams want reusable AI pipelines (research → draft → review) without
redeploying application code for every new role or step. Ad-hoc scripts are hard
to operate, observe, or hand to non-authors.

**Why it matters:** Config-driven orchestration lets you version workflows like
code, run them asynchronously at production scale, and plug in RAG, images, and
sibling services — while keeping a single HTTP and A2A surface for clients.

**What TeamWeave does:**

- Stores team definitions (`team.json`) in S3; Worker Lambda executes Step Functions pipelines step by step.
- Validates structured outputs against JSON Schema; optional repair via Converse.
- Supports Bedrock AgentCore (default), optional Classic Bedrock Agents, image steps (Bedrock or Gemini), and pre-invocation tools (MCP siblings, ContextWeave, DeviceWeave, etc.).
- Exposes REST (`openapi/teamweave.yaml`) and an A2A Agent Card for discovery.

**Honest limits:**

- **Requires AWS** for the full platform (Lambda, Step Functions, DynamoDB, S3, Bedrock). Local development is unit tests + offline config linting, not a full pipeline without cloud resources.
- Sibling Weave services (ContextWeave, ScreenWeave, …) are optional; many teams degrade gracefully when URLs or tokens are missing.
- MCP Gateway targets prepare AgentCore tool access; agents do not dynamically call gateway tools mid-turn yet — tools declared in config run deterministically before/after turns.
- Authentication is SIWE JWT for mutating routes; the agent card is public, operations are not.

**Try it:** See [README](../README.md) quickstart (tests + `examples/validate_team.py`). Deploy via SAM when you have an AWS account and Bedrock access.

---

## Communities where this fits

| Community | Angle |
|-----------|--------|
| [A2A / Linux Foundation AGENTS](https://github.com/a2aproject/A2A) | Card + async task polling pattern |
| [MCP](https://modelcontextprotocol.io/) | Pre-tool sibling calls; future MCP server facade |
| AWS Bedrock / AgentCore builders | Reference architecture for multi-step agents |
| OpenTelemetry GenAI semconv | Observability-minded agent pipelines |
| JSON Schema / structured-output practitioners | Pipeline contracts between steps |

---

## Synthetic evaluation (offline)

The repo includes **offline config linting**, not LLM quality benchmarks:

```bash
python examples/validate_team.py --all-examples
pytest tests/test_workflow_integrity.py tests/test_example_team_lint.py -q
```

On a representative run (synthetic, CI-free): **7 teams** linted clean; pytest
workflow integrity tests pass for all example teams. No production latency or
cost numbers are reported here.
