# TeamWeave — Claude Code Guide

## Project Overview

TeamWeave is a **config-driven, serverless multi-agent orchestration platform** on AWS. It lets teams define AI-powered workflows as JSON configs (no code redeploy needed), orchestrating Amazon Bedrock agents through Step Functions pipelines.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Python 3.12 (AWS Lambda) |
| AI/ML | Amazon Bedrock (Nova, Claude Haiku), Google Gemini 3.8 Flash |
| Orchestration | AWS Step Functions (Standard Workflows) |
| API | AWS API Gateway (REST) |
| State | DynamoDB (run/task metadata) |
| Vector DB | RDS PostgreSQL 15 + pgvector (RAG) |
| Storage | S3 (team configs, artifacts) |
| Secrets | AWS Secrets Manager |
| Observability | Amazon Managed Prometheus (AMP), CloudWatch, X-Ray |
| IaC | AWS SAM + CloudFormation |
| CI/CD | GitHub Actions (OIDC) |

---

## Repository Layout

```
src/orchestrator/     # Core Lambda handlers and business logic
  trigger_handler.py  # API entry point → starts Step Functions
  worker_handler.py   # Pipeline engine → executes workflow steps
  status_handler.py   # Polls Step Functions DescribeExecution
  config_loader.py    # Loads team.json from S3
  bedrock_invoke.py   # One agent turn: retries, gate, StepFailed contract
  agent_runtime.py    # Which substrate runs it (classic | agentcore)
  rag.py              # RAG mode dispatch (contextweave + explicit + history)
  contextweave_client.py  # ContextWeave knowledge-layer HTTP client
  db.py               # DynamoDB DAO
  gemini.py           # Gemini API integration
  amp_metrics.py      # Amazon Managed Prometheus telemetry
  requirements.txt    # Python dependencies

config/examples/      # Example team configurations and JSON schemas
  teams/              # Per-team workflow definitions (team.json)
  schemas/            # JSON Schema files for structured output validation

infra/                # Infrastructure as Code
  template.yaml       # Main SAM CloudFormation template
  bedrock-agents.yaml # Bedrock agent provisioning (11 agents)
  samconfig.toml      # SAM deployment parameters

tests/                # pytest unit tests
openapi/teamweave.yaml  # OpenAPI 3.0 description of the HTTP surface
scripts/stack_env.py    # Resolve API/table/index/bucket coordinates from stack outputs
docs/                 # Architecture docs and C4 diagrams
.github/workflows/    # GitHub Actions CI/CD pipeline
Makefile              # Lambda packaging targets
```

---

## Common Commands

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Tests

```bash
pytest
pytest tests/test_worker_handler.py   # single file
pytest -v                              # verbose
```

### Package Lambdas

```bash
make build-TriggerFunction
make build-WorkerFunction
make build-StatusFunction
# or build all:
make package-lambda
```

### Deploy

```bash
sam build -t infra/template.yaml
sam deploy --guided -t infra/template.yaml
```

---

## Architecture

```
Client
  └─► API Gateway
        ├─► Trigger Lambda  ──► Step Functions ──► Worker Lambda ──► Bedrock Agents
        ├─► Status Lambda   (polls Step Functions)                       │
        ├─► Provision Lambda (CRUD for agents/teams)                    ▼
        └─► Gemini Lambda   (external research)              DynamoDB / S3 / pgvector
```

**Two production workflow patterns:**
- **Visibility Team** — content marketing pipeline: strategy → draft → edit → distribute
- **Improvement Team** — personal learning loop: coach → plan → daily tasks

---

## Key Concepts

### Team Config (`team.json`)
All workflow logic lives in JSON stored in S3. Agents, pipeline steps, RAG settings, and output schemas are all defined there. No Lambda code changes needed for new workflows.

### RAG Modes
Selected per team by `globals.rag.mode` in `team.json`. All modes emit the same
`RAG_CONTEXT` string and degrade to an empty context (never an error) when
their backing store is unreachable.

- **ContextWeave mode** (`contextweave`) — `POST {CONTEXTWEAVE_URL}/query-expertise`
  on the ContextWeave knowledge layer. Options: `top_k`, `min_confidence`.
  **Recommended for new teams:** ContextWeave owns the expertise graph, the
  chunk store and a router that learns from feedback, so the platform gets one
  knowledge layer that improves with use instead of three services each keeping
  a static copy of the same corpus. The run's `queryId` is persisted in the step
  record (`inputs_json.rag_meta`), and a schema-valid run can up-vote the answer
  via `POST /feedback` when `CONTEXTWEAVE_FEEDBACK_ON_VALID_OUTPUT=1`.
- **Explicit mode** (`explicit`) — pgvector similarity search on `rag_chunks` table
- **History mode** (`history`) — DynamoDB-based execution history retrieval
- **`kb` / `none`** — no retrieval

Client: `src/orchestrator/contextweave_client.py`; mode dispatch: `src/orchestrator/rag.py`.

### Agent Runtime (substrate seam)

`bedrock_invoke.py` owns the retry policy, the Observatory gate and the
`StepFailed` contract; `agent_runtime.py` owns only "how do I turn a prompt
into text on this platform". **`AGENT_RUNTIME` defaults to `agentcore`.**
Bedrock Agents Classic is deprecated here: it is in maintenance mode, takes no
new features and its model catalogue is frozen at 30 July 2026. It stays
reachable as `AGENT_RUNTIME=classic` because it still runs agents deployed
before the switch and is a one-variable rollback — it is not where new work
goes.

**One runtime serves every agent.** Classic needed one Bedrock agent per
TeamWeave agent, because identity lived in the agent resource. On AgentCore it
does not: `prompt_builder` already composes `ROLE`, `STEP_GOAL` and the output
contract into the prompt, so the runtime is generic and its ARN is a
stack-level value (`AGENTCORE_RUNTIME_ARN`, wired from the stack output). A
per-agent `runtimeArn` in `team.json` still wins if an agent needs its own.
Without either, the call is rejected before it is attempted, and the message
names both ways to fix it.

This exists because Bedrock Agents Classic closed to new customers on
30 July 2026, takes no further features, and its model catalogue is frozen as
of that date — a model released after it is reachable only through AgentCore.
Existing workloads keep running, so the seam is preparation, not a migration.

`AgentCoreRuntime` is implemented, against the botocore service model for
`bedrock-agentcore` (2024-02-28) rather than from recollection. The detail
worth knowing: `runtimeSessionId` has a **minimum length of 33**, and
TeamWeave's run-scoped session ids are shorter — `agentcore_session_id()` pads
deterministically (a hash of the original, never random) so repeated turns of
one run land on the same AgentCore session. The response body is a *streaming
blob*, not an event stream, so it is read once rather than iterated.

Telemetry does not lapse across the substrate: `observe_agentcore_request()`
routes the call through the same mcp-observatory wrapper, under the same
`invoke_agent` operation name, so a migrated agent's spans stay comparable
with its own history. Shadow invocation is alias-shaped and has no AgentCore
equivalent yet — it warns rather than silently collecting nothing, which would
leave the DPO flywheel looking healthy while doing nothing.

`src/agentcore/app.py` is the program an AgentCore runtime runs. Classic is
declarative (register an instruction, Bedrock executes it); AgentCore hosts
your code behind `POST /invocations`, so the agent has to exist as a program.
It is thin on purpose: `prompt_builder` already composes the whole per-turn
prompt and the worker validates the result afterwards, so the program only
applies the instruction as a system prompt and calls Converse. Per-agent
settings (`AGENT_INSTRUCTION`, `AGENT_MODEL_ID`, `AGENT_MAX_TOKENS`) arrive as
environment variables set at `CreateAgentRuntime` time.

**`app.py` must expose `app` at module scope.** `BedrockAgentCoreApp` extends
Starlette: the platform imports the entrypoint file and serves what it finds,
so building the application inside a function leaves it nothing to serve and
the runtime never boots — a failure that surfaces at the first invocation,
long after the deploy reports success. The logic therefore lives in
`src/agentcore/agent.py` (no SDK import) and `app.py` holds only the
module-level `app` and the `@app.entrypoint` function.

**The artifact must be built for Linux ARM64.** AgentCore runtimes are ARM64
and CI runs on x86-64, so `pip install --target` resolves the wrong wheels for
anything compiled — `bedrock-agentcore` brings two, `pydantic_core` and
`websockets.speedups`. Every gate passed on that artifact (template valid,
cfn-lint clean, zip uploaded, entrypoint imported, CloudFormation accepted the
resource) and the runtime then refused to start nine minutes later, rolling
the stack back: *"Your artifact contains binary files that are incompatible
with Linux ARM64."* The shipped tree is now resolved with
`--platform manylinux2014_aarch64 --implementation cp --python-version 3.12
--only-binary=:all:`, and `scripts/check_arm64_artifact.py` reads the ELF
`e_machine` of every `.so` before the upload — the filename is a hint, the
header is evidence. Because aarch64 wheels cannot be imported on the runner,
the entrypoint boot check runs against a second, natively installed tree. The zip is **flat**:
both files sit at its root, which is why `EntryPoint` is `app.py` and the
import in `app.py` is `from agent import run_turn`. The packaging step builds
that zip and imports it before uploading, and
`tests/test_agentcore_entrypoint.py` boots the real artifact over `GET /ping`
and `POST /invocations`.

**Every deploy invokes the runtime for real.** A green `sam deploy` says
CloudFormation created the resource; it says nothing about whether the program
inside it boots, and every way the artifact can be wrong — no ASGI app, a zip
missing the SDK, an execution role that cannot reach Bedrock — fails at the
first invocation and nowhere earlier. `scripts/agentcore_smoke.py` runs after
the deploy and sends one real turn. Three outcomes, kept distinct on purpose:
a runtime that answers passes; a runtime that errors, returns nothing or
returns HTTP ≥ 400 **fails the deploy**; and a CI role that is not allowed to
invoke (or a botocore that does not know the service) is the *check* failing
rather than the runtime — it warns loudly with `NOT VERIFIED` and does not
fail the deploy, because failing every deploy on a permissions gap would be
wrong and reporting it as a pass would be worse.

`AGENT_RUNTIME=agentcore` is the default. An agent works with no `runtimeArn`
of its own — the stack runtime serves it — but every deploy now gives each one
a registry identity anyway.

**Every agent is registered in the AgentCore registry during deploy.**
AgentCore has no "agent" resource: its registry is a runtime plus named
**endpoints**, and `InvokeAgentRuntime`'s `qualifier` *is* an endpoint name
("an endpoint name that points to a specific version", per the service model).
So `scripts/register_agent_endpoints.py` creates one endpoint per agent on the
shared runtime and writes `runtimeArn` + `qualifier` back into each agent's
`bedrock` block in S3 — which `config_loader` already reads and
`AgentCoreRuntime` already sends.

One runtime, one endpoint per agent — **not a runtime per agent**. A runtime is
a whole code artifact; twelve would mean twelve builds, uploads and startup
validations per deploy, the exact shape of the Classic provisioning step that
outgrew the CLI timeout and orphaned agents each run. An endpoint is a name and
a version pin, so the platform gets per-agent identity, telemetry and version
pinning cheaply. It is also how shadow invocation works on AgentCore — two
qualifiers on one runtime — which the DPO flywheel needs.

Three things that fail quietly and are therefore tested: `EndpointName` is
`[a-zA-Z][a-zA-Z0-9_]{0,47}` (the alphabet that made `AgentRuntimeName` fail,
and agent ids are hyphenated far more often than stack names); two agent ids
sanitising to one endpoint name is a **hard error**, never a silent alias,
because a shared identity merges telemetry and moves version pins; and an
endpoint left on an older version keeps serving the previous deploy's artifact,
so existing endpoints are re-pointed rather than skipped. Nothing is ever
deleted — an agent dropped from a config may still be addressed by a run in
flight — and the CI role is not granted delete.

The step runs **after** the config sync, since it writes into the same S3
object the sync merges.

**AgentCore caps endpoints per runtime, and this account's limit is below the
agent count.** `CreateAgentRuntimeEndpoint` returns
`ServiceQuotaExceededException: maxEndpointsPerAgent limit exceeded`. The
decision taken was to **raise the quota via AWS Support** rather than move to a
runtime per agent, so until that lands the step registers as many agents as
fit, writes `qualifier` back only for those, and warns — naming the agents that
missed out and the exact request to file (quota, account, region, number).

That is deliberately a warning and not a failed deploy. An agent with no
`qualifier` falls back to the runtime's default endpoint, which is exactly how
every agent ran before this step existed: the platform is **degraded, not
broken**, and failing every deploy over a fixed account quota would help
nobody. A non-quota error still fails the step — degrading on a quota must not
turn every AWS error into a warning.

**`EnableAgentCore` must be in `--parameter-overrides`, and the workflow
passes it.** `sam deploy` sends `UsePreviousValue=true` for every parameter it
is not given, so a template `Default:` governs only the *first* deploy of a
stack. Changing the default to `"true"` therefore did nothing: the stack kept
the `"false"` it was created with across three green deploys, no AgentCore
resource was ever created, and the functions kept `AGENT_RUNTIME=classic` with
an empty `AGENTCORE_RUNTIME_ARN` — while the workflow, reading its own
`env.ENABLE_AGENTCORE`, skipped Classic provisioning on the strength of a
switch the stack had never seen. Two sources of truth disagreeing in silence.
`tests/test_deploy_parameters.py` holds the workflow to sending every feature
switch it also branches on, and fails when a new `if: env.X` gate appears
without a decision about whether the stack needs telling.

### Structured Output
Worker validates agent outputs against JSON Schema before passing them downstream (`schema_validate.py`, `structured_transform.py`).

### Async Execution
Every run is async: `POST /team/task` returns a `run_id`, then poll `GET /team/task/{run_id}` until `SUCCEEDED` or `FAILED`.

---

## Environment Variables (auto-wired by SAM)

| Variable | Purpose |
|----------|---------|
| `CONFIG_BUCKET` | S3 bucket for team configs |
| `ARTIFACT_BUCKET` | S3 bucket for run artifacts |
| `DDB_TABLE` | DynamoDB table for runs/tasks |
| `STATE_MACHINE_ARN` | Step Functions state machine ARN |
| `VECTOR_DB_SECRET_ARN` | Secrets Manager ARN for pgvector credentials |
| `GEMINI_SECRET_ARN` | Secrets Manager ARN for Gemini API key (optional) |
| `CONTEXTWEAVE_URL` | Base URL of the ContextWeave knowledge layer (required by `contextweave` RAG mode) |
| `CONTEXTWEAVE_API_KEY` | Optional `x-api-key` for ContextWeave (not a template parameter — inject via Secrets Manager) |
| `CONTEXTWEAVE_FEEDBACK_ON_VALID_OUTPUT` | `1` to up-vote the grounding answer after a schema-valid run (default off) |
| `TEAM_CONFIG_PREFIX` | S3 prefix for team configs (default: `teams`) |
| `VECTOR_DB_TABLE` | pgvector table name (default: `rag_chunks`) |
| `VPC_ID` | VPC for Lambda networking |
| `LAMBDA_SUBNET_IDS` | Comma-separated private subnet IDs |
| `RDS_SECURITY_GROUP_ID` | RDS security group for Lambda ingress |

---

## API Endpoints

`openapi/teamweave.yaml` is the full description, including request and
response shapes. `tests/test_openapi_contract.py` compares it against the
`Path:`/`Method:` pairs in `infra/template.yaml` method by method, so the spec
cannot document a route API Gateway does not forward (or miss one it does).

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/team/task` | Start a pipeline run → **202** with a `run_id` |
| `GET` | `/team/task/{run_id}` | Poll run status (RUNNING / SUCCEEDED / FAILED) |
| `GET` | `/teams/task/{run_id}` | Alias of the above |
| `GET` | `/improve/tasks` | List improvement tasks |
| `POST` | `/improve/task/done` | Mark a task complete |
| `GET`/`POST`/`DELETE` | `/agents` | Agent CRUD — GET is synchronous, writes return 202 |
| `GET`/`PUT` | `/agents/{name}` | |
| `GET`/`POST` | `/teams` | |
| `GET`/`PUT`/`DELETE` | `/teams/{team_name}` | |
| `GET`/`POST` | `/roles` | |
| `GET`/`PUT` | `/roles/{role_id}` | |
| `GET`/`POST` | `/departments` | |
| `GET`/`PUT` | `/departments/{dept_id}` | |
| `POST` | `/agent/converse` | One synchronous turn against a Bedrock agent alias |
| `GET` | `/observability/metrics` | SigV4 passthrough to Amazon Managed Prometheus |
| `GET` | `/observability/agent-metrics` | Query the Observatory span table (list or aggregate) |
| `GET` | `/observability` | Unified view: OBSERVATORY_METRICS + ContextWeave routing health + routing decisions |

Three things a client has to get right:

- **Writes are asynchronous.** `POST /team/task` and every provisioning
  mutation return 202 with a `run_id`; nothing has happened yet. Poll
  `GET /team/task/{run_id}`.
- **A `FAILED` run is a 200.** The failure is in the body. A harness that
  checks only the HTTP status records every failed run as a pass.
- **`DELETE` is routed on `/agents` and `/teams/{team_name}` only.** The
  trigger handler accepts DELETE on every management path, but API Gateway
  does not forward it elsewhere, so `DELETE /roles/{role_id}` never reaches
  the handler that would service it.

## Stack Outputs

Everything a harness needs is published by the stack; nothing needs to be
hardcoded. `scripts/stack_env.py` resolves them into environment variables:

```bash
eval "$(python scripts/stack_env.py --stack teamweave --format sh)"
curl -sS "$TEAMWEAVE_API_BASE/observability"
```

| Output | Why a harness needs it |
|--------|------------------------|
| `HttpApiUrl` | The server for every request |
| `DdbTable` | Run and task metadata |
| `ObservatoryMetricsTable` | The span table |
| `ObservatoryMetricsSpanTimelineIndex` | How the timeline is queried — without the index name a caller scans the table and gets a truncated aggregate that looks like data |
| `ObservatoryMetricsAgentIdTimestampIndex` | Per-agent span queries |
| `ConfigBucket` / `ArtifactBucket` | Team configs and run artifacts |
| `StateMachineArn` | The execution a `run_id` belongs to |
| `ContextWeaveUrl` | Empty when unconfigured — the only way to read `/observability`'s null `routingGraph` as "not configured" rather than "broken" |
| `VectorDbSecretArn` | pgvector host/port/dbname/password, fetched at point of use |

---

## AWS Context

- **Region**: `us-east-1`
- **Account**: `239571291755`
- **GitHub Actions Role**: `teamweave-github-actions-sam-deployer` (OIDC)
- **Bedrock Service Role**: `arn:aws:iam::239571291755:role/BedrockAgentServiceRole-Tarun`

---

## Development Notes

- Python 3.12 — match this version locally to avoid dependency drift
- All secrets go through AWS Secrets Manager — never hardcode credentials
- Lambda functions run inside a VPC for RDS access; local integration tests require VPN or RDS proxy
- Test coverage lives in `tests/` — CI runs `pytest tests/` before `sam build`, so a failing test stops the deploy
- CI/CD triggers on push to `main` — the GitHub Actions workflow runs `sam build` + `sam deploy`
- Team configs in `config/examples/` are examples only; live configs are fetched from S3 at runtime
- **Never `aws s3 sync` the team configs.** Provisioning writes the Bedrock
  `agentId`/`aliasId` back into the *same* S3 key, and a fresh CI checkout
  always has the newer mtime — so a plain sync erased them on every deploy and
  forced a full rebuild of every agent (which is how the provisioning step grew
  past the CLI timeout, and how each deploy orphaned the previous deploy's
  agents). `scripts/sync_team_configs.py` merges instead: the repository owns
  definitions, S3 owns the runtime identifiers.
