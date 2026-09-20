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

**Agent identity is a span attribute, not an AWS resource.** The first
attempt gave each agent its own AgentCore endpoint. AWS's quota refused at
twelve, and it was right to: endpoints are a *release* mechanism — production
on a stable version while staging tests a newer one — and the default of ten
per runtime is a budget for channels, not tenants. Spending it on identity
does not scale, and it consumed the very endpoints shadow invocation needs,
which was the argument for doing it.

The [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai)
put identity on the span: `invoke_agent` carries `gen_ai.agent.id` and
`gen_ai.agent.name`, `gen_ai.conversation.id` identifies the session, and an
orchestrator coordinating several agents reports `invoke_workflow` around them
— which is what the Step Functions pipeline is. `mcp_observatory.genai_attributes()`
emits those names **alongside** the originals, never instead of them: the
Observatory GSIs and every dashboard query `agent_id`/`operation`, and
renaming them to make a point about naming would blind all of it. An absent
value is omitted rather than written blank, because the conventions mark these
"when available" and on AgentCore every `alias_id` is empty.

So `scripts/register_agents.py` records the shared `runtimeArn` on every agent,
clears any per-agent `qualifier` left by the old scheme (it names an endpoint
that is no longer that agent's), and creates only **release-channel**
endpoints — `shadow` by default, with `DEFAULT` created by AgentCore itself.
The endpoint count no longer grows with the agent count. A full quota is now a
real signal rather than an expected outcome.

### A2A — how the rest of the platform reaches these agents

[A2A](https://github.com/a2aproject/A2A) reached 1.0.0 in January 2026 under
the Linux Foundation and is how agents built on different stacks discover and
call each other. That is the weave platform's premise, and until now a sibling
service reached TeamWeave through a hardcoded URL and private knowledge of its
payload shape.

**One card, many skills.** TeamWeave is one A2A agent; each TeamWeave agent is
an `AgentSkill` on it, id `{team}.{agent}`. This is the third time the platform
has had to answer "what identifies an agent" and the third time the answer is
not a piece of infrastructure — Classic made one Bedrock agent each, AgentCore
was nearly given one endpoint each, and what holds is a name in a document:
a skill id here, `gen_ai.agent.id` on the span. Adding an agent costs a line of
config.

`GET /.well-known/agent-card.json` is generated from the live team configs in
S3, so a skill cannot advertise an agent that no longer exists. `src/orchestrator/a2a.py`
is pure and holds the shapes; `a2a_handler.py` only translates, because
`message:send` starts the same Step Functions execution `POST /team/task`
starts and `tasks/{id}` reads the same one the status handler reads — an A2A
client and a native client must not drift into different behaviour.

Three things it deliberately does not claim, because a lie in a
machine-readable document is one a machine acts on:

- **One interface, `HTTP+JSON`** — the one actually served. v1.0 replaced the
  top-level `url`/`preferredTransport` with `supportedInterfaces`, each entry
  carrying its own `protocolBinding` and `protocolVersion`; a 0.3-shaped card
  parses fine and means nothing to a 1.0 client.
- **`capabilities.streaming: false`** — `message:stream` is not implemented,
  and a client that believed otherwise would wait on a stream that never opens.
- **`message:send` is non-blocking.** A2A's default is to block until the task
  is terminal; a TeamWeave pipeline outlives any API Gateway request, so the
  task returns in a working state to poll rather than the request timing out
  and losing the run id.

A Step Functions status with no A2A equivalent maps to
`TASK_STATE_UNSPECIFIED`, never a guess: reporting a timed-out run as
completed would be worse than reporting it as unknown. `tests/test_a2a.py`
holds the served card and `openapi/teamweave.yaml` to each other in both
directions, so neither can grow a field the other does not know.

**Discovery is the other half.** `contextweave_client` built every request as
`{CONTEXTWEAVE_URL}` plus a path constant held in *this* repository — so a
route moving in ContextWeave surfaced here as a 404 the RAG layer degraded
past in silence: the run lost its grounding and nothing said so.
`a2a_discovery.resolve_base_url()` now reads ContextWeave's own card and uses
the interface URL it publishes. The env var remains the *seed*, because
discovery needs a first address.

It is strictly additive, which is the property that matters: a sibling serving
no card, an unreachable one, a malformed card, or one naming no `HTTP+JSON`
interface all fall back to the seed — exactly the old behaviour. `A2A_DISCOVERY=0`
opts out without even attempting a fetch, and any exception in discovery is
caught, because a knowledge layer degrading to no context is designed
behaviour here while a discovery layer taking the run down would not be. The
card is cached per container (negative results too, so a card-less sibling is
not re-asked every turn) with a TTL so a redeployed sibling is picked up.

The binding is *matched*, not assumed: A2A orders `supportedInterfaces` by
preference and a client takes the first it supports, so taking the first entry
regardless would send HTTP+JSON to a gRPC endpoint on any sibling listing more
than one.

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

### Egress: IPv6 where AWS serves it

Lambda runs in the VPC with no NAT Gateway. Egress is an Egress-Only Internet
Gateway, which is IPv6-only, so `AWS_USE_DUALSTACK_ENDPOINT=true` is set
globally and every SDK call that can go over IPv6 does.

That setting is a *blanket* instruction, and that is the trap. Every boto3
client builds `{service}.{region}.api.aws` unless an `AWS_ENDPOINT_URL_*`
variable overrides it, and where AWS publishes no dual-stack endpoint that
hostname does not exist — the call dies resolving DNS. **Inside the VPC, at
run time.** Not at deploy, not in CI, because the runner has no dual-stack
setting of its own. Three services were in that state: `bedrock-agentcore`
(the substrate *every* agent turn runs on, which is why the AgentCore smoke
test passed on the runner and a real pipeline step would not have),
`bedrock-agentcore-control`, and `secretsmanager` — the last being the worst,
because `rag.py` swallows the failure and degrades to no RAG context, so runs
get quietly worse rather than failing.

So the template pins two different things, for two different reasons:

- **No dual-stack endpoint exists** — the six Bedrock services and Secrets
  Manager. The pin is the only thing making the call work.
- **A dual-stack endpoint exists and IPv4 is still right** — S3 and DynamoDB.
  Their Gateway VPC endpoints are free and keep the traffic off the internet;
  a dual-stack hostname would push it out through the Egress-Only Internet
  Gateway instead, worse on both cost and exposure. IPv6 is the goal where it
  replaces NAT, not where it replaces a private path.

The variable name comes from the service's **serviceId**, not its client
name — Secrets Manager is `AWS_ENDPOINT_URL_SECRETS_MANAGER`, Step Functions
is `AWS_ENDPOINT_URL_SFN`. A misspelled variable is read by nothing and pins
nothing while looking exactly like a pin that works.

`tests/test_dualstack_endpoints.py` is the gate: offline, and it fails if a
new `boto3.client(...)` names a service that is neither known to have IPv6 nor
pinned. `scripts/check_dualstack_pins.py` covers the other direction, which
tests cannot see — it resolves each hostname and reports pins that have become
unnecessary because AWS has since shipped dual-stack, so the list shrinks
instead of ossifying. It runs in CI as an informational step, never a gate: it
reads a DNS outage as "nothing has dual-stack", and a network blip must not
redden a deploy.

No VPC endpoints were added for any of this.

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
