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

**One production workflow.** The **Visibility Team** is the platform's product
path: five members, director (brief) → strategist (angle) → LinkedIn writer
(drafts) → managing editor (the post) → visual designer (the illustration).
It ended in distribution and approval steps that ran *after* the editor and
produced a plan and a sign-off rather than the thing asked for; those were
removed, and the rule they left behind is now a test — a step may follow the
editor only if it **consumes** the approved copy. The illustrator does; a
plan about the post does not.

`doc_rewrite_team` and `tarun_improvement_team` were both removed, each with
its AgentCore runtime and its entry in the runtime map.
`scripts/pipeline_smoke.py` runs the visibility team, so every deploy
exercises the one path people actually use.

The `/improve/tasks` and `/improve/task/done` endpoints are **not** part of
that removal and still work: they read a DynamoDB task list, which the
improvement team happened to populate but does not own. Nothing new writes to
it now.

Deleting a team from this repository is only half of removing it:
`scripts/sync_team_configs.py` also **prunes** teams S3 still serves that the
repository no longer defines, or `GET /teams` keeps listing one whose
definition is gone. That prune is deliberately narrower than
`aws s3 sync --delete` (banned, see below): it removes a team directory only
in its entirety and only when no such team exists here, never a key *within* a
team. It refuses to run at all when it finds no local teams, so an unreadable
root cannot read as "delete everything".

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

**`features.explicit_rag` gates every mode, not just the explicit one.** It
sits above the mode dispatch in `build_rag_context` and returns `("", {})`
before the dispatch is reached, so with it `false` the visibility team's
declared `mode: "explicit"` was dead config and every agent turn ran
ungrounded — the same "config says X, deployment does Y" shape as the image
`model_id` and `AGENT_MODEL_ID`. The name is misleading and kept for now;
`tests/test_grounding_instructions.py` pins the team's value so it cannot
silently revert.

**Grounding is conditional, and the absence of it is a statement.** The
retrieved block used to be dumped under a bare `RAG_CONTEXT:` label with no
instruction at all, which leaves an agent to infer its purpose — and produces
opposite failures in the two directions:

- with experience, the piece drifts into a career recital: the topic becomes a
  frame for the author rather than the reverse;
- with none, nothing forbids inventing some, so the agent supplies plausible
  projects and outcomes that never happened. Nothing catches that, because a
  fabricated anecdote is exactly as schema-valid as a real one.

`min_confidence` is what makes "only when the experience is actually relevant"
a *retrieval* decision rather than a request to the model: below the floor,
`_contextweave_context` returns no context at all. So an empty block is a real
signal — "nothing relevant was found" — and `prompt_builder` now says so out
loud in a `NO_VERIFIED_EXPERIENCE` branch that names general expertise as a
complete answer and forbids attributing projects, employers, incidents,
metrics or outcomes to the author. The populated branch is labelled
`VERIFIED_EXPERIENCE` and states that it is supporting evidence, not the
subject: a reader who has never heard of the author must still come away with
something.

The team's hard constraints moved the same way. `"Prefer concrete examples
from Tarun's work using RAG context"` was unconditional, so a run that
retrieved nothing was still told to prefer personal examples — and supplied
them.

Because the team now declares `contextweave`, `_validate_rag` **refuses** to
load it when `CONTEXTWEAVE_URL` is empty rather than degrading, and the deploy
resolves that URL from ContextWeave's own `APIEndpoint` stack output before
`sam deploy` rather than hardcoding it — at the parameter step, where the
cause is visible, rather than three layers away at smoke-test time.

The stack is `contextweave-rag-prod`, named in the workflow's `env` block as
`CONTEXTWEAVE_STACK_NAME` beside every other stack name. Neither name the
sibling repository suggests is right: its samconfig says
`contextweave-rag-dev` and its own CLAUDE.md says `expertise-rag-dev`. **One
named stack, not a list of candidates** — falling back to a dev stack when
prod is missing would point the live pipeline at a different knowledge layer
and still report success, which is the same silent-wrong-source failure this
section is about.

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

**AgentCore is a different service, with a different action.** The worker
roles granted `bedrock:InvokeAgent` — Classic — and nothing granted
`bedrock-agentcore:InvokeAgentRuntime`. With `AGENT_RUNTIME=agentcore` as the
default, every agent turn in the VPC was therefore refused with
`AccessDeniedException`, from functions that had been reaching the right
endpoint all along.

The deploy's smoke test could not catch it: `agentcore_smoke.py` invokes the
runtime **as the deployer role**, which may do anything. A check that runs as
the wrong identity proves the runtime answers somebody, not that it answers
the caller who needs it — which is why the pipeline run matters as a separate
gate. `tests/test_substrate_permissions.py` derives the requirement from the
template, so a substrate switch cannot outrun its permissions again, and
checks the grant names AgentCore *resources* rather than Classic ARNs (a
Bedrock agent ARN never matches an AgentCore runtime, and the diff looks
right either way).

`AGENT_RUNTIME=agentcore` is the default, and an agent carries no `runtimeArn`
of its own: its team's runtime serves it.

**One runtime per team.** A team is the deployment unit. Each team gets its
own `AWS::BedrockAgentCore::Runtime` (`teamweave_{team}`), and the stack
publishes a `team -> ARN` map as `AGENTCORE_TEAM_RUNTIME_ARNS`, which
`AgentCoreRuntime.resolve_arn` reads. Resolution is most-specific-first: the
agent's own `runtimeArn`, then its team's runtime, then the stack-wide
`AGENTCORE_RUNTIME_ARN` so a team added as JSON before its runtime exists
degrades rather than failing.

One runtime for the whole platform put every team's agents in one blast radius
and one endpoint budget — release channels are ten per runtime and *shared*, so
two teams could not be canaried independently and a bad version reached all of
them at once. Per team, each has its own version history, its own channels and
its own failure.

The cost is real and worth naming: adding a team now needs a template change,
on a platform whose premise is that a team is JSON in S3.
`tests/test_team_runtimes.py` is where that coupling is made visible — a team
in `config/examples/teams` with no runtime fails there rather than silently
falling back to the shared runtime and losing the isolation that was the point.
`Fn::ForEach` (the `AWS::LanguageExtensions` transform) would generate them
from a parameter and is the way to remove the coupling later.

The wiring is as important as the logic: the worker passes `team=` down to
`bedrock_invoke`, and `AgentRef` carries it. Without that every ref arrives
with `team=""`, every team resolves to the shared runtime, nothing fails, and
the isolation silently does not exist. A test walks the worker's AST and fails
any invocation that omits the keyword.

The map is the other place that wiring can be absent without failing. A team
can have its runtime resource and be missing from `AGENTCORE_TEAM_RUNTIME_ARNS`
— every one of its turns then falls back to the shared runtime, exactly as if
the resource were never written. The check for that searched the whole
template for the team's name, which every per-team runtime carries twice
already (`AGENT_TEAM` and its `Description`), so it passed on a template with
the team dropped from the map. It parses the `Fn::Sub` now and holds three
things: every configured team is a key, no key names a team that no longer
exists, and each key's `!GetAtt` resolves to the runtime whose `AGENT_TEAM`
is that same team — two teams' substitutions crossed would deploy and route
one team's agents into the other's runtime.

`register_agents.py` reported `--runtime-id` — the *shared* runtime — whatever
runtime each team's agents were written to, so its summary could not tell a
working setup from a team that had silently fallen back. It now names the
runtime per team (`visibility -> teamweave_tarun_visibility_team`) and warns
when a team has none of its own.

**Nothing is provisioned per agent at all.** An agent is a prompt: the runtime
is generic, its instruction arrives in the payload, and `prompt_builder`
composes `ROLE`, `STEP_GOAL` and the output contract per turn. So the script
derives no endpoint name from an agent id, refuses no collision between two
such names, and writes no `runtimeArn` back into `team.json` — the helpers
that did are gone rather than merely unused, because the per-agent endpoint
scheme is the mistake this design keeps being pulled back toward.

It does the opposite now: `clear_runtime_identity` **strips** the pins earlier
deploys stamped. Stopping writing them would not have been enough. `resolve_arn`
answers most-specific-first, so a stamped value is the most specific there is —
it shadows the team map entirely, and an agent pinned to last deploy's runtime
keeps going there after its team's runtime is replaced. Writing even the
*correct* ARN freezes the agent at the runtime that deploy happened to see.

`sync_team_configs.py` is the other half, and getting it wrong would have made
the clear undo itself forever: it carried `runtimeArn`/`qualifier` across as
S3-owned state, so the next deploy's merge would read them out of the previous
copy and write them straight back. They are off that list now.
`agentId`/`aliasId` stay on it — Classic really does provision one Bedrock
agent per TeamWeave agent, so those are genuine S3-owned state and clearing
them would make the `AGENT_RUNTIME=classic` rollback rebuild every agent.

The escape hatch survives on purpose: a `runtimeArn` a *person* writes in
`team.json` still wins at resolution and still survives the merge — as a
definition the repository owns, which is what it now is. What is gone is the
deploy writing one automatically, which made the hatch indistinguishable from
the default.

**The model has to travel with the turn, for the same reason the identity
does.** `AGENT_MODEL_ID` is an environment variable set at
`CreateAgentRuntime` time, so it belongs to the *runtime*, and one generic
runtime serves every agent of a team. A model read only from there serves one
model to every agent while `team.json` declares one per agent — the config
says four and the deployment answers one, with nothing failing and nothing
logged. `AgentRef` carries `model_id`, `build_payload` sends it as `modelId`
when set, and `run_turn` prefers it over the environment; omit it and the
runtime keeps its own default, so an agent that declares no model is
unaffected.

`build_payload` took an `instruction` argument from the day it was written and
`invoke` never passed one — the seam existed, was documented, and was
connected to nothing, so every turn fell back to the runtime's
`AGENT_INSTRUCTION`. A helper with a parameter nothing supplies reads exactly
like a wired feature. That is why the test drives the real `invoke` and
inspects the bytes it sends rather than calling `build_payload` directly: a
builder test reproduces the blind spot instead of catching it.

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

So `scripts/register_agents.py` creates only **release-channel** endpoints —
`shadow` by default, with `DEFAULT` created by AgentCore itself. The endpoint
count no longer grows with the agent count, and a full quota is a real signal
rather than an expected outcome.

### The tool gateway — which weave systems agents can call

The AgentCore gateway existed with **nothing behind it**. One
`AWS::BedrockAgentCore::GatewayTarget` was declared, gated on a
`ScreenWeaveMcpEndpoint` parameter that defaulted to empty and that the deploy
never passed — so the condition was false on every deploy, the target was
created on none, and the gateway was infrastructure for nothing. Nothing
failed: an unset parameter is not an error, a false condition is not an error,
and a gateway with zero targets deploys green. The same shape as the A2A card
that served `"skills": []`.

**Only an HTTP MCP server can be a target.** `TargetConfiguration.Mcp.McpServer`
requires an `Endpoint`, so a stdio server has nothing to point at. Four
siblings qualify — ScreenWeave, CipherWeave, DataDictionary and ToolWeave, each
serving `mcp.http_app(stateless_http=True)` at `/mcp`. **DeployWeave is absent
on purpose**: it calls `mcp.run()` with no transport, which is stdio, and a
target for it could never resolve.

Each target is gated on **its own** parameter, not a shared one. A single
condition would mean one unreachable sibling silently taking the other three
down with it, and a sibling that has not been deployed yet is not a broken
platform — it simply contributes no tool. So the deploy resolves each endpoint
from that sibling's own stack output, warns by name for the ones it could not
find, warns again if it found none at all, and never exits non-zero. That is
the opposite of the ContextWeave URL, which a declared RAG mode makes
mandatory and which therefore fails the step.

**Resolving a URL is not enough, and "one tool fewer" was not true.** AgentCore
handshakes the MCP server while *creating* the target, so a server that refuses
the handshake does not produce a missing tool — it produces

    GatewayTarget ... failed to stabilize, status: FAILED, reason: Failed to
    connect and fetch tools from the provided MCP target server.
    Error - Unsupported protocol version

and CloudFormation rolls the **whole stack** back. One sibling's bug takes the
platform's deploy with it, which is the exact opposite of the degradation the
per-target conditions are for. ScreenWeave's hand-rolled MCP Lambda answered
`initialize` with a hardcoded `2024-11-05` whatever the client asked for, and
did this on the first deploy that ever reached it — the earlier deploys could
not, because its stack name was wrong.

`scripts/mcp_handshake.py` closes that gap by asking the endpoint before the
parameter is passed. The check is the MCP handshake rule itself, so it needs no
knowledge of what AgentCore requires: a server that supports the requested
revision echoes it, and one that answers with a *different* version is saying
it does not speak ours. Three outcomes, and the middle one is the point:

| Outcome | Meaning | Action |
|---|---|---|
| `ok` | echoed the requested revision | wire the target |
| `refused` | answered with another revision | skip it, naming both versions |
| `unverified` | no answer, or one it could not read | **wire it anyway** |

`unverified` deliberately does not skip. The probe runs unauthenticated from a
CI runner while the Gateway calls with its own identity from its own network,
so a probe failure is weak evidence about the Gateway. Dropping a working tool
on it would be the silent-missing-tool failure this repository keeps hitting;
letting it through means CloudFormation says so, loudly, in a message whose
cause is now known.

Two shapes matter and both are handled: an MCP server over HTTP may answer
`initialize` as plain JSON or as an SSE `data:` frame, and a check reading only
one would report every streaming server as unverified and wire it blind.

**Each sibling's stack name is the one thing here nothing can derive.** Two of
the four were guessed wrong and the gateway came up with two targets instead
of four. The names share no convention, because each sibling's own deploy
chose it:

Read each one from that sibling's **CI workflow**, which is the thing that
creates the stack. A `samconfig` `default_name` or a `deploy.sh` derivation is
what someone gets running it by hand, and is not evidence of what exists —
three of these four were guessed from a local default and two of those were
wrong, `screenweave` twice over:

| Target | Stack | Source |
|---|---|---|
| ScreenWeave | `screenweave` | `.github/workflows/deploy.yaml`: `STACK_NAME: screenweave` (also its README and `docs/architecture.md`). `deploy.sh` *derives* `screenweave-${ENV}`, which produced two wrong guesses |
| CipherWeave | `cipherweave-prod` | workflow: `cipherweave-${{ inputs.stage \|\| 'prod' }}` |
| DataDictionary | `data-dictionary-mcp-prod` | workflow: `data-dictionary-mcp-${STAGE:-prod}`; its samconfig's unsuffixed name is the *local* default, not what CI deploys |
| ToolWeave | `toolweave` | samconfig, unsuffixed — and its workflow agrees, which is the only reason reading the samconfig worked here |

A wrong name and an undeployed sibling are indistinguishable from here — both
resolve to nothing, both warn, neither fails — so the wrong guess cost a round
trip through a human. The deploy now separates the two failures that used to
read alike: a stack that **does not exist** (wrong name, or not deployed) is
reported with the stacks that do exist and look close (`no stack
'screenweave-prod'; found: screenweave-dev` — the answer, in the log), while a
stack that exists but **publishes no such output** is reported with the keys it
does publish. One is fixed in this repository, the other in the sibling's
template.

`tests/test_gateway_targets.py` extracts that resolution loop and **runs it**
under bash against a fake `aws`, rather than only reading the workflow as text.
A structural assertion that the right strings appear passed happily on a
version whose missing-stack branch had been made dead, and on one that wired
all four targets while logging "infrastructure for nothing". It also holds the
two halves to each other in **both**
directions, because each is invisible from the other side: every declared MCP
parameter must have a target that reads it and a condition that gates on it,
and every parameter the workflow passes must be one the template declares —
`sam deploy` rejects an unknown override, but only on a run where that
sibling's stack actually resolves, so the mismatch deploys green until the day
it does not. It also checks no two targets share a condition, which is the
regression that would restore the original defect.

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
S3, so a skill cannot advertise an agent that no longer exists — which also
means the function needs `CONFIG_BUCKET`. It shipped without one, and the
deployed card served `"skills": []` while twelve agents were registered; a
client discovering TeamWeave learned it could do nothing. `STATE_MACHINE_ARN`
was missing too, so `message:send` would have returned 500 on every call. The
stack was valid, cfn-lint clean, the suite green and the deploy successful.
`tests/test_function_environment.py` now reads each handler's module for the
variables it takes from the environment and fails when the template does not
supply one, unless the name is allowlisted there with a reason. `src/orchestrator/a2a.py`
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

**The card is public; the operations are not.** The API sets
`DefaultAuthorizer: SiweAuthorizer`, which covers every route unless one opts
out — so the card first shipped behind the very authentication it exists to
describe, and the live URL answered `401 Unauthorized`. A card you need a
token to read cannot be used to discover anything, and TeamWeave's own
`a2a_discovery` fetches sibling cards with no credentials, so TeamWeave could
not be found by the mechanism it uses on others. The card route is now
`Authorizer: NONE`; `message:send` and `tasks/{id}` stay authenticated, and
the card declares the bearer scheme (`securitySchemes` + `security`) so a
client knows that before it calls rather than inferring it from a 401. It
names the scheme and never a credential — a test walks every value in the
served card for anything token-shaped.

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

### Asking a team to do something

This is what the platform is for, and it had no path in the UI at all until
September 2026: you could chat with a single agent, list teams and edit
prompts, but not run the pipeline the whole thing exists for.

Two pieces make it work without hardcoding a form per team:

- **`request_schema` in `team.json`** declares the inputs a run needs
  (`summary` plus `fields`, each with `name`/`label`/`type`/`required`).
  `GET /teams/{name}` already returns the whole document, so the UI builds the
  form from it with no new endpoint and adding a team stays a JSON change.
  `tests/test_request_schema.py` derives the check: a config that reads
  `request.document_text` anywhere must declare `document_text`, or the form
  would leave it empty, the run would start anyway, and the agent would be
  handed a blank where the brief should be.
- **`scripts/pipeline_smoke.py`** runs one real pipeline after every deploy
  and reads what it produced. A green `sam deploy` says CloudFormation
  accepted resources; `agentcore_smoke.py` says one runtime boots. Neither
  says a person can ask a team to do something and get an answer.

The smoke test's sharpest case is the **empty success**: a run that reaches
`SUCCEEDED` with nothing in its final step. Every signal the platform emits
says that worked — it is the same shape as the A2A card that served
`"skills": []` and the IPv6 query that matched nothing — so it fails the
deploy. A run that errors fails too, and names the status, because "produced
nothing usable" would send whoever reads it to debug the wrong thing. A CI
role that cannot start executions is the *check* failing rather than the
pipeline: it warns `NOT VERIFIED` and does not fail the deploy.

It starts the execution through Step Functions rather than the HTTP API,
because the API is behind the SIWE authorizer and CI holds no wallet. That is
the trade: it covers the pipeline, not the authorizer.

**A call must not outlive the invocation that made it.** The worker ran the
whole pipeline with a 300 s Lambda timeout while its Bedrock clients were
configured to wait **1800 s** for one response. Six times the function's
entire budget, which makes the SDK's read timeout unreachable by
construction: Lambda kills the process first, so every stall arrived as

    Sandbox.Timedout: Task timed out after 300.00 seconds

naming no step, no agent and no cause. No amount of logging inside the worker
would have helped — it never reached its error path. That is what the first
real pipeline run ever attempted produced.

`deadline.py` records what the Lambda context says is left, and each outbound
call gets that minus a reserve, so a stall raises `ReadTimeoutError` with
enough time for the worker to say which step hung. The AgentCore client is
built per call rather than cached, because the budget shrinks as earlier steps
spend it — a client made for the first agent would hand the last one a timeout
computed when the function was fresh, and that is exactly the call that
outlives it.

The worker's own timeout is now 900 s, Lambda's ceiling. One invocation runs
every step, so its budget is the sum of all of them; that ceiling is also the
limit on how large a team this architecture serves, and past it the pipeline
needs a Step Functions state per step rather than a loop inside one function.
`tests/test_deadline.py` derives the invariant from the template: no
`read_timeout` anywhere in `src/orchestrator/` may be as long as the function
that makes the call.

### An image member is not an agent turn

The visual designer is a full member of `team.json` — role, department, a
place in the workflow — and it never reaches the agent runtime. Bedrock's
image models do not implement `Converse`, which is all `src/agentcore/app.py`
speaks, so sending a Canvas id through the per-turn model seam would fail at
the first call. `bedrock.modality: "image"` is what routes it: the worker
branches to `bedrock_image` and the step rejoins the pipeline as an ordinary
schema-shaped output.

Teaching the runtime program to branch instead would put binary handling, S3
credentials and a second request shape inside the artifact whose ARM64
packaging already has its own failure mode — and an image step has no `ROLE`
or `STEP_GOAL` to compose. Same argument as `structured_transform`: a
transform is not an agent.

**The body is provider-defined and botocore does not describe it.**
`InvokeModel`'s `body` is an opaque blob in the service model, so unlike the
`bedrock-agentcore` client this shape cannot be verified offline — it is the
one thing here that needs a real call to confirm. That is exactly the trap
that produced `ValidationException: Malformed input request` when
`structured_transform` kept an Anthropic body on a Nova model, so the builder
is per family and **refuses** a family it does not know rather than sending a
body its provider will reject with a message naming nothing useful.

**The bytes never enter the pipeline.** Every step output travels through
Step Functions state, which caps at 256 KB; a base64 PNG inline fails the run
at the state transition, after paying for the image. The image goes to the
artifact bucket and the step returns `image_uri` (durable `s3://`) plus
`image_url` — a presigned URL, best-effort, signed with Lambda's temporary
credentials so it outlives the run by hours, not the seven days SigV4 allows.
A signing failure returns `""` rather than failing a step whose artifact is
already stored.

A response carrying no image is how these models report a content-filter
block, so `first_image_b64` raises instead of returning empty bytes — that
would put a corrupt object in the bucket and report the step as succeeded,
which is the empty-success failure this platform has been bitten by three
times.

**The UI had to change with it.** `finalOutput` showed the workflow's last
step as the deliverable, and the illustrator is now last — so the run would
have displayed `{image_uri, model_id, …}` where the post belongs and buried
the post behind the "earlier steps" disclosure. It skips image-producing
steps when choosing what to show, and `runImages` renders them beside the
copy they illustrate.

`scripts/pipeline_smoke.py` had to make the same change for a sharper reason.
Checking the *last* step for substance would have checked the image reference
— which is always a non-empty dict, even when generation failed — and passed
a run whose post was empty. That is precisely the empty success the script
exists to catch, so the two now agree on what the deliverable is.

**The illustration adorns the deliverable; it is not the deliverable.** The
first real run proved that the hard way: Bedrock refused the image model
(*"marked by provider as Legacy and you have not been actively using the
model in the last 30 days"* — the same refusal that killed the repair model)
and the whole pipeline ended `FAILED`, throwing away a finished, approved
post because a picture of it could not be made. The step degrades now, as the
RAG layer does: it records `error`, claims no image, and the run keeps its
post. That is not an empty success — nothing says an image exists — and both
the smoke test and the UI read `error` and say so.

**A member's declared model has to reach the call.** `bedrock_image.generate`
read only `IMAGE_MODEL_ID`, which nothing set, so the `model_id` in
`team.json` was ignored and every request went to the built-in default:
editing the config changed nothing and said nothing. That is the same trap
the text agents had with `AGENT_MODEL_ID`, reintroduced in the same session it
was fixed — which is why `model_id(declared)` resolves most-specific-first
(the member's, then the stack's `ImageModelId` parameter, then a default) and
a test walks the worker's real call to check the value is handed over rather
than merely resolvable.

**Stop guessing model ids; ask the account.** Two deploys were spent on
guesses that fail identically from the outside and are different problems:

    amazon.nova-canvas-v1:0  ->  "marked by provider as Legacy and you have
                                  not been actively using the model in the
                                  last 30 days"      (real id, no access)
    amazon.nova-canvas-v2:0  ->  "The provided model identifier is invalid"
                                                     (no such model)

`scripts/image_models.py` answers both from the account.
`ListFoundationModels` is on the **`bedrock` control-plane client**, not
`bedrock-runtime`, takes `byOutputModality="IMAGE"`, and returns
`modelLifecycle.status` (ACTIVE/LEGACY) beside `inferenceTypesSupported` — so
*exists*, *is not retired* and *can be called on demand* stay three readable
facts rather than one guess. It adds a fourth the account cannot know: whether
`bedrock_image.build_body` has a body shape for that family, because a model
this account can call is still unusable here if its provider takes a different
body.

The deploy runs it as an informational step (`continue-on-error`, never a
gate — a missing `bedrock:ListFoundationModels` grant must not redden a
deploy), and `pipeline_smoke.py` calls it **only when an image step failed**,
so the warning names real candidates instead of advice. `describe()` never
raises: it runs inside the reporting of another failure, and a traceback there
would replace a useful warning with a complaint about the warning.

Model access is granted per model in the Bedrock console, and an unentitled
model is refused at the call, not at deploy — so `ImageModelId` is a stack
parameter and a member's `model_id` in `team.json` overrides it, neither
needing a code change.

**Gemini is the second image provider, because Bedrock's blocker is an
entitlement rather than a bug.** Both refusals were about *access to a model*,
which is granted per model in a console and cannot be fixed from this
repository. The Gemini key is already in Secrets Manager, already read by
`gemini.py`, and already reaching `generativelanguage.googleapis.com` out of
this VPC for the research brief — so nothing new had to be unblocked.

`bedrock.image_provider` (`"bedrock"` | `"gemini"`) selects it, and
`src/orchestrator/gemini_image.py` answers only "prompt in, bytes out". It is
**not** a `tool_registry` tool: tools are pre/post processors that shape a
step's inputs or outputs around an agent turn, and the illustrator has no
agent turn to wrap — generating the image *is* the step. Everything
provider-independent — the prompt, the S3 write, the presign, the degrade, the
schema — stays in `_run_image_step`, so adding a provider is one function, the
same seam `agent_runtime.py` is for text.

Three details that would each have shipped broken:

- **`responseModalities: ["IMAGE"]`** is what separates an image from a
  description of one. Without it the model returns prose about the picture it
  would draw, which arrives as a step that succeeded and produced no image.
- **The mime type comes from the response.** Gemini may answer JPEG; writing
  that to a `.png` key serves a file whose extension lies, so the extension is
  derived from `content_type` rather than assumed.
- **An unknown provider degrades and names itself.** A typo silently falling
  through to Bedrock would send a Gemini model id there and reproduce the
  exact failure the switch exists to avoid.

Its body is provider-defined and unverifiable offline, like Bedrock's, so the
builder speaks only `:generateContent` and **refuses** Imagen — which answers
on `:predict` with a different body — rather than sending one shape to the
other's endpoint. `gemini_image.list_models()` is the `ListFoundationModels`
equivalent (`supportedGenerationMethods` says whether a model answers
`generateContent` at all), and the smoke test asks *whichever service
refused*: listing Bedrock models for a Gemini failure would name candidates
the run cannot use.

### Structured Output
Worker validates agent outputs against JSON Schema before passing them
downstream (`schema_validate.py`, `structured_transform.py`).

When an agent's answer is not schema-shaped, `structured_transform` reshapes
it rather than losing it. That repair model was hardcoded to
`anthropic.claude-3-haiku-20240307-v1:0` with no way to change it short of a
deploy, and Bedrock now refuses the id outright — *"marked by provider as
Legacy and you have not been actively using the model in the last 30 days"*.
So every repair failed and a working pipeline returned its answer wrapped in a
`fallback_response` envelope instead of the schema its team declared. The run
still **succeeded**, which is how this survived a green deploy — and a passing
test that asserted the broken value.

It defaults to the model the agent turns themselves run on, so the platform
has one model decision, and `STRUCTURED_TRANSFORM_MODEL_ID` overrides it
without touching code.

Changing the id was not enough on its own. `InvokeModel`'s request body is
defined by the model's **provider** — an Anthropic-shaped body
(`anthropic_version`, string `content`) is malformed for Nova — so the model
and the payload were coupled, and moving between them produced

    ValidationException: Malformed input request

which reached the caller as the same fallback envelope the dead model had. It
uses the **Converse** API now, which normalises the request across providers,
so the model id is the only thing that changes when moving between them. The
test's fake client raises if `InvokeModel` is called at all.

### DynamoDB has no float type

A run died on its **first step**, after the retrieval and the agent turn had
both been paid for:

    Failed DynamoDB put_item for table=... pk=RUN#... sk=STEP#...director:
    Float types are not supported. Use Decimal types instead.

`_contextweave_context` puts the knowledge layer's `confidence` into the meta
the worker persists as `inputs_json.rag_meta`, and boto3's DynamoDB resource
refuses a float outright. One value, nested two levels down in a telemetry
field, took out the whole pipeline.

Converting it at that call site would have been the wrong fix. What this DAO
writes is not a fixed set of fields — `inputs_json` and `output_json` carry
whatever the retrieval layer and the agents produced — so any agent answering
with a score, a ratio or a temperature would have failed identically, at a
different step, on a different day. `DbDao._to_dynamo_numbers` walks the whole
item in `_safe_put`, which every write in the DAO already funnels through;
`tests/test_dynamo_number_types.py` holds that chokepoint with an AST check,
because a new method calling `self.table.put_item` directly would be the whole
defect back.

Three details the conversion has to get right, each of which trades this
failure for a later one:

- **`Decimal(str(x))`, not `Decimal(x)`.** The latter takes the full binary
  expansion — `Decimal(0.1)` is 55 significant digits and DynamoDB accepts 38.
  Rounding to 8 d.p. matches `mcp_observatory._to_decimal`, so the platform
  stores numbers one way.
- **Non-finite floats are kept as strings.** `json.loads` accepts `NaN` and
  `Infinity` by default, so an agent's output can carry one, and
  `Decimal("NaN")` is refused by the same serializer with a different message.
  Dropping the key would lose the one value that explains what went wrong.
- **`bool` is not a float.** A conversion written against
  `isinstance(x, (int, float))` would store `cache_hit` as `0`.

The tests' fake table serialises with boto3's **real** `TypeSerializer` rather
than a hand-written rule about which values it rejects. That is the component
that raised in production; a fake written from recollection of its behaviour
agrees with whatever the code does, which is how the existing Step Functions
fake hid the `run_id` defect.

Nothing reads these step records back today (`get_run` has no caller), so the
`Decimal` reaches no HTTP response. When one appears, note that both handlers
serialise with `json.dumps(..., default=str)` — which renders a `Decimal` as a
JSON *string*, not a number.

### Async Execution
Every run is async: `POST /team/task` returns a `run_id`, then poll `GET /team/task/{run_id}` until `SUCCEEDED` or `FAILED`.

**Starting a run and reading one are authorized against different
resources.** `StartExecution` names the state machine; `DescribeExecution`
names the *execution*, whose ARN uses the `execution:` resource-type segment
rather than `stateMachine:`. The status role scoped it with
`!Sub "${StateMachine}:*"` — which expands to the state machine's own ARN —
so every poll of every run was denied and the UI could never report a result,
however well the pipeline ran. The A2A role, written later, had it right, and
nothing compared them. This is the same family as `bedrock:InvokeAgent` versus
`bedrock-agentcore:InvokeAgentRuntime`: the right-looking grant against the
wrong resource, which reviews as correct.

`tests/test_execution_permissions.py` checks every role's statements: an
execution-scoped action must name an `execution:` ARN, `StartExecution` must
not, and no policy may use `sfn:` — that is botocore's client name, not an IAM
prefix, so `sfn:StartExecution` and `sfn:DescribeExecution` granted nothing
while making the policy read as though they did.

**The id a caller is handed must be the id it can poll back.** Fixing that
permission exposed the next link in the same chain: the poll was now allowed
and answered `404 {"error": "run_id not found"}` instead. Both start sites
minted a uuid4, put it in the execution's *input*, and called
`start_execution` **without `name=`** — so Step Functions generated a
different uuid of its own and the id the caller was given named an execution
that never existed. The run itself was fine; only the handle was wrong.

`scripts/pipeline_smoke.py` cannot see this class of bug, and that is correct
behaviour for a harness: it polls the `executionArn` that `start_execution`
returns rather than an id it was handed, so it exercises the pipeline and not
the round trip. The existing trigger tests could not see it either, for a
worse reason — their fake Step Functions returned a fixed ARN whatever `name`
it was passed, which reports success for both the working and the broken call.

`src/orchestrator/run_ids.py` holds the one definition — minting an id and
turning it back into an ARN — and `tests/test_run_id_roundtrip.py` drives the
real handlers through start → poll, over both `POST /team/task` and A2A's
`message:send` / `tasks/{id}`, against a fake that invents its own name when
given none.

A unit test can only hold that rule against a fake written from the same
understanding of the ARN's shape: if the shape itself were wrong, both would
agree and both would be wrong. So `pipeline_smoke.py` now names its execution
and checks that `to_execution_arn` rebuilds the ARN **Step Functions actually
returned** — the one part of this that needs real AWS to falsify, run on every
deploy.

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
| `CONFIG_PREFIX` | S3 prefix for team configs (default: `teams`). The deploy workflow's own shell variable is `TEAM_CONFIG_PREFIX`; both come from the `TeamConfigPrefix` parameter, but this is the name the Lambdas read. |
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

None of the above did anything until September 2026, for a reason worth
knowing. The EOIGW, the dual-stack setting, `Ipv6AllowedForDualStack` on every
function and `AssignIpv6AddressOnCreation` on both subnets were all in place —
but a subnet only gets an IPv6 /64 if the stack is told the /56 that AWS
assigned the VPC, and the deploy discovers that with

    Vpcs[0].Ipv6CidrBlockAssociationSet[?State==`associated`]

A `VpcIpv6CidrBlockAssociation` has no top-level `State`; it is at
`Ipv6CidrBlockState.State`. The filter matched nothing, the expression
returned None, `VpcIpv6Block` was never passed, and the shared stack reported
`DualStackEnabled: false` on every deploy while every log line said IPv6 was
on. All egress went out the NAT instance.

Nothing failed. A JMESPath matching nothing is not an error, the `|| echo ""`
fallback never fired because the CLI call succeeded, and the two-pass deploy —
which exists precisely to activate dual-stack in the same run that requests
the block — used the same function for its second look and concluded the block
still had not been assigned. `tests/test_ipv6_discovery.py` evaluates the
workflow's own query against a response built from botocore's EC2 service
model, so the shape is checked against the data the CLI itself validates
against rather than against recollection.

**The NAT instance is the whole platform's egress.** Because every Bedrock
service is IPv4-only from here, one `t4g.nano` NAT instance is the only route
out for every agent turn. It was a **spot** instance with
`InstanceInterruptionBehavior: stop`, so an interruption stopped it, the route
kept pointing at a stopped instance, and every call failed with

    Connect timeout on endpoint URL:
    https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/.../invocations

Silently — nothing watched the instance, and a connect timeout reads like a
slow service rather than a missing route. That is what the first real pipeline
run ever attempted died of. It is on demand now, and the deploy checks the
instance is running before anything tries to invoke an agent.

An interface VPC endpoint for `bedrock-agentcore` is the more robust answer:
private, no NAT dependency, no single instance. It costs more per AZ, so it is
a deliberate decision rather than a build fix, and is named here rather than
taken unilaterally.

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
- **The config bucket's layout is a contract between the deploy and the
  provisioner.** `ProvisionTeamFunction` derives three keys from one variable:
  `teams_prefix = "{OUTPUT_PREFIX}/teams"`, `roles_key =
  "{OUTPUT_PREFIX}/roles.json"`, departments likewise. With `OUTPUT_PREFIX:
  teams` it scanned `teams/teams/` for team configs while the deploy wrote
  them to `teams/<name>/<version>/team.json`, and looked for roles at
  `teams/roles.json` while the deploy put them at the bucket root. So
  `GET /teams` answered `{"teams": [], "count": 0}` with a **200** — no error,
  no exception, an empty team picker and empty Roles and Departments tabs.
  Neither side was wrong alone, which is why nothing caught it: the workflow
  uploads exactly where it says and the Lambda reads exactly where it is told.
  `OUTPUT_PREFIX` is now empty, and `tests/test_config_bucket_layout.py`
  computes both sides from their real sources and compares them.

- **A trimmed workflow must not leave dangling references.**
  `tests/test_workflow_integrity.py` checks, for every team: each step's
  `inputs` name a step that exists, every step has an agent, every agent is
  run by a step, every `schema_ref` resolves, no schema is orphaned, and a
  `default_jump_to_step` survives. A survivor naming a dropped step resolves
  to nothing — the agent gets a prompt missing the context it was written for,
  which is a worse answer rather than an error.

- **Never `aws s3 sync` the team configs.** Provisioning writes the Bedrock
  `agentId`/`aliasId` back into the *same* S3 key, and a fresh CI checkout
  always has the newer mtime — so a plain sync erased them on every deploy and
  forced a full rebuild of every agent (which is how the provisioning step grew
  past the CLI timeout, and how each deploy orphaned the previous deploy's
  agents). `scripts/sync_team_configs.py` merges instead: the repository owns
  definitions, S3 owns the runtime identifiers — and on AgentCore there are no
  per-agent runtime identifiers to own, so `runtimeArn`/`qualifier` are
  deliberately *not* carried across (see the registry section above).
