# MCP Observatory Implementation & Adoption Guide

**Date:** April 2026  
**Status:** Production  
**TeamWeave Version:** 1.0+  
**mcp-observatory Version:** 0.2.0

---

## Table of Contents

1. [Overview](#overview)
2. [Core Implementation](#core-implementation)
3. [Architecture & Design Decisions](#architecture--design-decisions)
4. [Integration Points](#integration-points)
5. [Adoption Guide for New Projects](#adoption-guide-for-new-projects)
6. [Operational Considerations](#operational-considerations)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### What is mcp-observatory?

**mcp-observatory** is a telemetry instrumentation library that wraps Bedrock agent and model invocations with comprehensive observability signals. It instruments every call through the `InvocationWrapperAPI`, capturing:

- **Per-call telemetry**: Trace ID, token estimates, cost, latency, policy decisions
- **Risk scoring**: Hallucination risk, composite risk, gate decisions
- **Shadow analysis**: Dual-invoke disagreement and numeric variance signals
- **Structured logging**: Consistent, queryable log records for every span
- **Persistent storage**: Telemetry persisted to DynamoDB with configurable TTL

### Why mcp-observatory?

TeamWeave operates multi-agent pipelines that invoke Bedrock agents and models for critical workflows (content marketing, personal learning loops). Without observability:

- ❌ Hidden quality degradation in outputs
- ❌ No early warning when models disagree
- ❌ No way to correlate latency/cost with policy decisions
- ❌ No signal for training data collection (DPO, preference learning)

**mcp-observatory solves this** by:

✅ Capturing disagreement signals from shadow aliases (dual-invoke)  
✅ Computing risk scores from disagreement & consistency signals  
✅ Persisting structured telemetry for analytics & model improvement  
✅ Integrating with Amazon Managed Prometheus for real-time dashboards  
✅ Providing zero-friction integration: wrap → invoke → observe

---

## Core Implementation

### Source Files

```
src/orchestrator/
├── mcp_observatory.py           # Core wrapper + telemetry pipeline
├── observatory_handler.py        # Lambda handler for PromQL queries
├── bedrock_invoke.py             # Bedrock agent invocation wrapper
├── amp_metrics.py                # Amazon Managed Prometheus integration
└── db.py                         # DynamoDB DAO operations

tests/
├── test_mcp_observatory.py        # 50+ unit tests
└── test_observatory_handler.py    # Handler integration tests
```

### Core Components

#### 1. **mcp_observatory.py** — Main Instrumentation Module

**Public API:**

```python
def observe_agent_request(
    runtime_client,
    *,
    agent_id: str,
    alias_id: str,
    session_id: str,
    input_text: str,
    shadow_alias_id: Optional[str] = None,
) -> Tuple[dict, dict]:
    """Invoke Bedrock agent with full telemetry.
    
    Returns:
        (agent_output, span_metrics_dict)
        
    When shadow_alias_id is provided:
        - Primary and shadow alias both invoked in parallel
        - Disagreement score and numeric variance captured
        - Risk scores computed from shadow signals
    """

def observe_model_request(
    runtime_client,
    *,
    model_id: str,
    body: str,
    content_type: Optional[str] = None,
    accept: Optional[str] = None,
) -> dict:
    """Invoke Bedrock model with full telemetry.
    
    Returns:
        model_output (response_dict)
    """
```

**Internal Architecture:**

```
┌─ Global _wrapper ────────────────────────────────────────┐
│  _wrapper = instrument_wrapper_api("teamweave-bedrock")   │
│  Singleton per process; reused for all invocations        │
└──────────────────────────────────────────────────────────┘
                           ↓
          ┌─ asyncio.run(_wrapper.invoke(...)) ┐
          │                                      │
          ├─ source: "agent" or "model"         │
          ├─ model: "bedrock-agent" or model_id │
          ├─ prompt: input text                 │
          ├─ call: sync Bedrock invocation      │
          ├─ dual_invoke: boolean (shadow)      │
          └─ shadow_*: shadow alias details     ┘
                           ↓
          ┌─ WrapperResult ──────────────────────┐
          │  .output          (agent/model resp) │
          │  .span            (TraceContext)     │
          │  .decision        (PolicyDecision)   │
          └──────────────────────────────────────┘
                           ↓
          ┌─ _push_metric(operation, span, ...) ┐
          │                                      │
          ├─ Extract span fields                │
          ├─ Enrich risk scores (shadow signal) │
          ├─ DynamoDB put_item (guarded, best-  │
          │  effort; swallows errors)           │
          └─ AMP remote_write (independent)     ┘
```

#### 2. **Data Model — ObservatoryMetrics DynamoDB Table**

**Schema:**

| Field | Type | Notes |
|-------|------|-------|
| `pk` | String | `OBSERVATORY#{operation}` (invoke_agent, invoke_model) |
| `sk` | String | `{iso_timestamp}#{trace_id}` (sortable by time + trace) |
| `trace_id` | String | Unique span identifier from mcp-observatory |
| `operation` | String | "invoke_agent" or "invoke_model" |
| `timestamp` | String | ISO 8601 timestamp |
| `prompt_tokens` | Decimal | Estimated prompt token count |
| `completion_tokens` | Decimal | Estimated completion token count |
| `cost_usd` | Decimal | Computed cost (model pricing) |
| `decision` | String | Policy decision: "allow" or "block" |
| `decision_reason` | String | Reason from policy engine |
| `ttl` | Number | Unix epoch + 90 days (auto-delete) |
| **Primary context** | | |
| `agent_id` | String | Bedrock agent ID |
| `alias_id` | String | Agent alias version |
| `session_id` | String | Agent session ID (state tracking) |
| `model_id` | String | Bedrock model ID (invoke_model only) |
| `input_len` | Decimal | Length of input prompt |
| `body_len` | Decimal | Length of request body (models) |
| **Risk scoring** | | |
| `composite_risk_score` | Decimal | Computed from disagreement + variance |
| `composite_risk_level` | String | "low" \| "medium" \| "high" \| "critical" |
| `hallucination_risk_score` | Decimal | Disagreement-derived hallucination risk |
| `hallucination_risk_level` | String | Risk level (low/medium/high) |
| `risk_tier` | String | Mirrors `composite_risk_level` |
| `gate_blocked` | Boolean | True when policy engine blocks |
| **Shadow signals** | | |
| `shadow_alias_id` | String | Shadow alias invoked (if dual_invoke=true) |
| `shadow_disagreement_score` | Decimal | Output disagreement [0, 1] |
| `shadow_numeric_variance` | Decimal | Numeric instability score |
| **Trace context** | | |
| `span_id` | String | Span identifier from mcp-observatory |
| `parent_span_id` | String | Parent trace ID (for distributed tracing) |
| `self_consistency_risk` | Decimal | Derived from disagreement |
| `numeric_instability_risk` | Decimal | Derived from variance |
| `confidence` | Decimal | Model output confidence |
| `grounding_score` | Decimal | Hallucination detection signal |
| `verifier_score` | Decimal | Output verification signal |
| `fallback_used` | Boolean | Fallback model invoked |
| `fallback_type` | String | Type of fallback (cascade, retry, etc.) |
| `policy_decision` | String | Policy engine raw decision |
| `policy_id` | String | Policy ID that made decision |
| `exec_token_*` | Various | Execution token verification fields |
| `start_time` | String | Span start time (ISO 8601) |
| `end_time` | String | Span end time (ISO 8601) |

#### 3. **observatory_handler.py** — PromQL Query Gateway

**Purpose:** Exposes `GET /observability/metrics` Lambda endpoint to query Amazon Managed Prometheus.

**Endpoints:**

```
GET /observability/metrics?query=<promql>
GET /observability/metrics?query=<promql>&start=<epoch>&end=<epoch>&step=<dur>
GET /observability/metrics?query=<promql>&time=<epoch>
```

**Example Queries:**

```promql
# Agent invocation rate (per minute)
rate(teamweave_bedrock_requests_total{operation="invoke_agent"}[1m])

# Average cost per invocation
avg(teamweave_bedrock_cost_usd) by (operation)

# P95 latency
histogram_quantile(0.95, teamweave_bedrock_request_duration_ms)

# Policy blocks vs allows
count(teamweave_bedrock_policy_block) vs count(teamweave_bedrock_policy_allow)
```

---

## Architecture & Design Decisions

### ADR-1: Shared Singleton Wrapper per Process

**Decision:** One `_wrapper = instrument_wrapper_api("teamweave-bedrock")` per Lambda process, reused for all invocations.

**Rationale:**
- **Connection pooling**: mcp-observatory maintains internal state for distributed tracing context
- **Cost**: Initializing a new wrapper per call is wasteful
- **Correctness**: Parent-child span relationships tracked across calls within a session

**Trade-offs:**
- ✅ Efficient; minimizes initialization overhead
- ✅ Spans inherit proper context (parent_span_id set correctly)
- ⚠️ Process-scoped (not thread-safe for multi-threaded runtimes, but AWS Lambda is single-threaded)

**Adoption Impact:** Use one global wrapper instance per Lambda function.

---

### ADR-2: Dual-Invoke (Shadow Alias) for Quality Signals

**Decision:** When `shadow_alias_id` is provided, both primary and shadow aliases are invoked in parallel; disagreement is captured and used to compute risk scores.

**Rationale:**
- **Model improvement**: Disagreement between primary and shadow (A/B tested) alias reveals model drift
- **Cost**: Shadow invocation happens in parallel; minimal added latency (same duration as primary)
- **Training data**: Disagreement scores can be used for preference learning / DPO ranking
- **Early warning**: Detects quality degradation without waiting for user feedback

**Shadow Signals Captured:**
1. `shadow_disagreement_score` — Semantic or token-level output disagreement [0, 1]
2. `shadow_numeric_variance` — Numeric instability (when outputs are numeric)

**Risk Score Computation:**
```python
self_consistency = 1.0 - shadow_disagreement_score
hallucination_risk = compute_hallucination_risk_score(
    self_consistency_score=self_consistency,
    numeric_variance_score=shadow_numeric_variance,
    ...
)
composite_risk = composite_risk_score({
    "self_consistency_risk": shadow_disagreement_score,
    "numeric_instability_risk": shadow_numeric_variance,
})
```

**Trade-offs:**
- ✅ Low-latency quality signals (no need for user feedback loops)
- ✅ Parallel invocation: shadow cost added but not latency
- ⚠️ Requires deploying shadow alias alongside primary (extra Bedrock version to maintain)
- ⚠️ Shadow invocation costs double (must be justified by team)

**Adoption Impact:** 
- Optional: `shadow_alias_id` is optional; omit if quality signals not needed
- If enabled, budget for 2x Bedrock invocation costs during evaluation phase
- Monitor disagreement baseline; alert on sustained increases

---

### ADR-3: Best-Effort Telemetry Persistence (No Exceptions Bubble Up)

**Decision:** DynamoDB writes are guarded, failures are logged as warnings, never propagate to caller.

**Implementation:**
```python
def _push_metric(...):
    try:
        table.put_item(Item=item)
    except Exception as exc:
        log.warning("observatory_metric_write_failed", extra={"err": str(exc)})
        # ← Never raise; call succeeds regardless
```

**Rationale:**
- **Resilience**: Observability infrastructure must not break production workflows
- **Graceful degradation**: If DynamoDB is throttled, agent invocation still succeeds
- **Operational**: Teams can debug observability issues independently; workflows unaffected

**Trade-offs:**
- ✅ Observability is non-critical path (production succeeds without it)
- ✅ Fail-open: Workflow resilience prioritized over telemetry completeness
- ⚠️ Silent failures: Must monitor DynamoDB throttling, quota usage separately
- ⚠️ Gaps in telemetry possible; metrics may not be 100% complete

**Adoption Impact:**
- Set up CloudWatch alarms for `PutItem` throttling on ObservatoryMetricsTable
- Monitor DynamoDB consumed write capacity; scale provisioning proactively
- Treat telemetry completeness as SLO, not SLI

---

### ADR-4: 90-Day TTL with Auto-Expiration

**Decision:** All telemetry items expire automatically after 90 days (TTL attribute).

**Rationale:**
- **Cost containment**: DynamoDB storage grows unbounded without expiration; TTL is cheap auto-cleanup
- **Compliance**: 90 days aligns with typical data retention policies
- **Query performance**: Smaller table = faster queries; recent data (3 months) is most useful

**Configuration:**
```python
_TTL_SECONDS = 90 * 24 * 60 * 60  # 90 days in seconds
item["ttl"] = int(time.time()) + _TTL_SECONDS
```

**Trade-offs:**
- ✅ Automatic cleanup; no manual maintenance
- ✅ Cost predictable (storage bounded)
- ⚠️ Long-term analytics must export to data warehouse before expiration
- ⚠️ Compliance: Check if 90 days meets your retention policy

**Adoption Impact:**
- If longer retention needed: Change `_TTL_SECONDS` (but budget for storage)
- Export old telemetry to S3/Athena periodically if archival required
- Set up scheduled Lambda to copy items to archive table before TTL triggers

---

### ADR-5: Field Type Mapping — Native Python → Decimal → DynamoDB

**Decision:** All numeric spans fields converted to `Decimal` for DynamoDB; tracked explicitly for each field class.

**Field Categories:**
| Category | Example Fields | Conversion | DynamoDB Type |
|----------|----------------|------------|---|
| Float | `confidence`, `risk_score` | `Decimal(str(round(v, 8)))` | N |
| Int | `retries`, `prompt_size_chars` | `Decimal(int(v))` | N |
| Bool | `fallback_used`, `gate_blocked` | `bool(v)` | BOOL |
| String | `risk_tier`, `policy_id` | `str(v)` | S |
| DateTime | `start_time`, `end_time` | `.isoformat()` | S |

**Rationale:**
- **Precision**: Floats → Decimal avoids floating-point rounding errors (critical for risk scores)
- **Queryability**: DynamoDB native types enable efficient filtering (e.g., `gate_blocked = true`)
- **Consistency**: Explicit field mapping prevents silent type coercion

**Trade-offs:**
- ✅ Correct numeric handling (no floating-point precision loss)
- ✅ DynamoDB native types work with FilterExpression, scanning
- ⚠️ Verbose: Must enumerate all field types (but forwards-compatible)

**Adoption Impact:**
- When adding new span fields: Classify as float/int/bool/string and add to appropriate list
- For new risk metrics: Always use `Decimal` constructor with precision guarantee
- Testing: Verify Decimal conversion in unit tests (see `test_mcp_observatory.py`)

---

### ADR-6: Risk Enrichment Layer — Gate + Hallucination + Composite Risk

**Decision:** Three-layer risk computation in `_enrich_risk_fields`:

1. **gate_blocked** — Policy engine decision
2. **hallucination_risk_*** — Self-consistency from disagreement
3. **composite_risk_*** — Multi-signal risk aggregation

**Computation Flow:**
```python
def _enrich_risk_fields(item, decision):
    # Layer 1: Gate decision
    if decision.action == "block":
        item["gate_blocked"] = True
    
    # Layer 2: Hallucination risk (from shadow disagreement)
    if "shadow_disagreement_score" in item:
        self_consistency = 1.0 - shadow_disagreement
        h_score = compute_hallucination_risk_score(
            self_consistency_score=self_consistency,
            numeric_variance_score=shadow_variance,
        )
        item["hallucination_risk_score"] = h_score
        item["hallucination_risk_level"] = risk_level_for_score(h_score)
    
    # Layer 3: Composite risk (aggregates all signals)
    components = {
        "self_consistency_risk": shadow_disagreement,
        "numeric_instability_risk": shadow_variance,
    }
    c_score, c_level = composite_risk_score(components)
    item["composite_risk_score"] = c_score
    item["composite_risk_level"] = c_level
    item["risk_tier"] = c_level  # Mirror for readability
```

**Rationale:**
- **Layered signals**: Gate (policy), hallucination (semantic), composite (multi-signal)
- **Shadow-derived**: Disagreement acts as proxy for self-consistency
- **Extensible**: New risk signals added by extending `components` dict

**Trade-offs:**
- ✅ Multiple risk perspectives (gate, semantic, quantitative)
- ✅ Shadow signals directly feed hallucination risk computation
- ⚠️ Computation only happens when shadow data present (omitted if `shadow_alias_id=None`)
- ⚠️ Library-dependent: `compute_hallucination_risk_score` from mcp-observatory

**Adoption Impact:**
- Use `composite_risk_level` as primary risk indicator for dashboarding
- Use `hallucination_risk_level` for semantic quality concerns
- Use `gate_blocked` for policy enforcement tracking
- Query example: `SELECT * FROM ObservatoryMetrics WHERE composite_risk_level = 'high'`

---

### ADR-7: Independent AMP Integration (Separate from DynamoDB)

**Decision:** `_push_metric` independently writes to AMP via `amp_metrics.py`, not dependent on DynamoDB success.

**Implementation:**
```python
def _push_metric(operation, span, decision, extra):
    # DynamoDB write (guarded, best-effort)
    if table:
        try:
            table.put_item(Item=item)
        except Exception:
            log.warning(...)  # Swallow
    
    # AMP write (independent)
    if operation == "invoke_agent":
        _amp.record_agent_span(span, decision, extra)  # Never fails caller
```

**Rationale:**
- **Decoupled**: Real-time metrics (AMP) independent of batch storage (DynamoDB)
- **Resilience**: If DynamoDB fails, AMP metrics still recorded (dashboards not blind)
- **Latency**: AMP writes async; don't block on completion

**Trade-offs:**
- ✅ Real-time monitoring independent of DynamoDB health
- ✅ DynamoDB failures don't cascade to AMP
- ⚠️ Two systems to manage (AMP workspace setup, remote_write auth)
- ⚠️ Potential data inconsistency (DynamoDB has 0 items, AMP has metrics)

**Adoption Impact:**
- Set up AMP workspace separately from DynamoDB table
- Monitor both: DynamoDB PutItem throttling + AMP remote_write failures
- Treat AMP as source-of-truth for real-time dashboards; DynamoDB for analytics

---

### ADR-8: No Wrapper Caching or Serialization of Spans

**Decision:** Spans are not cached; every invocation is a fresh `asyncio.run(_wrapper.invoke(...))`.

**Rationale:**
- **Thread safety**: mcp-observatory span state is mutable; caching could cause races
- **Isolation**: Each invocation gets fresh context (trace ID, decision)
- **Simplicity**: No serialization/deserialization logic needed

**Trade-offs:**
- ✅ No concurrency bugs
- ✅ Straightforward: invoke → observe → persist
- ⚠️ Small overhead per call (asyncio event loop creation)
- ⚠️ Lambda cold start impact minimal but non-zero

**Adoption Impact:**
- Lambda performance: Profile `asyncio.run` overhead in warm vs. cold starts
- If latency-critical: Consider caching asyncio loop (but requires careful cleanup)
- Test: Ensure no span state leaks between invocations

---

## Integration Points

### 1. Bedrock Agent Invocation (`bedrock_invoke.py`)

**Before:**
```python
resp = brt.invoke_agent(
    agentId=agent_id,
    agentAliasId=alias_id,
    sessionStateVariables={"var": "value"},
    inputText=input_text,
)
```

**After:**
```python
from .mcp_observatory import observe_agent_request

resp, span_metrics = observe_agent_request(
    brt,
    agent_id=agent_id,
    alias_id=alias_id,
    session_id=session_id,
    input_text=input_text,
    shadow_alias_id=shadow_alias_id,  # optional
)

# Use span_metrics for downstream ranking/scoring
dpo_ranking = rank_by_quality(resp, span_metrics)
```

### 2. Worker Lambda (`worker_handler.py`)

**Context:** Step Functions worker invokes agents per workflow step.

**Usage:**
```python
from .bedrock_invoke import invoke_agent

# Shadow alias resolved from team config
shadow_alias = team_config.get("shadow_alias_id")

output = invoke_agent(
    agent_id=step.agent_id,
    alias_id=step.alias_id,
    session_id=session_id,
    input_text=step.input,
    shadow_alias_id=shadow_alias,  # From team.json
)
```

### 3. DPO Training Data Collection

**Use Case:** Ranking agent responses for preference learning.

```python
# Collect multiple responses with varying shadow signals
responses = []
for i in range(3):
    output, metrics = invoke_agent_with_metrics(...)
    responses.append({
        "output": output,
        "composite_risk_score": metrics["composite_risk_score"],
    })

# Rank by quality (lower risk = higher quality)
ranked = sorted(responses, key=lambda r: r["composite_risk_score"])

# Use for DPO preference pairs
winner = ranked[0]
loser = ranked[-1]
```

### 4. Observability / Metrics Query (`observatory_handler.py`)

**Dashboards:**
```promql
# Agent quality (success rate by risk level)
sum(rate(teamweave_bedrock_requests_total{operation="invoke_agent"}[5m])) by (risk_tier)

# Cost analysis
sum(teamweave_bedrock_cost_usd) by (alias_id)

# Policy decision distribution
count(teamweave_bedrock_policy_block) / count(teamweave_bedrock_policy_allow)
```

---

## Adoption Guide for New Projects

### Step 1: Add mcp-observatory Dependency

**In `requirements.txt`:**
```
mcp-observatory==0.2.0
boto3>=1.28.0
botocore>=1.31.0
```

**Install:**
```bash
pip install -r requirements.txt
```

### Step 2: Create ObservatoryMetrics DynamoDB Table

**Option A: CloudFormation (SAM):**
```yaml
ObservatoryMetricsTable:
  Type: AWS::DynamoDB::Table
  Properties:
    TableName: !Sub "${AWS::StackName}-OBSERVATORY_METRICS"
    BillingMode: PAY_PER_REQUEST
    AttributeDefinitions:
      - AttributeName: pk
        AttributeType: S
      - AttributeName: sk
        AttributeType: S
    KeySchema:
      - AttributeName: pk
        KeyType: HASH
      - AttributeName: sk
        KeyType: RANGE
    TTL:
      AttributeName: ttl
      Enabled: true
    StreamSpecification:
      StreamViewType: NEW_AND_OLD_IMAGES
```

**Option B: AWS CLI:**
```bash
aws dynamodb create-table \
  --table-name myapp-observatory-metrics \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --ttl-specification AttributeName=ttl,Enabled=true
```

### Step 3: Configure Environment Variables

**Lambda environment:**
```
OBSERVATORY_METRICS_TABLE=myapp-observatory-metrics
AMP_WORKSPACE_ID=ws-xxx         # optional, for AMP integration
AMP_REGION=us-east-1             # optional
```

### Step 4: Create mcp_observatory Wrapper Module

**`src/observability.py`:**
```python
"""Observability wrapper for your project."""

from mcp_observatory.instrument import instrument_wrapper_api
import asyncio
import logging

log = logging.getLogger("observability")

# Singleton wrapper per process
_wrapper = instrument_wrapper_api("myapp-bedrock")

def observe_invoke(
    runtime_client,
    source: str,
    model: str,
    prompt: str,
    call_func,
    shadow_call_func=None,
):
    """Wrap any Bedrock invocation with observability."""
    has_shadow = shadow_call_func is not None
    
    result = asyncio.run(
        _wrapper.invoke(
            source=source,
            model=model,
            prompt=prompt,
            input_payload={},
            call=call_func,
            dual_invoke=has_shadow,
            shadow_call=shadow_call_func,
            shadow_source=source if has_shadow else None,
            shadow_model=f"{model}/shadow" if has_shadow else None,
        )
    )
    
    log.info(
        "bedrock_invocation",
        extra={
            "source": source,
            "model": model,
            "trace_id": result.span.trace_id,
            "cost_usd": result.span.cost_usd,
            "decision": result.decision.action,
        }
    )
    
    return result
```

### Step 5: Instrument Bedrock Calls

**Before:**
```python
def invoke_my_model(model_id, prompt):
    return runtime.invoke_model(
        modelId=model_id,
        body=json.dumps({"prompt": prompt}),
    )
```

**After:**
```python
from .observability import observe_invoke

def invoke_my_model(model_id, prompt, shadow_model_id=None):
    call = lambda: runtime.invoke_model(
        modelId=model_id,
        body=json.dumps({"prompt": prompt}),
    )
    
    shadow_call = None
    if shadow_model_id:
        shadow_call = lambda: runtime.invoke_model(
            modelId=shadow_model_id,
            body=json.dumps({"prompt": prompt}),
        )
    
    result = observe_invoke(
        runtime,
        source="model",
        model=model_id,
        prompt=prompt,
        call_func=call,
        shadow_call_func=shadow_call,
    )
    
    return result.output
```

### Step 6: Set Up Amazon Managed Prometheus (Optional)

**Create AMP workspace:**
```bash
aws amp create-workspace --alias myapp-observability --region us-east-1
# Returns: arn:aws:aps:us-east-1:123456789012:workspace/ws-xxx
```

**Configure remote_write in mcp-observatory:**
```python
# In your initialization code
from mcp_observatory.amp import configure_remote_write

configure_remote_write(
    workspace_id="ws-xxx",
    region="us-east-1",
    role_arn="arn:aws:iam::123456789012:role/MyAppLambdaRole",
)
```

### Step 7: Query Telemetry

**DynamoDB Queries:**
```python
import boto3
from decimal import Decimal

ddb = boto3.resource("dynamodb")
table = ddb.Table("myapp-observatory-metrics")

# Recent invocations with high risk
response = table.query(
    KeyConditionExpression="pk = :pk",
    FilterExpression="composite_risk_level = :risk",
    ExpressionAttributeValues={
        ":pk": "OBSERVATORY#invoke_agent",
        ":risk": "high",
    },
    Limit=100,
    ScanIndexForward=False,  # Most recent first
)

for item in response["Items"]:
    print(f"Trace: {item['trace_id']}, Risk: {item['composite_risk_level']}, Cost: ${item['cost_usd']}")
```

**PromQL Queries (if using AMP):**
```promql
# Average risk score by agent
avg(teamweave_bedrock_risk_score) by (agent_id)

# Policy block rate
rate(teamweave_bedrock_policy_decision_total{decision="block"}[5m])

# Cost trend
sum(teamweave_bedrock_cost_usd) by (alias_id) 
```

### Step 8: Testing

**Unit test template:**
```python
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from src.observability import observe_invoke

class ObserveInvokeTests(unittest.TestCase):
    def test_observe_invoke_returns_wrapper_output(self):
        mock_result = MagicMock()
        mock_result.output = {"text": "hello"}
        mock_result.span.trace_id = "t1"
        mock_result.decision.action = "allow"
        
        with patch("src.observability._wrapper.invoke", new=AsyncMock(return_value=mock_result)):
            result = observe_invoke(
                MagicMock(),
                source="model",
                model="amazon.nova-micro",
                prompt="test",
                call_func=lambda: {"text": "hello"},
            )
        
        self.assertEqual(result.output, {"text": "hello"})
```

---

## Operational Considerations

### Monitoring & Alerting

**Critical metrics:**

| Metric | Threshold | Action |
|--------|-----------|--------|
| DynamoDB `ConsumedWriteCapacity` | > 80% provisioned | Scale table or enable autoscaling |
| `PutItem` throttle errors | > 0 | Investigate; swallowed but logged |
| AMP `remote_write` errors | > 0 in 5m | Check workspace, credentials, network |
| Span processing latency | > 500ms | Profile `asyncio.run` overhead |
| Policy `block` rate | > 5% | Review risk thresholds |

**CloudWatch alarms:**
```yaml
ObservatoryMetricsThrottleAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    MetricName: UserErrors
    Namespace: AWS/DynamoDB
    Dimensions:
      - Name: TableName
        Value: !Ref ObservatoryMetricsTable
    Statistic: Sum
    Period: 300
    EvaluationPeriods: 1
    Threshold: 10
    ComparisonOperator: GreaterThanOrEqualToThreshold
    AlarmActions:
      - !Ref AlertSNSTopic
```

### Cost Estimation

**DynamoDB (PAY_PER_REQUEST):**
- Write: $1.25 per 1M items written
- Storage: $0.25 per GB-month
- TTL cleanup: Free

**Estimate:** 100 agent invocations/day × 30 days × 0.0001KB avg item ≈ 300KB stored ≈ $0.08/month

**Bedrock (dual-invoke):**
- Shadow invocation doubles Bedrock costs during shadow testing
- Baseline: $0.80 per 1M input tokens (varies by model)

**Recommendation:** Start with dual-invoke on small % of traffic; expand after confidence.

### Data Retention & Archival

**90-day TTL auto-expires items.** For longer retention:

**Export before expiration:**
```python
import boto3
from datetime import datetime, timedelta

def archive_old_telemetry():
    ddb = boto3.resource("dynamodb")
    s3 = boto3.client("s3")
    table = ddb.Table("observatory-metrics")
    
    cutoff = int((datetime.now() - timedelta(days=85)).timestamp())
    
    response = table.scan(
        FilterExpression="attribute_not_exists(ttl) OR #t < :cutoff",
        ExpressionAttributeNames={"#t": "ttl"},
        ExpressionAttributeValues={":cutoff": cutoff},
    )
    
    # Export to S3 in JSON Lines format
    for item in response["Items"]:
        s3.put_object(
            Bucket="myapp-archives",
            Key=f"observatory/{item['sk']}.json",
            Body=json.dumps(item),
        )
```

---

## Troubleshooting

### Issue: No DynamoDB Items Written

**Symptoms:** `OBSERVATORY_METRICS_TABLE` is empty despite Bedrock calls.

**Diagnosis:**
```bash
# Check environment variable
echo $OBSERVATORY_METRICS_TABLE

# Verify table exists
aws dynamodb describe-table --table-name $OBSERVATORY_METRICS_TABLE

# Check Lambda logs for warnings
aws logs tail /aws/lambda/myapp-worker --follow | grep "observatory_metric_write_failed"
```

**Solutions:**
1. Ensure `OBSERVATORY_METRICS_TABLE` env var is set
2. Grant Lambda IAM permission: `dynamodb:PutItem` on table ARN
3. Check DynamoDB table billing mode (PAY_PER_REQUEST recommended)

### Issue: High Latency on Agent Invocations

**Symptoms:** `asyncio.run(_wrapper.invoke)` adding 200ms+ per call.

**Diagnosis:**
```python
import time

start = time.time()
result = asyncio.run(_wrapper.invoke(...))
elapsed = time.time() - start
log.info(f"wrapper_invoke_ms={elapsed*1000:.0f}")
```

**Solutions:**
1. **Cold start**: First invocation creates asyncio event loop; warm requests faster
2. **Dual-invoke**: If shadow enabled, latency = `max(primary, shadow)` (parallel execution)
3. **Profile mcp-observatory**: Add tracing inside library initialization

### Issue: DynamoDB Throttling (ConsumedWriteCapacity > Provisioned)

**Symptoms:** CloudWatch shows throttled `PutItem` operations.

**Diagnosis:**
```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/DynamoDB \
  --metric-name ConsumedWriteCapacityUnits \
  --dimensions Name=TableName,Value=observatory-metrics \
  --start-time 2024-04-01T00:00:00Z \
  --end-time 2024-04-28T00:00:00Z \
  --period 300 \
  --statistics Sum
```

**Solutions:**
1. Switch to PAY_PER_REQUEST billing (auto-scales)
2. If provisioned: Increase write capacity
3. Filter out low-value items: Skip logging for trivial invocations

### Issue: Risk Scores All Zero

**Symptoms:** `composite_risk_score = 0` for all items, even with shadow disagreement.

**Diagnosis:**
```python
# Check if shadow data is present
response = table.scan(
    FilterExpression="attribute_exists(shadow_disagreement_score)",
    Limit=10,
)

if not response["Items"]:
    print("No shadow signals; risk enrichment skipped")
```

**Solutions:**
1. Verify `shadow_alias_id` passed to `observe_agent_request`
2. Confirm shadow alias exists in Bedrock: `aws bedrock-agent list-agent-aliases --agent-id <id>`
3. Check mcp-observatory logs for shadow invocation failures

---

## Summary: Key Takeaways

| Aspect | Decision | Impact |
|--------|----------|--------|
| **Wrapper** | Singleton per process | Efficient; proper span context |
| **Dual-invoke** | Optional shadow alias | Quality signals; 2x Bedrock cost |
| **Persistence** | Best-effort; swallows errors | Resilient; no production impact |
| **Storage** | DynamoDB + 90-day TTL | Cost-bounded; auto-cleanup |
| **Risk scoring** | Three-layer enrichment | Gate + hallucination + composite |
| **AMP** | Independent from DynamoDB | Real-time dashboards separate |
| **Type mapping** | Decimal for all numerics | No precision loss |

**Adoption Steps:**
1. Add `mcp-observatory` to requirements
2. Create ObservatoryMetrics DynamoDB table
3. Set `OBSERVATORY_METRICS_TABLE` env var
4. Wrap Bedrock calls with `observe_agent_request` / `observe_model_request`
5. Optional: Configure AMP for real-time metrics
6. Query telemetry via PromQL or DynamoDB Scan/Query

---

**For questions or improvements, please refer to the test suite at `tests/test_mcp_observatory.py` (750+ lines, comprehensive coverage).**
