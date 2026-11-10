# Integration opportunities

TeamWeave is a **config-driven pipeline orchestrator** on AWS (Step Functions +
Lambda + Bedrock AgentCore). The table below lists ecosystems where an adapter
would help adoption. Effort is rough engineering time for a production-quality
integration, not calendar time.

| Integration | Fit | Status today | Adapter shape | Effort |
|-------------|-----|--------------|---------------|--------|
| **A2A (Agent2Agent)** | High | **Implemented** | HTTP+JSON card at `/.well-known/agent-card.json`; `message:send` starts the same runs as `POST /team/task`. See `src/orchestrator/a2a.py`, `examples/generate_a2a_card.py`. | Done |
| **MCP (Model Context Protocol)** | High | **Partial** | Worker calls sibling MCP servers via `mcp_client.py`; AgentCore Gateway targets are provisioned for ScreenWeave, CipherWeave, DataDictionary, ToolWeave when endpoints resolve at deploy. Teams cannot invoke MCP mid-turn yet — pre-tools only. Adapter: expose TeamWeave as an MCP server listing teams/skills and starting runs. | Medium |
| **OpenTelemetry / GenAI semconv** | High | **Implemented** | Spans via MCP Observatory wrapper (`mcp_observatory`); `gen_ai.*` attributes alongside legacy keys. Adapter: export OTLP from Lambda env / sidecar; document collector setup. | Small (docs) |
| **Amazon Bedrock / AgentCore** | High | **Implemented** | Default runtime substrate (`AGENT_RUNTIME=agentcore`); per-team runtimes in CloudFormation. Adapter for other accounts: parameterize stack outputs in `scripts/stack_env.py`. | Done (AWS-specific) |
| **LangGraph / LangChain** | Medium | Not implemented | Map each `workflow` step to a LangGraph node; `team.json` → graph compiler; worker becomes optional. Good for local/dev graphs; production still needs persistence and tool gateway. | Large |
| **OpenAI / Anthropic SDKs** | Medium | Not implemented | Today all text turns go through Bedrock Converse on AgentCore. Adapter: alternate `agent_runtime` implementation calling vendor APIs directly (swap `AgentCoreRuntime` in `agent_runtime.py`). | Medium |
| **ContextWeave / RAG backends** | Medium | **Partial** | `rag.py` dispatches `contextweave`, `explicit` (pgvector), `history`. Adapter interface: implement `build_rag_context` branch + env URL. | Small–medium |
| **SIWE wallet auth** | Medium | **Implemented** | API Gateway default authorizer; A2A documents bearer JWT. External clients reuse the same tokens for `POST /team/task`. | Done |
| **Gemini (search / image)** | Medium | **Partial** | `gemini.py`, `web_search` tool, `gemini_image` for illustrator steps — optional secrets. | Done where configured |
| **AWS CDK / Terraform consumers** | Low | Not implemented | SAM template is source of truth (`infra/template.yaml`). Adapter: wrap SAM outputs as CDK constructs or Terraform modules. | Large |

## Highest-value next adapters

1. **MCP server facade** — lets Claude Desktop / other MCP clients discover teams and start runs without custom HTTP glue. Fit: **high**; not built.
2. **LangGraph export (dev-only)** — compile `team.json` to a local graph for fast iteration; Fit: **medium**; not built.
3. **OTLP cookbook** — document wiring existing spans to a collector (behavior exists; packaging is docs). Fit: **high**; mostly documentation.

## Related code paths

- HTTP API contract: [`openapi/teamweave.yaml`](../openapi/teamweave.yaml)
- Pre/post tools (deterministic, not LLM-chosen): [`src/orchestrator/tool_registry.py`](../src/orchestrator/tool_registry.py)
- A2A: [`src/orchestrator/a2a_handler.py`](../src/orchestrator/a2a_handler.py)
