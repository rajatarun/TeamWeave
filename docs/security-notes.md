# Security notes (threat model sketch)

This is a high-level view for adopters. It is not a formal audit.

## Assets

- Team configs and schemas (S3)
- Run metadata and step outputs (DynamoDB; artifacts in S3)
- API keys and DB credentials (Secrets Manager)
- Optional end-user health records (ContextWeave, caller-bound tokens for `health_prep` only)
- CipherWeave client credentials for sibling service calls (per-sibling secrets)

## Trust boundaries

| Boundary | What crosses it |
|----------|-----------------|
| Client → API Gateway | SIWE JWT on protected routes; agent card is unauthenticated |
| Worker → Bedrock / AgentCore | IAM role; model prompts include RAG and prior-step outputs |
| Worker → sibling MCP / HTTP APIs | Optional CipherWeave bearer tokens; DeviceWeave may use a separate secret |
| Step Functions execution history | May retain request payloads (including short-lived bearer tokens for health tools — see `CLAUDE.md` / `health_prep` design) |

## Main risks

1. **Prompt injection via RAG or crawled content** — retrieved text and tool results are embedded in prompts. Use `min_confidence`, grounding instructions, and schema validation; treat outputs as untrusted until reviewed.
2. **Over-privileged IAM** — deploy templates grant broad Bedrock and network access. Scope roles per environment.
3. **Secrets in team configs** — never store credentials in `team.json`; use Secrets Manager and env vars wired by SAM.
4. **Sibling token plumbing** — the worker can mint OAuth-style client credentials for configured siblings. Misconfigured secrets send unauthenticated calls (siblings may reject). Details live in code (`cipherweave_tokens.py`); treat as operational auth, not end-user session design.
5. **Device control** — `device_controller` can propose destructive actions; confirmation gates exist for risky commands.

## What this repo does not provide

- Per-call argument signing or custom authorization-token schemes beyond standard AWS IAM and configured HTTP bearer tokens.
- Multi-tenant isolation beyond your AWS account boundaries and config bucket layout.

Report vulnerabilities per [SECURITY.md](../SECURITY.md).
