# Model map

`config/model_map.yaml` is the one place a model id is chosen. A team config
names a `model_category`. `resolve_model` is what a turn calls. An explicit
`model_id` on an agent is an override: it is logged, and the category's
fallbacks still follow it.

Prices below are us-east-1 on-demand, per 1 million tokens, checked
**2026-09-25**. Claude rates are the provider rates Bedrock bills. DeepSeek
is the rate on the AWS Bedrock pricing page for N. Virginia. Gemini rates
are the Gemini API paid tier. Nova token prices did not appear on the
fetched Bedrock pricing page, so Nova text models are not in this map.

| Source | What it prices |
|---|---|
| [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) | Claude, checked 2026-09-25 |
| [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) | DeepSeek V3.2 in N. Virginia, checked 2026-09-25 |
| [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) | Gemini, checked 2026-09-25 |
| [Claude Sonnet 5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html) | Geo id `us.anthropic.claude-sonnet-5` |
| [Claude Haiku 4.5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-4-5.html) | Geo id `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| [Claude Opus 5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html) | Geo id `us.anthropic.claude-opus-5` |
| [Claude Sonnet 4.6 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html) | Geo id `us.anthropic.claude-sonnet-4-6` |
| [DeepSeek V3.2 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html) | In-region id `deepseek.v3.2` |

## Models

| Model | Input / 1M | Output / 1M | Cost tier | Latency | Why it is here |
|---|---:|---:|---|---|---|
| `us.anthropic.claude-sonnet-5` | $2 | $10 | standard | fast | Current coding Sonnet. Near-Opus, cheaper than Sonnet 4.6 and Opus 5. |
| `us.anthropic.claude-sonnet-4-6` | $3 | $15 | standard | fast | In the catalogue so the comparison is recorded. More expensive than Sonnet 5, so it is not a primary. |
| `us.anthropic.claude-haiku-4-5-20251001-v1:0` | $1 | $5 | low | fastest | Near-frontier Anthropic model. First fallback wherever instruction-following matters. |
| `us.anthropic.claude-opus-5` | $5 | $25 | high | medium | In the catalogue. 2.5× Sonnet 5. Not a primary. |
| `deepseek.v3.2` | $0.62 | $1.85 | low | medium | Best Bedrock quality-for-cost in this map. Default and analysis. |
| `gemini-3.8-flash` | $0.75 | $3.75 | low | fast | Introductory price through 2026-12-31, then $1.50 / $7.50. Research and vision. |
| `gemini-3.1-flash-lite` | $0.25 | $1.50 | low | fastest | Cheapest text model here. Summaries, chat, and the quick path. |
| `gemini-3.1-flash-lite-image` | $0.25 | $30 image tokens | low | fastest | About $0.0336 per 1K image. The illustrator. |
| `amazon.nova-2-multimodal-embeddings-v1:0` | — | — | infra | fast | Knowledge-base embeddings. The pricing page did not list a per-token rate, so estimated invocation cost stays 0. |

Cost tiers: **low** is the cheap band (Haiku, DeepSeek, Gemini), **standard** is Sonnet 5, **high** is Opus 5, **infra** is an embedding with no published token rate.

## Categories

Quality-for-cost picks the primary. Sonnet 5 is reserved for coding, planning, health, and finance, where a wrong answer costs more than the tokens. Opus 5 is not used: Sonnet 5 is the coding model and costs less than half as much. Everything else starts on Haiku, DeepSeek, or Gemini Flash-Lite.

| Category | Primary | Cost tier | Latency | Fallbacks | Why |
|---|---|---|---|---|---|
| coding | Sonnet 5 | standard | fast | Haiku 4.5, DeepSeek V3.2 | Coding is worth the Sonnet price. Opus is not: Sonnet 5 is the coding model at $2/$10 versus Opus at $5/$25. |
| planning | Sonnet 5 | standard | fast | Haiku 4.5, DeepSeek V3.2 | Planning is one of the categories where quality is worth paying for, still at the Sonnet 5 price rather than Opus. |
| research_web | Gemini 3.8 Flash | low | fast | DeepSeek V3.2, Haiku 4.5 | Already the research model, at the introductory $0.75/$3.75 through 2026-12-31. |
| finance | Sonnet 5 | standard | fast | Haiku 4.5, DeepSeek V3.2 | A fabricated holding is worse than a slower answer. Sonnet 5 is still cheaper than Sonnet 4.6 and Opus 5. |
| health_medical | Sonnet 5 | standard | fast | Haiku 4.5, DeepSeek V3.2 | Safety rules have to be followed. Haiku is the first fallback so a throttle still lands on an instruction-following model. |
| writing_creative | Haiku 4.5 | low | fastest | Sonnet 5, DeepSeek V3.2 | Publishable prose at $1/$5. Sonnet is the fallback, not the model every draft pays for. |
| summarization | Gemini 3.1 Flash-Lite | low | fastest | DeepSeek V3.2, Haiku 4.5 | $0.25/$1.50. A summary does not need a frontier model. |
| data_analysis | DeepSeek V3.2 | low | medium | Haiku 4.5, Sonnet 5 | $0.62/$1.85, about a third of Sonnet 5's input price. Sonnet is the last fallback. |
| customer_support | Gemini 3.1 Flash-Lite | low | fastest | Haiku 4.5, DeepSeek V3.2 | Lowest latency that still follows instructions. |
| quick | Gemini 3.1 Flash-Lite | low | fastest | DeepSeek V3.2, Haiku 4.5 | The cheap path. Bedrock fallbacks keep the turn alive without a Gemini key. |
| multimodal_vision | Gemini 3.8 Flash | low | fast | Sonnet 5, Haiku 4.5 | Accepts images at the introductory Gemini price. No team uses this category yet. |
| image_generation | Gemini 3.1 Flash-Lite Image | low | fastest | — | Nova Canvas reaches end of life on 30 Sep 2026. A failure degrades the step. |
| embeddings | Nova multimodal embeddings | infra | fast | — | Health and portfolio knowledge bases. Estimated cost stays 0 until a rate is written on the model record. |
| schema_repair | Haiku 4.5 | low | fastest | Sonnet 5 | Reshapes an answer into the step schema. Converse, so the id can move without a provider-shaped body. |
| default | DeepSeek V3.2 | low | medium | Haiku 4.5, Sonnet 5 | An unknown category resolves here and logs a warning. |

## Who uses which category

| Team | Agent | Category |
|---|---|---|
| tarun_visibility_team | director, strategist | planning |
| tarun_visibility_team | writer, editor | writing_creative |
| tarun_visibility_team | illustrator | image_generation |
| linkedin_quick_post | angle | planning |
| linkedin_quick_post | writer | writing_creative |
| job_hunter | reader | summarization |
| job_hunter | fit | data_analysis |
| job_hunter | writer | writing_creative |
| daily_operator | triage, plan | planning |
| health_prep | both | health_medical |
| financial_advisors | both | finance |

## Observatory

Every selection goes through `emit_model_event`, which calls
`record_model_selection` in `mcp_observatory`. That writes a
`model_selection` row on the existing Observatory metrics table. The row
carries the agent, team, run id, category, model, whether a fallback was
used, latency, token counts, success or error, a result-quality signal when
the worker has one, the cost tier, and `estimated_cost_usd`.

Estimated cost is `(prompt tokens / 1e6) * cost_per_1m_input + (completion tokens / 1e6) * cost_per_1m_output`. An image call with no output tokens uses `cost_per_image`. An id that is not in the map estimates 0. A down observatory is logged and does not fail the turn. There is no new endpoint: the table the agent spans already use is the sink.

## What to enable

In the Bedrock console for us-east-1, enable on-demand access for:

- `us.anthropic.claude-sonnet-5`
- `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- `deepseek.v3.2`

Gemini uses the existing `GEMINI_SECRET_ARN`. Opus 5 and Sonnet 4.6 are in the catalogue and are not primaries. A missing grant or a throttle advances the category's fallback list.
