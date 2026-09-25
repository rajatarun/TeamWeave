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

This account can call Claude Sonnet 4.6. It cannot call Claude Sonnet 5, and
Claude Opus 5 is not a fallback. Both ids are listed under `unavailable` in
the map, and `resolve_model` refuses them, including as an override.

Sonnet 4.6 is invoked with the US geo cross-region inference profile. The
[model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html)
names that id for Geo US, with us-east-1 as a source region. Anthropic's
base on-demand price is $3 input and $15 output per million tokens. Starting
with Claude Sonnet 4.5 and all later models, a Bedrock regional endpoint
(the `us.` profile) adds a 10% premium, so the stored price is **$3.30 /
$16.50**.

| Source | What it prices |
|---|---|
| [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) | Sonnet 4.6 base $3 / $15, and the 10% regional premium. Checked 2026-09-25. |
| [Claude on Amazon Bedrock](https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy) | Regional endpoints (the `us.` prefix) cost 10% more than global. Checked 2026-09-25. |
| [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) | DeepSeek V3.2 in N. Virginia, checked 2026-09-25 |
| [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) | Gemini, checked 2026-09-25 |
| [Claude Sonnet 4.6 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html) | Geo US id `us.anthropic.claude-sonnet-4-6` |
| [Claude Haiku 4.5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-4-5.html) | Geo id `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| [DeepSeek V3.2 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html) | In-region id `deepseek.v3.2` |

## Models

| Model | Input / 1M | Output / 1M | Cost tier | Latency | Why it is here |
|---|---:|---:|---|---|---|
| `us.anthropic.claude-sonnet-4-6` | $3.30 | $16.50 | standard | fast | The quality model this account can call. US geo profile, base $3/$15 plus 10%. |
| `us.anthropic.claude-haiku-4-5-20251001-v1:0` | $1 | $5 | low | fastest | Near-frontier Anthropic model. First fallback wherever instruction-following matters. |
| `deepseek.v3.2` | $0.62 | $1.85 | low | medium | Best Bedrock quality-for-cost in this map. Default and analysis. |
| `gemini-3.8-flash` | $0.75 | $3.75 | low | fast | Introductory price through 2026-12-31, then $1.50 / $7.50. Research, vision, and a last fallback. |
| `gemini-3.1-flash-lite` | $0.25 | $1.50 | low | fastest | Cheapest text model here. Summaries, chat, and the quick path. |
| `gemini-3.1-flash-lite-image` | $0.25 | $30 image tokens | low | fastest | About $0.0336 per 1K image. The illustrator. |
| `amazon.nova-2-multimodal-embeddings-v1:0` | — | — | infra | fast | Knowledge-base embeddings. The pricing page did not list a per-token rate, so estimated invocation cost stays 0. |

Cost tiers: **low** is the cheap band (Haiku, DeepSeek, Gemini), **standard** is Sonnet 4.6, **infra** is an embedding with no published token rate. Claude Sonnet 5 and Claude Opus 5 are not in the catalogue.

## Categories

Quality-for-cost picks the primary. Sonnet 4.6 is reserved for coding, planning, health, and finance, where a wrong answer costs more than the tokens. Fallbacks on those categories are Haiku 4.5, DeepSeek V3.2, and Gemini 3.8 Flash. Everything else starts on Haiku, DeepSeek, or Gemini Flash-Lite.

| Category | Primary | Cost tier | Latency | Fallbacks | Why |
|---|---|---|---|---|---|
| coding | Sonnet 4.6 | standard | fast | Haiku 4.5, DeepSeek V3.2, Gemini 3.8 Flash | Coding is worth the Sonnet price this account can call, $3.30/$16.50. |
| planning | Sonnet 4.6 | standard | fast | Haiku 4.5, DeepSeek V3.2, Gemini 3.8 Flash | Planning is one of the categories where quality is worth paying for. |
| research_web | Gemini 3.8 Flash | low | fast | DeepSeek V3.2, Haiku 4.5 | Already the research model, at the introductory $0.75/$3.75 through 2026-12-31. |
| finance | Sonnet 4.6 | standard | fast | Haiku 4.5, DeepSeek V3.2, Gemini 3.8 Flash | A fabricated holding is worse than a slower answer. |
| health_medical | Sonnet 4.6 | standard | fast | Haiku 4.5, DeepSeek V3.2, Gemini 3.8 Flash | Safety rules have to be followed. Haiku is the first fallback so a throttle still lands on an instruction-following model. |
| writing_creative | Haiku 4.5 | low | fastest | Sonnet 4.6, DeepSeek V3.2 | Publishable prose at $1/$5. Sonnet 4.6 is the fallback, not the model every draft pays for. |
| summarization | Gemini 3.1 Flash-Lite | low | fastest | DeepSeek V3.2, Haiku 4.5 | $0.25/$1.50. A summary does not need a frontier model. |
| data_analysis | DeepSeek V3.2 | low | medium | Haiku 4.5, Sonnet 4.6 | $0.62/$1.85, about a fifth of Sonnet 4.6's US geo input price. Sonnet 4.6 is the last fallback. |
| customer_support | Gemini 3.1 Flash-Lite | low | fastest | Haiku 4.5, DeepSeek V3.2 | Lowest latency that still follows instructions. |
| quick | Gemini 3.1 Flash-Lite | low | fastest | DeepSeek V3.2, Haiku 4.5 | The cheap path. Bedrock fallbacks keep the turn alive without a Gemini key. |
| multimodal_vision | Gemini 3.8 Flash | low | fast | Sonnet 4.6, Haiku 4.5 | Accepts images at the introductory Gemini price. Sonnet 4.6 also accepts images. No team uses this category yet. |
| image_generation | Gemini 3.1 Flash-Lite Image | low | fastest | — | Nova Canvas reaches end of life on 30 Sep 2026. A failure degrades the step. |
| embeddings | Nova multimodal embeddings | infra | fast | — | Health and portfolio knowledge bases. Estimated cost stays 0 until a rate is written on the model record. |
| schema_repair | Haiku 4.5 | low | fastest | Sonnet 4.6 | Reshapes an answer into the step schema through Bedrock Converse, so the fallback stays a Bedrock id. |
| default | DeepSeek V3.2 | low | medium | Haiku 4.5, Sonnet 4.6 | An unknown category resolves here and logs a warning. |

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

- `us.anthropic.claude-sonnet-4-6`
- `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- `deepseek.v3.2`

Gemini uses the existing `GEMINI_SECRET_ARN`. Claude Sonnet 5 and Claude Opus 5 are not in the catalogue. A missing grant or a throttle advances the category's fallback list. The worker role grants `bedrock:InvokeModel` on `inference-profile/*`, so the template does not name a per-model ARN.
