# Portfolio uploads and live web search

`financial_advisors` reads two sources before it writes. The portfolio
knowledge base is the statements in this stack's bucket. The web search is
Gemini Google Search grounding, and every cited page carries its URL and the
UTC date it was retrieved.

The health bucket is different. ContextWeave publishes that bucket, and this
stack only indexes it. There is no ContextWeave portfolio bucket, so
TeamWeave creates one.

## Upload statements

After the stack exists, `PortfolioBucketName` is the bucket. CSV, PDF, and
brokerage statements all go in as objects. The knowledge base reads the
bucket; it does not need a particular prefix.

```bash
eval "$(python scripts/stack_env.py --stack tarun-content-team --format sh)"
aws s3 cp ./statement.pdf "s3://${TEAMWEAVE_PORTFOLIO_BUCKET}/statement.pdf"
aws s3 cp ./positions.csv "s3://${TEAMWEAVE_PORTFOLIO_BUCKET}/positions.csv"
```

`scripts/stack_env.py` maps that name from the `PortfolioBucketName` output.
The bucket is SSE-S3 encrypted, versioned, and blocked from public access.
The name is `${AWS::StackName}-portfolios-${AWS::AccountId}`. The stack name
has to stay lowercase, and the full name has to stay within 63 characters.

The first deploy starts an ingestion job for objects already in the bucket.
A later upload or delete sends an EventBridge event onto an SQS queue. The
sync function runs with reserved concurrency 1, reads up to 100 messages
gathered over 60 seconds, and starts one ingestion job for the batch. If a
job is already running, or `StartIngestionJob` is throttled, those messages
return to the queue and are tried again after the visibility timeout
(6 minutes). Ingestion is asynchronous. A run that starts before the job
finishes sees `found: false` or `error` on `query_portfolio`, and the
agent is told to say what was searched rather than invent holdings.

A burst of uploads that failed before this queue existed is not replayed.
After deploying the queue, confirm the deploy log started an ingestion job.
If that step warned, start one job so the bucket is read as it is now:

```bash
aws bedrock-agent start-ingestion-job \
  --region us-east-1 \
  --knowledge-base-id "$TEAMWEAVE_PORTFOLIO_KNOWLEDGE_BASE_ID" \
  --data-source-id "$TEAMWEAVE_PORTFOLIO_DATA_SOURCE_ID"
```

The health base uses the same sync path. If its lambda was failing the same
way, start one job with `HealthKnowledgeBaseId` and `HealthDataSourceId`.

## The knowledge base

Same shape as the health base, owned here:

| Setting | Value |
|---|---|
| Type | Managed knowledge base (`health_kb_provision.handler`) |
| Embedding | `amazon.nova-2-multimodal-embeddings-v1:0` |
| Embedding type | `CUSTOM` |
| Dimensions | 1024 |
| Data type | `FLOAT32` |
| Data source | `portfolio-s3` on `PortfolioBucket` |
| Supplemental storage | `${AWS::StackName}-portfolio-mm-${AWS::AccountId}` |

Supplemental storage holds media the model extracts during ingestion. It is
a separate bucket because the health multimodal bucket exists only when
ContextWeave's health outputs are present, and the API takes a bucket URI,
not a prefix. Do not upload statements there. The knowledge base role can
read the statement bucket and can write only the supplemental bucket.

Outputs `PortfolioKnowledgeBaseId` and `PortfolioDataSourceId` are what the
worker and the sync function read (`PORTFOLIO_KNOWLEDGE_BASE_ID`,
`PORTFOLIO_DATA_SOURCE_ID`). The worker is granted `bedrock:Retrieve` on
that base's ARN.

Nova model access is granted in the Bedrock console, per model. Without it,
`CreateKnowledgeBase` can still succeed and the ingestion job fails. The
deploy step warns and does not roll the stack back. Creating the base is
unconditional: unlike health, an empty ContextWeave import does not skip it,
and a failed create fails the deploy.

## Web search secret

The tool is `web_search`. The provider is `WEB_SEARCH_PROVIDER=gemini`, which
calls Gemini `generateContent` with `tools: [{"google_search": {}}]` and
keeps `groundingChunks[].web.uri` plus `retrieved_on`.

Set the key in Secrets Manager. The secret string is either

```json
{"key": "the Gemini API key"}
```

or the raw key. Pass that secret's ARN as the stack parameter
`GeminiSecretArn`. The worker already reads it as `GEMINI_SECRET_ARN`; the
research brief uses the same secret.

To use a different secret for search than for the research brief, pass
`WebSearchSecretArn`. The worker receives it as `WEB_SEARCH_SECRET_ARN` and
the role is granted `secretsmanager:GetSecretValue` on it. Leave the
parameter empty to keep using `GeminiSecretArn`.

No key is written into the template, the team config, or the logs. With
neither secret set, `web_search` returns an error that names
`GEMINI_SECRET_ARN` and `GeminiSecretArn`, and the agent must not invent
prices, rates, or headlines.

`WEB_SEARCH_PROVIDER` set to anything other than `gemini` is an error. The
module is the place a second provider would be added; a typo must not look
like an empty market.

## Before a deploy that should answer

1. Grant `amazon.nova-2-multimodal-embeddings-v1:0` in the Bedrock console
   (the same grant the health base needs).
2. Set `GeminiSecretArn` if the insights should cite live pages. The
   parameter can stay empty; the team then runs with portfolio retrieval
   only and reports the web lookup as an error.
3. Deploy. If `CreateKnowledgeBase` fails, the stack rolls back. An
   ingestion failure after that is a warning in the deploy log.
4. Upload statements to `PortfolioBucketName` and wait until the ingestion
   job finishes before expecting excerpts.
