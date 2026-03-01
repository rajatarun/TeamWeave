# Infra

## Deploy
```bash
cd infra
sam build -t template.yaml
sam deploy --guided
```

If `ConfigBucketName` and `ArtifactBucketName` are not provided during deploy, the stack now creates and wires S3 buckets automatically.

The stack also creates a Bedrock Guardrail (`AWS::Bedrock::Guardrail`) with:
- Topic filtering (harmful content, personal information, inappropriate content)
- Content filtering (sexual, violence, hate, insults, misconduct, prompt attack)
- Word filtering (custom sensitive words + managed profanity list)
- Sensitive info protections (PII blocking/anonymization + API key regex blocking)

You can override guardrail metadata during deploy with:
- `GuardrailName`
- `GuardrailDescription`

## Upload example config (both teams)
```bash
CONFIG_BUCKET="<your-config-bucket>"
aws s3 cp ../config/examples/ s3://$CONFIG_BUCKET/ --recursive
```

## Useful calls
POST /team/task
GET /improve/tasks?owner=Tarun%20Raja&limit=50
POST /improve/task/done   body: {"owner":"Tarun Raja","task_id":"..."}


## Gemini research
Create a Secrets Manager secret with JSON like {"key":"<GEMINI_API_KEY>"} and pass its ARN as GeminiSecretArn.

## Explicit RAG with PostgreSQL/pgvector
When `globals.rag.mode` is `explicit`, configure these deploy parameters so Lambda can connect:
- `VectorDbTable`
- `VectorDbHost`
- `VectorDbPort` (default `5432`)
- `VectorDbName`
- `VectorDbUser`
- `VectorDbPassword`
- `VectorDbUrl` (optional override, full connection URL)

If `VectorDbUrl` is set, the app parses it into host/port/name/user/password, then concatenates them into a normalized URL like `postgresql://<user>:<password>@<host>:<port>/<db>` before connecting.

The runtime reads the same values from env vars `VECTOR_DB_*`.
