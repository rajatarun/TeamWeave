# Infra

## Deploy
```bash
cd infra
sam build -t template.yaml
sam deploy --guided
```

If `ConfigBucketName` and `ArtifactBucketName` are not provided during deploy, the stack now creates and wires S3 buckets automatically.

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
