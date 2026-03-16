Trigger a TeamWeave pipeline run for testing.

Steps:
1. Get the ApiEndpoint from CloudFormation outputs:
   `aws cloudformation describe-stacks --stack-name tarun-content-team --query "Stacks[0].Outputs"`
2. Send a test invocation:
   `curl -X POST <ApiEndpoint>/team/task -H "Content-Type: application/json" -d '{"team": "tarun_visibility_team", "version": "v1", "request": {"topic": "$ARGUMENTS"}}'`
3. Poll /team/task/<run_id> until status == "SUCCEEDED"
4. Print the S3 artifact paths from the response
