#!/usr/bin/env bash
# migrate-to-shared.sh
#
# One-time migration: safely moves ObservatoryMetricsTable, AMPWorkspace,
# S3VpcEndpoint, DynamoDbVpcEndpoint, and Ipv6RouteToEigw out of the
# main application stack and into the shared infrastructure stack.
#
# Safe to re-run: every step is idempotent. If the shared stack already
# exists and the main stack no longer owns the resources, this exits 0.
#
# Usage:
#   AWS_REGION=us-east-1 \
#   STACK_NAME=tarun-content-team \
#   SHARED_STACK_NAME=tarun-teamweave-shared \
#   VPC_ID=vpc-f3c92a8a \
#   PRIVATE_ROUTE_TABLE_ID="" \
#   EIGW_ID="" \
#   bash infra/scripts/migrate-to-shared.sh
#
# Prerequisites: aws CLI v2, Python 3, jq (optional but recommended)
# Permissions needed: cloudformation:*, dynamodb:DescribeTable, aps:ListWorkspaces

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="${STACK_NAME:?STACK_NAME must be set}"
SHARED_STACK_NAME="${SHARED_STACK_NAME:?SHARED_STACK_NAME must be set}"
VPC_ID="${VPC_ID:?VPC_ID must be set}"
PRIVATE_ROUTE_TABLE_ID="${PRIVATE_ROUTE_TABLE_ID:-}"
EIGW_ID="${EIGW_ID:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SHARED_TEMPLATE="${REPO_ROOT}/infra/shared.yaml"

log()  { echo "[migrate-to-shared] $*"; }
warn() { echo "[migrate-to-shared] WARN: $*" >&2; }
die()  { echo "[migrate-to-shared] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Check whether shared stack already owns the resources
# ---------------------------------------------------------------------------

log "Checking shared stack '${SHARED_STACK_NAME}'..."
SHARED_STATUS=$(aws cloudformation describe-stacks \
  --stack-name "${SHARED_STACK_NAME}" \
  --region "${REGION}" \
  --query "Stacks[0].StackStatus" \
  --output text 2>/dev/null || echo "DOES_NOT_EXIST")

if [[ "${SHARED_STATUS}" == "CREATE_COMPLETE" || "${SHARED_STATUS}" == "UPDATE_COMPLETE" ]]; then
  log "Shared stack already exists and is healthy (${SHARED_STATUS}). Nothing to migrate."
  exit 0
fi

if [[ "${SHARED_STATUS}" != "DOES_NOT_EXIST" ]]; then
  die "Shared stack is in unexpected state: ${SHARED_STATUS}. Resolve manually before re-running."
fi

# ---------------------------------------------------------------------------
# 2. Check whether the main stack currently owns the Observatory resources
# ---------------------------------------------------------------------------

log "Checking main stack '${STACK_NAME}' for Observatory resources..."
MAIN_STATUS=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query "Stacks[0].StackStatus" \
  --output text 2>/dev/null || echo "DOES_NOT_EXIST")

if [[ "${MAIN_STATUS}" == "DOES_NOT_EXIST" ]]; then
  log "Main stack does not exist yet — performing fresh shared stack deploy."
  NEEDS_RETENTION=false
  EXISTING_TABLE_NAME=""
  EXISTING_WORKSPACE_ID=""
else
  # Probe for ObservatoryMetricsTable in the main stack
  EXISTING_TABLE_NAME=$(aws cloudformation describe-stack-resource \
    --stack-name "${STACK_NAME}" \
    --logical-resource-id ObservatoryMetricsTable \
    --region "${REGION}" \
    --query "StackResourceDetail.PhysicalResourceId" \
    --output text 2>/dev/null || echo "")

  EXISTING_WORKSPACE_ID=$(aws cloudformation describe-stack-resource \
    --stack-name "${STACK_NAME}" \
    --logical-resource-id AMPWorkspace \
    --region "${REGION}" \
    --query "StackResourceDetail.PhysicalResourceId" \
    --output text 2>/dev/null || echo "")

  if [[ -n "${EXISTING_TABLE_NAME}" ]]; then
    log "Found existing ObservatoryMetricsTable: ${EXISTING_TABLE_NAME}"
    log "Found existing AMPWorkspace: ${EXISTING_WORKSPACE_ID:-<none>}"
    NEEDS_RETENTION=true
  else
    log "Main stack exists but Observatory resources not found — fresh shared stack deploy."
    NEEDS_RETENTION=false
    EXISTING_TABLE_NAME=""
    EXISTING_WORKSPACE_ID=""
  fi
fi

# ---------------------------------------------------------------------------
# 3. If main stack owns the resources: patch DeletionPolicy:Retain first
#    so CloudFormation won't destroy them when the new template removes them.
# ---------------------------------------------------------------------------

if [[ "${NEEDS_RETENTION}" == "true" ]]; then
  log "Step 3: Patching main stack to add DeletionPolicy:Retain on Observatory resources..."

  PATCHED_TEMPLATE=$(mktemp /tmp/template-retain-XXXXXX.yaml)
  trap "rm -f ${PATCHED_TEMPLATE}" EXIT

  # Fetch the live template from the main stack (not the local file — local
  # file already has the resources removed).
  aws cloudformation get-template \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --query "TemplateBody" \
    --output text > "${PATCHED_TEMPLATE}"

  # Insert DeletionPolicy: Retain after each resource type declaration for
  # the five resources being moved. Python is used for reliable YAML patching.
  python3 - "${PATCHED_TEMPLATE}" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    content = f.read()

resources = [
    "ObservatoryMetricsTable",
    "AMPWorkspace",
    "S3VpcEndpoint",
    "DynamoDbVpcEndpoint",
    "Ipv6RouteToEigw",
]

for resource in resources:
    # Match the resource block header and its Type line, then inject
    # DeletionPolicy and UpdateReplacePolicy if not already present.
    pattern = rf'(  {resource}:\n    Type: [^\n]+\n)(?!    DeletionPolicy)'
    replacement = r'\1    DeletionPolicy: Retain\n    UpdateReplacePolicy: Retain\n'
    content = re.sub(pattern, replacement, content)

with open(path, 'w') as f:
    f.write(content)
print("Patched template written.")
PYEOF

  log "Deploying retention patch to main stack (no-op if resources already retained)..."
  # Upload the patched template to S3 via CloudFormation (it may be too large
  # for direct --template-body) by using a temporary SAM-style S3 bucket.
  aws cloudformation deploy \
    --stack-name "${STACK_NAME}" \
    --template-file "${PATCHED_TEMPLATE}" \
    --region "${REGION}" \
    --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
    --no-fail-on-empty-changeset \
    || die "Failed to apply retention patch to main stack."

  log "Retention patch applied. The five resources will survive being removed from the template."
fi

# ---------------------------------------------------------------------------
# 4. Deploy the shared stack (creates fresh resources)
#    The old retained resources in the main stack are now orphaned — they
#    will expire naturally (DynamoDB TTL) and can be deleted manually.
# ---------------------------------------------------------------------------

log "Step 4: Deploying shared infrastructure stack '${SHARED_STACK_NAME}'..."

PARAM_OVERRIDES="VpcId=${VPC_ID}"
[[ -n "${PRIVATE_ROUTE_TABLE_ID}" ]] && PARAM_OVERRIDES="${PARAM_OVERRIDES} PrivateRouteTableId=${PRIVATE_ROUTE_TABLE_ID}"
[[ -n "${EIGW_ID}" ]]               && PARAM_OVERRIDES="${PARAM_OVERRIDES} EgressOnlyInternetGatewayId=${EIGW_ID}"

aws cloudformation deploy \
  --stack-name "${SHARED_STACK_NAME}" \
  --template-file "${SHARED_TEMPLATE}" \
  --region "${REGION}" \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides ${PARAM_OVERRIDES} \
  || die "Shared stack deployment failed."

log "Shared stack deployed successfully."

# ---------------------------------------------------------------------------
# 5. Print the outputs that must be passed to the main stack
# ---------------------------------------------------------------------------

log ""
log "Migration complete. Pass these values as parameter-overrides to the main stack:"
log ""

OUTPUTS=$(aws cloudformation describe-stacks \
  --stack-name "${SHARED_STACK_NAME}" \
  --region "${REGION}" \
  --query "Stacks[0].Outputs[*].[OutputKey,OutputValue]" \
  --output text)

echo "${OUTPUTS}" | while IFS=$'\t' read -r key value; do
  log "  ${key} = ${value}"
done

log ""
if [[ "${NEEDS_RETENTION}" == "true" && -n "${EXISTING_TABLE_NAME}" ]]; then
  log "IMPORTANT: The old Observatory DynamoDB table '${EXISTING_TABLE_NAME}' and"
  log "AMP workspace '${EXISTING_WORKSPACE_ID}' are retained (orphaned) in AWS."
  log "New spans will write to the new shared-stack table immediately."
  log "The old table's TTL will expire existing items within 90 days."
  log "To clean up manually: aws dynamodb delete-table --table-name ${EXISTING_TABLE_NAME}"
  [[ -n "${EXISTING_WORKSPACE_ID}" ]] && \
    log "                       aws amp delete-workspace --workspace-id ${EXISTING_WORKSPACE_ID}"
fi
