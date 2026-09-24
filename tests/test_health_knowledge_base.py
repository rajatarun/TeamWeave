"""The health knowledge base indexes ContextWeave's bucket. It does not create one.

ContextWeave's stack output ``HealthDocsBucketName`` is the bucket name (it
publishes no ARN) and ``KMSKeyArn`` is the key that bucket is encrypted with.
Deploy reads both the same way it reads ``APIEndpoint`` into ``ContextWeaveUrl``.
A missing health output warns and leaves the base uncreated. The visibility
team's URL is the one that fails the deploy, because that team cannot run
without it. Health can.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
TEMPLATE = (REPO / "infra" / "template.yaml").read_text()
WORKFLOW = (REPO / ".github" / "workflows" / "deploy.yml").read_text()

EMBEDDING_MODEL_ID = "twelvelabs.marengo-embed-3-0-v1:0"


class CfnLoader(yaml.SafeLoader):
    pass


def _keep(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {"__fn__": suffix, "__arg__": value}


CfnLoader.add_multi_constructor("!", _keep)


@pytest.fixture(scope="module")
def template():
    return yaml.load(TEMPLATE, Loader=CfnLoader)


def test_the_bucket_name_is_a_parameter_taken_from_contextweave(template):
    param = template["Parameters"]["HealthDocsBucketName"]
    assert param["Default"] == ""
    assert "HealthDocsBucketName" in param["Description"]
    kms = template["Parameters"]["HealthDocsKmsKeyArn"]
    assert kms["Default"] == ""
    assert "KMSKeyArn" in kms["Description"]


def test_the_deploy_reads_those_outputs_from_the_contextweave_stack():
    """The same describe-stacks read as APIEndpoint, against the same stack.

    A hardcoded bucket name would drift from the bucket ContextWeave actually
    created, and a deploy that omitted the override would keep the previous
    value forever.
    """
    start = WORKFLOW.index("OutputKey=='HealthDocsBucketName'")
    end = WORKFLOW.index("MCP_WIRED=()", start)
    block = WORKFLOW[start:end]
    assert "CONTEXTWEAVE_STACK_NAME" in block
    assert "OutputKey=='KMSKeyArn'" in block
    assert 'HealthDocsBucketName=${HEALTH_DOCS_BUCKET}' in block
    assert 'HealthDocsKmsKeyArn=${HEALTH_DOCS_KMS}' in block
    assert "exit 1" not in block, "a missing health bucket must not fail the deploy"


def test_there_is_no_second_health_document_bucket(template):
    assert "HealthDocsBucket" not in template["Resources"]
    source = template["Resources"]["HealthKnowledgeBaseDataSource"]
    rendered = str(source)
    assert "HealthDocsBucketName" in rendered
    assert "HealthMultimodalBucket" not in rendered
    assert source["Properties"]["DataSourceConfiguration"]["Type"] == "S3"
    # The vector bucket is the index, and its name says so.
    vector = template["Resources"]["HealthVectorBucket"]
    assert vector["Type"] == "AWS::S3Vectors::VectorBucket"
    assert "tw-health-vec-" in str(vector["Properties"]["VectorBucketName"])


def test_marengo_declares_a_multimodal_storage_destination(template):
    """CreateKnowledgeBase 400s without one: Marengo requires a multimodal
    storage destination. It holds extracted media, so it is its own bucket."""
    vector = (
        template["Resources"]["HealthKnowledgeBase"]["Properties"]
        ["KnowledgeBaseConfiguration"]["VectorKnowledgeBaseConfiguration"]
    )
    locations = vector["SupplementalDataStorageConfiguration"]["SupplementalDataStorageLocations"]
    assert len(locations) == 1
    assert locations[0]["SupplementalDataStorageLocationType"] == "S3"
    uri = str(locations[0]["S3Location"]["URI"])
    assert "s3://" in uri
    assert "HealthMultimodalBucket" in uri
    assert "HealthDocsBucketName" not in uri
    bucket = template["Resources"]["HealthMultimodalBucket"]
    assert bucket["Type"] == "AWS::S3::Bucket"
    assert bucket["Condition"] == "HealthKnowledgeBaseEnabled"
    assert "tw-health-mm-" in str(bucket["Properties"]["BucketName"])
    role = template["Resources"]["HealthKnowledgeBaseRole"]
    rendered = str(role)
    assert "s3:PutObject" in rendered
    assert "s3:DeleteObject" in rendered
    assert "HealthMultimodalBucket" in rendered
    # The document bucket grant is still read-only.
    docs_write = False
    for statement in role["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]:
        if not isinstance(statement, dict):
            continue
        actions = statement.get("Action")
        actions = actions if isinstance(actions, list) else [actions]
        resource = str(statement.get("Resource"))
        if "HealthDocsBucketName" in resource and "s3:PutObject" in actions:
            docs_write = True
    assert not docs_write


def test_the_base_embeds_with_marengo_at_512(template):
    assert EMBEDDING_MODEL_ID in TEMPLATE
    kb = template["Resources"]["HealthKnowledgeBase"]["Properties"]
    arn = kb["KnowledgeBaseConfiguration"]["VectorKnowledgeBaseConfiguration"]["EmbeddingModelArn"]
    assert EMBEDDING_MODEL_ID in str(arn)
    index = template["Resources"]["HealthVectorIndex"]["Properties"]
    assert index["Dimension"] == 512
    assert index["DistanceMetric"] == "cosine"
    assert index["DataType"] == "float32"
    role = str(template["Resources"]["HealthKnowledgeBaseRole"])
    assert EMBEDDING_MODEL_ID in role


def test_the_base_exists_only_when_both_imports_are_present(template):
    condition = str(template["Conditions"]["HealthKnowledgeBaseEnabled"])
    assert "HealthDocsBucketName" in condition
    assert "HealthDocsKmsKeyArn" in condition
    for name in (
        "HealthKnowledgeBase",
        "HealthKnowledgeBaseDataSource",
        "HealthVectorBucket",
        "HealthVectorIndex",
        "HealthMultimodalBucket",
        "HealthKbSyncFunction",
    ):
        assert template["Resources"][name]["Condition"] == "HealthKnowledgeBaseEnabled"


def test_the_worker_retrieves_and_only_from_this_base(template):
    worker = template["Resources"]["WorkerFunction"]["Properties"]["Environment"]["Variables"]
    assert "HealthKnowledgeBase" in str(worker["HEALTH_KNOWLEDGE_BASE_ID"])
    role = str(template["Resources"]["WorkerRole"])
    assert "bedrock:Retrieve" in role
    assert "HealthKnowledgeBase.KnowledgeBaseArn" in role or "KnowledgeBaseArn" in role
    # The sync function starts jobs. The worker, which sees the excerpts, does not.
    assert "bedrock:StartIngestionJob" not in role
    sync = str(template["Resources"]["HealthKbSyncRole"])
    assert "bedrock:StartIngestionJob" in sync
    assert "s3:GetObject" not in sync


def test_a_bucket_change_reindexes_and_a_deploy_indexes_what_is_already_there():
    rule = yaml.load(TEMPLATE, Loader=CfnLoader)["Resources"]["HealthKbSyncRule"]
    pattern = rule["Properties"]["EventPattern"]
    assert pattern["source"] == ["aws.s3"]
    assert "Object Created" in pattern["detail-type"]
    assert "Object Deleted" in pattern["detail-type"]
    assert "HealthDocsBucketName" in str(pattern["detail"])
    assert "start-ingestion-job" in WORKFLOW
    assert "HealthKnowledgeBaseId" in WORKFLOW
    assert "HealthDataSourceId" in WORKFLOW


# ── the tool actually calls it, and still refuses without a person ──────────

def test_a_configured_base_is_what_health_prep_reads(monkeypatch):
    from orchestrator.tools import weave_tools

    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    seen = {}

    class Client:
        def retrieve(self, **kwargs):
            seen.update(kwargs)
            return {"retrievalResults": [{
                "content": {"text": "HbA1c 6.1 in March"},
                "location": {"s3Location": {"uri": "s3://contextweave-health/labs.txt"}},
            }]}

    monkeypatch.setattr(
        "orchestrator.health_kb._client",
        lambda service: Client() if service == "bedrock-agent-runtime" else (_ for _ in ()).throw(
            AssertionError(service)),
    )
    posted = []
    monkeypatch.setattr(
        "orchestrator.contextweave_client._post",
        lambda *a, **k: posted.append(1) or {"found": True},
    )
    result = weave_tools.query_health_record("last lab", caller_token="tok", top_k=4)
    assert seen["knowledgeBaseId"] == "KBHEALTH"
    assert seen["retrievalQuery"] == {"text": "last lab"}
    assert seen["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"] == 4
    assert result["found"] is True
    assert result["excerpts"][0]["content"] == "HbA1c 6.1 in March"
    assert result["source"] == "bedrock-knowledge-base"
    assert not posted, "the health API was called while the knowledge base is configured"


def test_without_a_token_the_base_is_not_asked(monkeypatch):
    from orchestrator.tools import weave_tools

    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    monkeypatch.setattr(
        "orchestrator.health_kb._client",
        lambda service: (_ for _ in ()).throw(AssertionError("retrieved without a token")),
    )
    result = weave_tools.query_health_record("chest tightness", caller_token="")
    assert "error" in result
    assert "found" not in result


def test_an_empty_base_is_not_reported_as_a_failure(monkeypatch):
    from orchestrator import health_kb

    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")

    class Client:
        def retrieve(self, **kwargs):
            return {"retrievalResults": []}

    result = health_kb.retrieve("q", client=Client())
    assert result["found"] is False
    assert result["excerpts"] == []
    assert "error" not in result


def test_a_failed_retrieve_does_not_look_like_an_empty_record(monkeypatch):
    from botocore.exceptions import ClientError

    from orchestrator import health_kb

    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")

    class Client:
        def retrieve(self, **kwargs):
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "s3://secret/key"}},
                "Retrieve",
            )

    result = health_kb.retrieve("q", client=Client())
    assert "error" in result
    assert "found" not in result
    assert "secret" not in result["error"]


def test_with_no_base_the_health_api_is_still_used(monkeypatch):
    from orchestrator.tools import weave_tools

    monkeypatch.delenv("HEALTH_KNOWLEDGE_BASE_ID", raising=False)
    seen = {}

    def fake_post(path, body, max_retries=2, bearer=""):
        seen.update(path=path, bearer=bearer)
        return {"found": False, "answer": ""}

    monkeypatch.setattr("orchestrator.contextweave_client._post", fake_post)
    result = weave_tools.query_health_record("q", caller_token="tok")
    assert seen["path"] == "/health/query"
    assert seen["bearer"] == "tok"
    assert result["found"] is False


def test_a_sync_already_running_is_not_a_failure(monkeypatch):
    from botocore.exceptions import ClientError

    from orchestrator import health_kb

    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    monkeypatch.setenv("HEALTH_DATA_SOURCE_ID", "DSHEALTH")

    class Client:
        def start_ingestion_job(self, **kwargs):
            raise ClientError(
                {"Error": {"Code": "ConflictException", "Message": "running"}},
                "StartIngestionJob",
            )

    assert health_kb.start_sync(client=Client()) == {"kbSync": "already-running"}


def test_the_sync_handler_does_not_log_the_object_key(monkeypatch):
    """The event is the S3 notification. Its key identifies a medical document."""
    from orchestrator import health_kb

    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    monkeypatch.setenv("HEALTH_DATA_SOURCE_ID", "DSHEALTH")
    logged = []
    monkeypatch.setattr(health_kb.log, "info", lambda *a, **k: logged.append((a, k)))
    monkeypatch.setattr(health_kb.log, "warning", lambda *a, **k: logged.append((a, k)))

    class Client:
        def start_ingestion_job(self, **kwargs):
            return {"ingestionJob": {"ingestionJobId": "job"}}

    monkeypatch.setattr(health_kb, "_client", lambda service: Client())

    class Context:
        def get_remaining_time_in_millis(self):
            return 60_000

    result = health_kb.sync_handler(
        {"detail": {"object": {"key": "discharge-summary.pdf"}}}, Context())
    assert result == {"kbSync": "started"}
    assert "discharge-summary" not in str(logged)
