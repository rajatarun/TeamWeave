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
    """The document bucket is ContextWeave's. Managed search owns the index.

    A VECTOR base with an S3 Vectors index is what rejected Marengo. Neither
    the index nor a second document bucket is in this template.
    """
    assert "HealthDocsBucket" not in template["Resources"]
    assert "HealthVectorBucket" not in template["Resources"]
    assert "HealthVectorIndex" not in template["Resources"]
    assert "HealthKnowledgeBaseDataSource" not in template["Resources"]
    assert "AWS::S3Vectors::" not in TEMPLATE
    assert "S3_VECTORS" not in TEMPLATE
    kb = template["Resources"]["HealthKnowledgeBase"]
    assert kb["Type"] == "AWS::CloudFormation::CustomResource"
    rendered = str(kb)
    assert "HealthDocsBucketName" in rendered
    assert "DocsBucketName" in kb["Properties"]
    # The multimodal bucket is the extracted-media destination, not the source.
    assert "HealthMultimodalBucket" not in str(kb["Properties"]["DocsBucketName"])


def test_marengo_declares_a_multimodal_storage_destination(template):
    """CreateKnowledgeBase 400s without one: Marengo requires a multimodal
    storage destination. It holds extracted media, so it is its own bucket."""
    kb = template["Resources"]["HealthKnowledgeBase"]["Properties"]
    assert "HealthMultimodalBucket" in str(kb["MultimodalBucket"])
    assert "HealthDocsBucketName" not in str(kb["MultimodalBucket"])
    bucket = template["Resources"]["HealthMultimodalBucket"]
    assert bucket["Type"] == "AWS::S3::Bucket"
    assert bucket["Condition"] == "HealthKnowledgeBaseEnabled"
    assert "tw-health-mm-" in str(bucket["Properties"]["BucketName"])
    role = template["Resources"]["HealthKnowledgeBaseRole"]
    rendered = str(role)
    assert "s3:PutObject" in rendered
    assert "s3:DeleteObject" in rendered
    assert "HealthMultimodalBucket" in rendered
    assert "s3vectors:" not in rendered
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


def test_the_base_embeds_with_marengo(template):
    assert EMBEDDING_MODEL_ID in TEMPLATE
    kb = template["Resources"]["HealthKnowledgeBase"]["Properties"]
    assert EMBEDDING_MODEL_ID in str(kb["EmbeddingModelArn"])
    assert template["Resources"]["HealthKnowledgeBase"]["Type"] == "AWS::CloudFormation::CustomResource"
    role = str(template["Resources"]["HealthKnowledgeBaseRole"])
    assert EMBEDDING_MODEL_ID in role
    provision = str(template["Resources"]["HealthKbProvisionRole"])
    assert "bedrock:CreateKnowledgeBase" in provision
    assert "iam:PassRole" in provision
    assert "HealthKnowledgeBaseRole" in provision
    function = template["Resources"]["HealthKbProvisionFunction"]
    assert function["Properties"]["Handler"] == "src/orchestrator/health_kb_provision.handler"
    assert function["Properties"]["Timeout"] == 900
    assert "VpcConfig" not in function["Properties"]


def test_the_base_exists_only_when_both_imports_are_present(template):
    condition = str(template["Conditions"]["HealthKnowledgeBaseEnabled"])
    assert "HealthDocsBucketName" in condition
    assert "HealthDocsKmsKeyArn" in condition
    for name in (
        "HealthKnowledgeBase",
        "HealthKbProvisionFunction",
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
    loaded = yaml.load(TEMPLATE, Loader=CfnLoader)
    source_id = loaded["Resources"]["HealthKbSyncFunction"]["Properties"]["Environment"]["Variables"]["HEALTH_DATA_SOURCE_ID"]
    assert "HealthKnowledgeBase" in str(source_id)
    assert "HealthKnowledgeBaseDataSource" not in str(source_id)
    assert "DataSourceId" in str(loaded["Outputs"]["HealthDataSourceId"]["Value"])


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
    assert seen["retrievalConfiguration"]["managedSearchConfiguration"]["numberOfResults"] == 4
    assert "vectorSearchConfiguration" not in seen["retrievalConfiguration"]
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


# ── CreateKnowledgeBase is the managed Marengo body, not a VECTOR base ──────

MODEL_ARN = (
    "arn:aws:bedrock:us-east-1::foundation-model/twelvelabs.marengo-embed-3-0-v1:0"
)
REQUEST_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def test_the_create_body_is_the_documented_managed_shape():
    """botocore accepts this body. A VECTOR body is what the API rejected."""
    from botocore.session import Session
    from botocore.validate import ParamValidator

    from orchestrator.health_kb_provision import (
        data_source_fields,
        knowledge_base_configuration,
    )

    configuration = knowledge_base_configuration(MODEL_ARN, "tw-health-mm-1-us-east-1")
    assert configuration == {
        "type": "MANAGED",
        "managedKnowledgeBaseConfiguration": {
            "embeddingModelType": "CUSTOM",
            "embeddingModelArn": MODEL_ARN,
            "embeddingModelConfiguration": {
                "bedrockEmbeddingModelConfiguration": {
                    "embeddingDataType": "FLOAT",
                    "modelConfiguration": {
                        "version": "1",
                        "audio": {
                            "segmentation": {
                                "method": "dynamic",
                                "dynamic": {"minDurationSec": 4},
                            }
                        },
                        "video": {
                            "segmentation": {
                                "method": "fixed",
                                "fixed": {"durationSec": 6},
                            }
                        },
                    },
                }
            },
            "supplementalDataStorageConfiguration": {
                "storageLocations": [
                    {
                        "type": "S3",
                        "s3Location": {"uri": "s3://tw-health-mm-1-us-east-1/"},
                    }
                ]
            },
        },
    }
    assert "storageConfiguration" not in configuration
    assert "dimensions" not in str(configuration)

    service = Session().get_service_model("bedrock-agent")
    validator = ParamValidator()
    create = validator.validate(
        {
            "name": "health-registry-teamweave",
            "roleArn": "arn:aws:iam::239571291755:role/kb",
            "description": "Index of ContextWeave's health bucket for the health_prep team.",
            "knowledgeBaseConfiguration": configuration,
        },
        service.operation_model("CreateKnowledgeBase").input_shape,
    )
    assert not create.has_errors(), create.generate_report()

    fields = data_source_fields("contextweave-health-docs", "239571291755")
    assert fields["dataSourceConfiguration"]["type"] == "MANAGED_KNOWLEDGE_BASE_CONNECTOR"
    connection = fields["dataSourceConfiguration"]["managedKnowledgeBaseConnectorConfiguration"]
    assert connection["connectorParameters"]["connectionConfiguration"] == {
        "bucketName": "contextweave-health-docs",
        "bucketOwnerAccountId": "239571291755",
    }
    assert fields["dataDeletionPolicy"] == "DELETE"
    assert fields["vectorIngestionConfiguration"]["parsingConfiguration"]["parsingStrategy"] == "SMART_PARSING"
    data_source = validator.validate(
        {"knowledgeBaseId": "KBID123456", **fields},
        service.operation_model("CreateDataSource").input_shape,
    )
    assert not data_source.has_errors(), data_source.generate_report()

    runtime = Session().get_service_model("bedrock-agent-runtime")
    retrieve = validator.validate(
        {
            "knowledgeBaseId": "KBID123456",
            "retrievalQuery": {"text": "last lab"},
            "retrievalConfiguration": {"managedSearchConfiguration": {"numberOfResults": 4}},
        },
        runtime.operation_model("Retrieve").input_shape,
    )
    assert not retrieve.has_errors(), retrieve.generate_report()
    # The members the runtime client does not have. A model that validates
    # the call is the one the functions are packaged with.
    assert "managedKnowledgeBaseConfiguration" in service.shape_for("KnowledgeBaseConfiguration").members
    managed = service.shape_for("ManagedKnowledgeBaseConfiguration").members
    assert "supplementalDataStorageConfiguration" in managed
    assert "modelConfiguration" in service.shape_for("BedrockEmbeddingModelConfiguration").members
    assert "MANAGED_KNOWLEDGE_BASE_CONNECTOR" in service.shape_for("DataSourceType").enum
    assert "managedSearchConfiguration" in runtime.shape_for("KnowledgeBaseRetrievalConfiguration").members


class _Context:
    log_stream_name = "stream"

    def get_remaining_time_in_millis(self):
        return 120_000


def _event(request_type, physical=""):
    event = {
        "RequestType": request_type,
        "ResponseURL": "https://cloudformation-custom-resource-response.s3.amazonaws.com/response",
        "StackId": "arn:aws:cloudformation:us-east-1:239571291755:stack/teamweave/guid",
        "RequestId": REQUEST_ID,
        "LogicalResourceId": "HealthKnowledgeBase",
        "ResourceProperties": {
            "ServiceToken": "arn:aws:lambda:us-east-1:239571291755:function:provision",
            "KnowledgeBaseName": "health-registry-teamweave",
            "Description": "Index of ContextWeave's health bucket for the health_prep team.",
            "RoleArn": "arn:aws:iam::239571291755:role/health-kb",
            "EmbeddingModelArn": MODEL_ARN,
            "MultimodalBucket": "tw-health-mm-239571291755-us-east-1",
            "DocsBucketName": "contextweave-health-docs",
            "DocsBucketOwnerAccountId": "239571291755",
        },
    }
    if physical:
        event["PhysicalResourceId"] = physical
    return event


def _install_response(monkeypatch):
    sent = []

    def urlopen(request, timeout=None):
        sent.append(request)

        class _Response:
            def read(self):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("orchestrator.health_kb_provision.time.sleep", lambda *_: None)
    return sent


def _body(request):
    import json
    return json.loads(request.data.decode())


def test_create_calls_the_managed_api_and_returns_the_ids(monkeypatch):
    from orchestrator import health_kb_provision

    sent = _install_response(monkeypatch)
    seen = {}

    class Client:
        def create_knowledge_base(self, **kwargs):
            seen["create"] = kwargs
            return {"knowledgeBase": {"knowledgeBaseId": "KBHEALTH", "status": "CREATING"}}

        def get_knowledge_base(self, **kwargs):
            return {"knowledgeBase": {
                "knowledgeBaseId": "KBHEALTH",
                "knowledgeBaseArn": "arn:aws:bedrock:us-east-1:239571291755:knowledge-base/KBHEALTH",
                "status": "ACTIVE",
            }}

        def list_data_sources(self, **kwargs):
            return {"dataSourceSummaries": []}

        def create_data_source(self, **kwargs):
            seen["data_source"] = kwargs
            return {"dataSource": {"dataSourceId": "DSHEALTH", "status": "CREATING"}}

        def get_data_source(self, **kwargs):
            return {"dataSource": {"dataSourceId": "DSHEALTH", "status": "AVAILABLE"}}

    monkeypatch.setattr(health_kb_provision, "_client", lambda: Client())
    result = health_kb_provision.handler(_event("Create"), _Context())
    assert result["KnowledgeBaseId"] == "KBHEALTH"
    assert result["DataSourceId"] == "DSHEALTH"
    create = seen["create"]
    assert "storageConfiguration" not in create
    assert create["knowledgeBaseConfiguration"]["type"] == "MANAGED"
    managed = create["knowledgeBaseConfiguration"]["managedKnowledgeBaseConfiguration"]
    assert managed["embeddingModelArn"].endswith(EMBEDDING_MODEL_ID)
    assert managed["embeddingModelConfiguration"]["bedrockEmbeddingModelConfiguration"]["embeddingDataType"] == "FLOAT"
    assert managed["supplementalDataStorageConfiguration"]["storageLocations"][0]["s3Location"]["uri"] == (
        "s3://tw-health-mm-239571291755-us-east-1/"
    )
    source = seen["data_source"]
    assert source["dataSourceConfiguration"]["type"] == "MANAGED_KNOWLEDGE_BASE_CONNECTOR"
    connection = source["dataSourceConfiguration"]["managedKnowledgeBaseConnectorConfiguration"]["connectorParameters"]
    assert connection["connectionConfiguration"]["bucketName"] == "contextweave-health-docs"
    assert "s3Configuration" not in source["dataSourceConfiguration"]
    body = _body(sent[0])
    assert request_method(sent[0]) == "PUT"
    assert body["Status"] == "SUCCESS"
    assert body["PhysicalResourceId"] == "KBHEALTH"
    assert body["Data"]["KnowledgeBaseArn"].endswith("knowledge-base/KBHEALTH")
    assert body["Data"]["DataSourceId"] == "DSHEALTH"
    content_type = sent[0].get_header("Content-type")
    assert content_type in ("", None)


def request_method(request):
    return request.get_method()


def test_update_keeps_the_same_base(monkeypatch):
    from orchestrator import health_kb_provision

    sent = _install_response(monkeypatch)
    seen = {}

    class Client:
        def create_knowledge_base(self, **kwargs):
            raise AssertionError("update created a second knowledge base")

        def update_knowledge_base(self, **kwargs):
            seen["update"] = kwargs

        def get_knowledge_base(self, **kwargs):
            return {"knowledgeBase": {
                "knowledgeBaseId": "KBHEALTH",
                "knowledgeBaseArn": "arn:aws:bedrock:us-east-1:239571291755:knowledge-base/KBHEALTH",
                "status": "ACTIVE",
            }}

        def list_data_sources(self, **kwargs):
            return {"dataSourceSummaries": [
                {"dataSourceId": "DSHEALTH", "name": "health-s3", "status": "AVAILABLE"},
            ]}

        def update_data_source(self, **kwargs):
            seen["data_source"] = kwargs

        def get_data_source(self, **kwargs):
            return {"dataSource": {"dataSourceId": "DSHEALTH", "status": "AVAILABLE"}}

    monkeypatch.setattr(health_kb_provision, "_client", lambda: Client())
    health_kb_provision.handler(_event("Update", "KBHEALTH"), _Context())
    assert seen["update"]["knowledgeBaseId"] == "KBHEALTH"
    assert seen["update"]["knowledgeBaseConfiguration"]["type"] == "MANAGED"
    assert "storageConfiguration" not in seen["update"]
    assert seen["data_source"]["dataSourceId"] == "DSHEALTH"
    assert seen["data_source"]["dataSourceConfiguration"]["type"] == "MANAGED_KNOWLEDGE_BASE_CONNECTOR"
    body = _body(sent[0])
    assert body["Status"] == "SUCCESS"
    assert body["PhysicalResourceId"] == "KBHEALTH"


def test_delete_removes_the_connector_before_the_base(monkeypatch):
    from botocore.exceptions import ClientError

    from orchestrator import health_kb_provision

    sent = _install_response(monkeypatch)
    deleted = []
    lists = {"n": 0}

    class Client:
        def list_data_sources(self, **kwargs):
            lists["n"] += 1
            if lists["n"] == 1:
                return {"dataSourceSummaries": [
                    {"dataSourceId": "DSHEALTH", "name": "health-s3", "status": "AVAILABLE"},
                ]}
            return {"dataSourceSummaries": []}

        def delete_data_source(self, **kwargs):
            deleted.append(("data-source", kwargs["dataSourceId"]))

        def delete_knowledge_base(self, **kwargs):
            deleted.append(("knowledge-base", kwargs["knowledgeBaseId"]))

        def get_knowledge_base(self, **kwargs):
            raise ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": "gone"}},
                "GetKnowledgeBase",
            )

    monkeypatch.setattr(health_kb_provision, "_client", lambda: Client())
    result = health_kb_provision.handler(_event("Delete", "KBHEALTH"), _Context())
    assert deleted == [("data-source", "DSHEALTH"), ("knowledge-base", "KBHEALTH")]
    assert result["PhysicalResourceId"] == "KBHEALTH"
    assert _body(sent[0])["Status"] == "SUCCESS"


def test_delete_of_a_missing_base_succeeds(monkeypatch):
    from botocore.exceptions import ClientError

    from orchestrator import health_kb_provision

    sent = _install_response(monkeypatch)

    class Client:
        def list_data_sources(self, **kwargs):
            raise ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": "gone"}},
                "ListDataSources",
            )

    monkeypatch.setattr(health_kb_provision, "_client", lambda: Client())
    health_kb_provision.handler(_event("Delete", "KBHEALTH"), _Context())
    assert _body(sent[0])["Status"] == "SUCCESS"


def test_a_create_failure_reports_the_base_id_so_rollback_can_delete_it(monkeypatch):
    from orchestrator import health_kb_provision

    sent = _install_response(monkeypatch)
    logged = []
    monkeypatch.setattr(
        health_kb_provision.log, "warning", lambda *a, **k: logged.append((a, k)),
    )

    class Client:
        def create_knowledge_base(self, **kwargs):
            return {"knowledgeBase": {"knowledgeBaseId": "KBHEALTH", "status": "CREATING"}}

        def get_knowledge_base(self, **kwargs):
            return {"knowledgeBase": {
                "knowledgeBaseId": "KBHEALTH",
                "status": "FAILED",
                "failureReasons": ["The specified embedding model is not supported."],
            }}

    monkeypatch.setattr(health_kb_provision, "_client", lambda: Client())
    health_kb_provision.handler(_event("Create"), _Context())
    body = _body(sent[0])
    assert body["Status"] == "FAILED"
    assert body["PhysicalResourceId"] == "KBHEALTH"
    assert "not supported" in body["Reason"]
    assert "discharge" not in str(logged)


def test_only_the_callers_bundle_a_botocore_that_knows_managed_bases():
    """The runtime copy rejects the body before Bedrock sees it.

    src/requirements.txt stays free of boto3 so the other functions keep
    the runtime SDK. The two callers install a pin whose model accepts the
    Marengo body. 1.43.32 is not that pin: it knows the managed type and
    still rejects modelConfiguration and the supplemental location.
    """
    pinned = (REPO / "src" / "requirements-bedrock-kb.txt").read_text()
    assert "botocore>=1.43.92" in pinned
    assert "boto3>=1.43.92" in pinned
    shared = (REPO / "src" / "requirements.txt").read_text()
    assert "boto3" not in shared
    assert "botocore" not in shared
    makefile = (REPO / "Makefile").read_text()
    assert "src/requirements-bedrock-kb.txt" in makefile
    recipes = {}
    for line in makefile.splitlines():
        if not line.startswith("build-") or ":" not in line:
            continue
        name, deps = line.split(":", 1)
        recipes[name.removeprefix("build-")] = deps
    bundled = {name for name, deps in recipes.items() if "install-bedrock-kb-sdk" in deps}
    assert bundled == {"WorkerFunction", "HealthKbProvisionFunction"}


def test_the_provision_client_model_accepts_the_body_it_sends():
    """The client the function constructs, not a model loaded beside it."""
    from botocore.validate import ParamValidator

    from orchestrator.health_kb_provision import (
        _client,
        data_source_fields,
        knowledge_base_configuration,
    )

    client = _client()
    model = client.meta.service_model
    validator = ParamValidator()
    create = validator.validate(
        {
            "name": "health-registry-teamweave",
            "roleArn": "arn:aws:iam::239571291755:role/kb",
            "description": "Index of ContextWeave's health bucket for the health_prep team.",
            "knowledgeBaseConfiguration": knowledge_base_configuration(
                MODEL_ARN, "tw-health-mm-1-us-east-1",
            ),
        },
        model.operation_model("CreateKnowledgeBase").input_shape,
    )
    assert not create.has_errors(), create.generate_report()
    source = validator.validate(
        {"knowledgeBaseId": "KBID123456", **data_source_fields("contextweave-health-docs", "239571291755")},
        model.operation_model("CreateDataSource").input_shape,
    )
    assert not source.has_errors(), source.generate_report()
    update = validator.validate(
        {
            "knowledgeBaseId": "KBID123456",
            "name": "health-registry-teamweave",
            "roleArn": "arn:aws:iam::239571291755:role/kb",
            "knowledgeBaseConfiguration": knowledge_base_configuration(
                MODEL_ARN, "tw-health-mm-1-us-east-1",
            ),
        },
        model.operation_model("UpdateKnowledgeBase").input_shape,
    )
    assert not update.has_errors(), update.generate_report()


def test_an_old_client_model_is_refused_before_create():
    """The production failure: the runtime model has no managed member."""
    from orchestrator.health_kb_provision import _assert_managed_model

    class _Shape:
        def __init__(self, members=None, enum=None):
            self.members = members or {}
            self.enum = enum

    class _Model:
        def shape_for(self, name):
            if name == "KnowledgeBaseConfiguration":
                return _Shape(members={
                    "type": None,
                    "vectorKnowledgeBaseConfiguration": None,
                    "kendraKnowledgeBaseConfiguration": None,
                    "sqlKnowledgeBaseConfiguration": None,
                })
            if name == "DataSourceType":
                return _Shape(enum=["S3"])
            raise KeyError(name)

    class _Client:
        class meta:
            service_model = _Model()

    with pytest.raises(RuntimeError, match="managedKnowledgeBaseConfiguration"):
        _assert_managed_model(_Client())


def test_a_model_that_stops_at_the_managed_type_is_not_new_enough():
    """1.43.32 has the managed member and still rejects the Marengo fields."""
    from orchestrator.health_kb_provision import _assert_managed_model

    class _Shape:
        def __init__(self, members=None, enum=None):
            self.members = members or {}
            self.enum = enum

    shapes = {
        "KnowledgeBaseConfiguration": _Shape(members={
            "type": None,
            "managedKnowledgeBaseConfiguration": None,
        }),
        "ManagedKnowledgeBaseConfiguration": _Shape(members={
            "embeddingModelType": None,
            "embeddingModelArn": None,
            "embeddingModelConfiguration": None,
        }),
        "BedrockEmbeddingModelConfiguration": _Shape(members={
            "dimensions": None,
            "embeddingDataType": None,
            "audio": None,
            "video": None,
        }),
        "DataSourceType": _Shape(enum=["S3", "MANAGED_KNOWLEDGE_BASE_CONNECTOR"]),
    }

    class _Model:
        def shape_for(self, name):
            return shapes[name]

    class _Client:
        class meta:
            service_model = _Model()

    with pytest.raises(RuntimeError, match="modelConfiguration"):
        _assert_managed_model(_Client())


def test_retrieve_refuses_a_client_without_managed_search():
    from orchestrator.health_kb import _assert_managed_retrieve

    class _Shape:
        members = {"vectorSearchConfiguration": None}

    class _Model:
        def shape_for(self, name):
            return _Shape()

    class _Client:
        class meta:
            service_model = _Model()

    with pytest.raises(RuntimeError, match="managedSearchConfiguration"):
        _assert_managed_retrieve(_Client())


def test_the_installed_retrieve_client_accepts_managed_search():
    from botocore.validate import ParamValidator

    from orchestrator.health_kb import _client

    client = _client("bedrock-agent-runtime")
    err = ParamValidator().validate(
        {
            "knowledgeBaseId": "KBID123456",
            "retrievalQuery": {"text": "last lab"},
            "retrievalConfiguration": {"managedSearchConfiguration": {"numberOfResults": 4}},
        },
        client.meta.service_model.operation_model("Retrieve").input_shape,
    )
    assert not err.has_errors(), err.generate_report()
