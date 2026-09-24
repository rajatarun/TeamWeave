"""The health knowledge base and the ingest import fix live in ContextWeave.

This token cannot push to that repository. The patch in this tree is the
change, checked here so a rewrite cannot drop the model id or the import fix
and still look complete.
"""
from pathlib import Path

PATCH = Path("patches/contextweave-health-kb-and-ingest.patch").read_text()


def test_the_health_bucket_is_a_marengo_knowledge_base():
    assert "twelvelabs.marengo-embed-3-0-v1:0" in PATCH
    assert "AWS::Bedrock::KnowledgeBase" in PATCH
    assert "Dimension: 512" in PATCH
    assert "Type: S3" in PATCH


def test_the_health_api_reads_that_knowledge_base():
    assert "HEALTH_KNOWLEDGE_BASE_ID" in PATCH
    assert "bedrock:Retrieve" in PATCH
    assert "retrieve_excerpts" in PATCH


def test_the_ingest_handler_no_longer_imports_above_the_top_level_package():
    """Lambda's handler is health_api.ingest with CodeUri src/. A relative
    import past that package is the Runtime.ImportModuleError."""
    assert "def shared_module" in PATCH
    # Deleted, not merely mentioned. A comment that names the old import
    # would not satisfy this.
    assert "\n-from ..shared import health_db\n" in PATCH
