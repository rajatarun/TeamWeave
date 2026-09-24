"""The ingest import fix lives in ContextWeave, which this token cannot push to.

The patch in this tree is that fix and nothing else. The knowledge base is
wired in this repository, against ContextWeave's existing bucket output, so
the patch must not grow a second base or a second document bucket.
"""
from pathlib import Path

PATCH = Path("patches/contextweave-health-kb-and-ingest.patch").read_text()


def test_the_ingest_handler_no_longer_imports_above_the_top_level_package():
    """Lambda's handler is health_api.ingest with CodeUri src/. A relative
    import past that package is the Runtime.ImportModuleError."""
    assert "def shared_module" in PATCH
    # Deleted, not merely mentioned. A comment that names the old import
    # would not satisfy this, and the leading '-' is what makes it a deletion.
    assert "\n-from ..shared import health_db\n" in PATCH
    assert "\n-        from ..shared.embedder import embed_texts as embed\n" in PATCH
    assert "\n-        from ..shared.embedder import embed_text as embed\n" in PATCH
    assert 'shared_module("embedder").embed_texts' in PATCH
    assert 'shared_module("embedder").embed_text' in PATCH


def test_the_patch_does_not_create_the_knowledge_base():
    """The base is TeamWeave's, pointed at ContextWeave's bucket output."""
    assert "AWS::Bedrock::KnowledgeBase" not in PATCH
    assert "AWS::S3::Bucket" not in PATCH
    assert "HealthVectorBucket" not in PATCH
