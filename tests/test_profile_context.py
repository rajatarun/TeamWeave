import importlib
import sys
import types
import unittest
from unittest.mock import patch


class OwnerProfileContextTests(unittest.TestCase):
    def test_logs_top_k_during_profile_retrieval(self):
        fake_rag = types.ModuleType("src.orchestrator.rag")
        fake_rag.retrieve_from_vector_store = lambda *_args, **_kwargs: [{"source": "s", "text": "t"}]

        with patch.dict(sys.modules, {"src.orchestrator.rag": fake_rag}):
            profile_context = importlib.import_module("src.orchestrator.profile_context")
            profile_context = importlib.reload(profile_context)
            team_raw = {"globals": {"owner_profile": {"enabled": True, "top_k": 9}}}
            request_obj = {"topic": "wins"}

            with patch.dict("os.environ", {"VECTOR_DB_TABLE": "profiles"}, clear=False):
                with patch.object(profile_context.log, "info") as mock_info:
                    profile_context.get_owner_profile_context(request_obj, team_raw, "Jane Doe")

        self.assertTrue(mock_info.called)
        mock_info.assert_any_call(
            "owner_profile_context_retrieval",
            extra={"collection_id": "profiles", "top_k": 9, "owner": "Jane Doe"},
        )

    def test_redacts_owner_pii_from_profile_context(self):
        fake_rag = types.ModuleType("src.orchestrator.rag")
        fake_rag.retrieve_from_vector_store = lambda *_args, **_kwargs: [
            {
                "source": "crm://owner/jane.doe@example.com",
                "text": "Contact Jane at jane.doe@example.com or +1 (555) 123-4567, SSN 123-45-6789.",
            }
        ]

        with patch.dict(sys.modules, {"src.orchestrator.rag": fake_rag}):
            profile_context = importlib.import_module("src.orchestrator.profile_context")
            profile_context = importlib.reload(profile_context)
            team_raw = {"globals": {"owner_profile": {"enabled": True}}}
            request_obj = {"topic": "wins"}

            with patch.dict("os.environ", {"VECTOR_DB_TABLE": "profiles"}, clear=False):
                result = profile_context.get_owner_profile_context(request_obj, team_raw, "Jane Doe")

        self.assertIn("[REDACTED_EMAIL]", result)
        self.assertIn("[REDACTED_PHONE]", result)
        self.assertIn("[REDACTED_SSN]", result)
        self.assertNotIn("jane.doe@example.com", result)
        self.assertNotIn("555", result)
        self.assertNotIn("123-45-6789", result)


if __name__ == "__main__":
    unittest.main()
