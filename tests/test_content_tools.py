"""
tests/test_content_tools.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the visibility-team content tools.
"""

import unittest
from datetime import datetime, timezone

from src.orchestrator.tools.content_tools import (
    analyse_draft_quality,
    compute_optimal_post_time,
    extract_topic_keywords,
    format_approval_decision,
    format_distribution_checklist,
    measure_post_quality,
)


# ---------------------------------------------------------------------------
# extract_topic_keywords
# ---------------------------------------------------------------------------


class ExtractTopicKeywordsTests(unittest.TestCase):
    def test_returns_keywords_list(self):
        result = extract_topic_keywords("AI orchestration with AWS Lambda")
        self.assertIn("keywords", result)
        self.assertIsInstance(result["keywords"], list)
        self.assertTrue(len(result["keywords"]) > 0)

    def test_filters_stopwords(self):
        result = extract_topic_keywords("the power of AI in the cloud")
        keywords = result["keywords"]
        for kw in keywords:
            self.assertNotIn(kw, {"the", "of", "in"})

    def test_identifies_domain_tags(self):
        result = extract_topic_keywords("AWS Bedrock multi-agent orchestration")
        self.assertIn("AWS", result["domain_tags"])
        self.assertIn("Orchestration", result["domain_tags"])

    def test_returns_primary_entity(self):
        result = extract_topic_keywords("fintech governance platform")
        self.assertIsInstance(result["primary_entity"], str)
        self.assertTrue(len(result["primary_entity"]) > 0)

    def test_empty_topic_returns_empty_structure(self):
        result = extract_topic_keywords("")
        self.assertEqual(result["keywords"], [])
        self.assertEqual(result["domain_tags"], [])
        self.assertEqual(result["word_count"], 0)

    def test_none_topic_returns_empty_structure(self):
        result = extract_topic_keywords(None)
        self.assertEqual(result["keywords"], [])

    def test_word_count_matches_token_count(self):
        result = extract_topic_keywords("cloud native serverless architecture")
        self.assertEqual(result["word_count"], 4)


# ---------------------------------------------------------------------------
# analyse_draft_quality
# ---------------------------------------------------------------------------


def _make_drafts(post_text: str) -> list:
    return [{"linkedin_post": post_text, "variant": "A", "hashtags": []}]


class AnalyseDraftQualityTests(unittest.TestCase):
    def _good_post(self) -> str:
        # ~270 words, 5 hashtags, first-person, no hype
        words = "I built a system that changed how we think about orchestration. " * 18
        return words.strip() + "\n\n#AI #AWS #Cloud #Fintech #Orchestration"

    def test_returns_draft_quality_report_key(self):
        result = analyse_draft_quality(_make_drafts(self._good_post()))
        self.assertIn("draft_quality_report", result)

    def test_word_count_in_report(self):
        post = " ".join(["word"] * 300) + " #tag1 #tag2 #tag3 #tag4 #tag5"
        result = analyse_draft_quality(_make_drafts(post))
        report = result["draft_quality_report"]
        self.assertGreater(report["word_count"], 0)

    def test_detects_hashtags(self):
        post = "Some content here.\n#AI #AWS #Cloud #Fintech #Orchestration"
        result = analyse_draft_quality(_make_drafts(post))
        self.assertEqual(result["draft_quality_report"]["hashtag_count"], 5)

    def test_detects_hype_words(self):
        post = "This is an incredible and amazing platform. " * 5
        post += "#A #B #C #D #E"
        result = analyse_draft_quality(_make_drafts(post))
        report = result["draft_quality_report"]
        self.assertTrue(len(report["hype_words_found"]) > 0)

    def test_detects_first_person(self):
        post = "I built this system. #A #B #C #D #E"
        result = analyse_draft_quality(_make_drafts(post))
        self.assertTrue(result["draft_quality_report"]["is_first_person"])

    def test_missing_first_person(self):
        post = "The system was built for scale. #A #B #C #D #E"
        result = analyse_draft_quality(_make_drafts(post))
        self.assertFalse(result["draft_quality_report"]["is_first_person"])

    def test_length_check_fails_short_post(self):
        post = "Short post. #AI"
        result = analyse_draft_quality(_make_drafts(post))
        self.assertFalse(result["draft_quality_report"]["passes_length_check"])

    def test_empty_drafts_list(self):
        result = analyse_draft_quality([])
        report = result["draft_quality_report"]
        self.assertEqual(report["word_count"], 0)

    def test_falls_back_to_post_field(self):
        drafts = [{"post": "I wrote this. #A #B #C #D #E", "variant": "A", "hashtags": []}]
        result = analyse_draft_quality(drafts)
        self.assertTrue(result["draft_quality_report"]["is_first_person"])


# ---------------------------------------------------------------------------
# compute_optimal_post_time
# ---------------------------------------------------------------------------


class ComputeOptimalPostTimeTests(unittest.TestCase):
    def _monday(self) -> datetime:
        # 2026-03-23 is a Monday at 10:00 UTC
        return datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)

    def _tuesday_midnight(self) -> datetime:
        return datetime(2026, 3, 24, 0, 0, 0, tzinfo=timezone.utc)

    def _thursday_afternoon(self) -> datetime:
        # Thursday 2026-03-26 at 14:00 UTC (after morning slot, before evening)
        return datetime(2026, 3, 26, 14, 0, 0, tzinfo=timezone.utc)

    def test_returns_required_keys(self):
        result = compute_optimal_post_time(now=self._monday())
        for key in ("next_post_slot", "day_of_week", "time_slot", "hours_from_now"):
            self.assertIn(key, result)

    def test_from_monday_returns_tuesday(self):
        result = compute_optimal_post_time(now=self._monday())
        self.assertEqual(result["day_of_week"], "Tuesday")

    def test_from_monday_time_slot_is_morning(self):
        result = compute_optimal_post_time(now=self._monday())
        self.assertEqual(result["time_slot"], "morning_peak")

    def test_from_tuesday_midnight_returns_tuesday_morning(self):
        result = compute_optimal_post_time(now=self._tuesday_midnight())
        self.assertEqual(result["day_of_week"], "Tuesday")
        self.assertEqual(result["time_slot"], "morning_peak")

    def test_from_thursday_afternoon_returns_thursday_evening(self):
        result = compute_optimal_post_time(now=self._thursday_afternoon())
        self.assertEqual(result["day_of_week"], "Thursday")
        self.assertEqual(result["time_slot"], "evening_peak")

    def test_hours_from_now_is_positive(self):
        result = compute_optimal_post_time(now=self._monday())
        self.assertGreater(result["hours_from_now"], 0)

    def test_next_post_slot_is_iso_string(self):
        result = compute_optimal_post_time(now=self._monday())
        # Should be parseable as ISO datetime
        dt = datetime.fromisoformat(result["next_post_slot"])
        self.assertIsInstance(dt, datetime)

    def test_called_without_args_returns_valid_result(self):
        result = compute_optimal_post_time()
        self.assertIn("next_post_slot", result)
        self.assertIn(result["day_of_week"], ("Tuesday", "Thursday"))


# ---------------------------------------------------------------------------
# measure_post_quality
# ---------------------------------------------------------------------------


class MeasurePostQualityTests(unittest.TestCase):
    def _good_post(self) -> str:
        return " ".join(["word"] * 280) + " #AI #AWS #Cloud #Fintech #Orchestration"

    def test_returns_post_metrics_key(self):
        result = measure_post_quality(self._good_post())
        self.assertIn("post_metrics", result)

    def test_word_count_correct(self):
        post = " ".join(["word"] * 300) + " #A #B #C #D #E"
        result = measure_post_quality(post)
        self.assertEqual(result["post_metrics"]["word_count"], 305)

    def test_hashtag_count(self):
        post = "content " * 10 + "#AI #AWS #Cloud #Fintech #Orchestration"
        result = measure_post_quality(post)
        self.assertEqual(result["post_metrics"]["hashtag_count"], 5)

    def test_passes_length_check_for_valid_post(self):
        result = measure_post_quality(self._good_post())
        self.assertTrue(result["post_metrics"]["passes_length_check"])

    def test_fails_length_check_for_short_post(self):
        result = measure_post_quality("Too short. #AI")
        self.assertFalse(result["post_metrics"]["passes_length_check"])

    def test_quality_score_between_0_and_100(self):
        result = measure_post_quality(self._good_post())
        score = result["post_metrics"]["quality_score"]
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_hype_words_reduce_quality_score(self):
        good = measure_post_quality(self._good_post())
        hype = measure_post_quality(self._good_post() + " incredible amazing revolutionary")
        self.assertGreaterEqual(good["post_metrics"]["quality_score"],
                                hype["post_metrics"]["quality_score"])

    def test_none_input_handled_gracefully(self):
        result = measure_post_quality(None)
        self.assertIn("post_metrics", result)
        self.assertEqual(result["post_metrics"]["word_count"], 0)

    def test_estimated_read_time_positive(self):
        post = " ".join(["word"] * 280) + " #A #B #C #D #E"
        result = measure_post_quality(post)
        self.assertGreater(result["post_metrics"]["estimated_read_time_sec"], 0)


# ---------------------------------------------------------------------------
# format_distribution_checklist
# ---------------------------------------------------------------------------


class FormatDistributionChecklistTests(unittest.TestCase):
    def test_formats_as_numbered_list(self):
        checklist = ["Reply to first 10 comments", "Tag relevant people", "Cross-post to newsletter"]
        result = format_distribution_checklist(checklist)
        text = result["checklist_formatted"]
        self.assertIn("1. Reply", text)
        self.assertIn("2. Tag", text)
        self.assertIn("3. Cross-post", text)

    def test_returns_checklist_formatted_key(self):
        result = format_distribution_checklist(["Do thing"])
        self.assertIn("checklist_formatted", result)

    def test_empty_list_returns_empty_string(self):
        result = format_distribution_checklist([])
        self.assertEqual(result["checklist_formatted"], "")

    def test_non_list_handled_gracefully(self):
        result = format_distribution_checklist(None)
        self.assertEqual(result["checklist_formatted"], "")

    def test_strips_whitespace_from_items(self):
        result = format_distribution_checklist(["  Do something  "])
        self.assertIn("Do something", result["checklist_formatted"])
        self.assertNotIn("  Do something  ", result["checklist_formatted"])

    def test_items_separated_by_newlines(self):
        result = format_distribution_checklist(["A", "B", "C"])
        lines = result["checklist_formatted"].split("\n")
        self.assertEqual(len(lines), 3)


# ---------------------------------------------------------------------------
# format_approval_decision
# ---------------------------------------------------------------------------


class FormatApprovalDecisionTests(unittest.TestCase):
    def test_approved_sets_publish_ready_true(self):
        approval = {"APPROVED": True, "revision_notes": "Looks great.", "jump_to_step": ""}
        result = format_approval_decision(approval)
        self.assertTrue(result["publish_ready"])

    def test_rejected_sets_publish_ready_false(self):
        approval = {"APPROVED": False, "revision_notes": "Hook needs work.", "jump_to_step": "writer"}
        result = format_approval_decision(approval)
        self.assertFalse(result["publish_ready"])

    def test_decision_summary_contains_approved_label(self):
        approval = {"APPROVED": True, "revision_notes": "Good.", "jump_to_step": ""}
        result = format_approval_decision(approval)
        self.assertIn("[APPROVED]", result["decision_summary"])

    def test_decision_summary_contains_rejected_label(self):
        approval = {"APPROVED": False, "revision_notes": "Fix the hook.", "jump_to_step": "writer"}
        result = format_approval_decision(approval)
        self.assertIn("[REJECTED]", result["decision_summary"])

    def test_revision_notes_included_in_summary(self):
        approval = {"APPROVED": False, "revision_notes": "Hook is too generic.", "jump_to_step": "writer"}
        result = format_approval_decision(approval)
        self.assertIn("Hook is too generic.", result["decision_summary"])

    def test_returns_required_keys(self):
        result = format_approval_decision({"APPROVED": True, "revision_notes": "", "jump_to_step": ""})
        self.assertIn("decision_summary", result)
        self.assertIn("publish_ready", result)

    def test_non_dict_input_handled_gracefully(self):
        result = format_approval_decision(None)
        self.assertFalse(result["publish_ready"])
        self.assertIn("[REJECTED]", result["decision_summary"])


# ---------------------------------------------------------------------------
# Integration: output_key and deep path resolution via tool_registry
# ---------------------------------------------------------------------------


class RegistryIntegrationTests(unittest.TestCase):
    """Verify the new tools work end-to-end through the registry."""

    def test_content_tools_registered(self):
        from src.orchestrator.tool_registry import TOOL_REGISTRY
        for name in (
            "extract_topic_keywords",
            "analyse_draft_quality",
            "compute_optimal_post_time",
            "measure_post_quality",
            "format_distribution_checklist",
            "format_approval_decision",
        ):
            self.assertIn(name, TOOL_REGISTRY, f"{name} not registered")

    def test_deep_path_resolution(self):
        from src.orchestrator.tool_registry import _resolve_source_key

        step_inputs = {
            "TVT_DEPT-003_PBM-006_writer_linkedin": {
                "drafts": [
                    {"linkedin_post": "I built this. #AI", "variant": "A", "hashtags": []}
                ]
            }
        }
        resolved = _resolve_source_key(
            "TVT_DEPT-003_PBM-006_writer_linkedin.drafts", step_inputs
        )
        self.assertIsInstance(resolved, list)
        self.assertEqual(resolved[0]["variant"], "A")

    def test_output_key_overrides_param_name(self):
        from src.orchestrator.tool_registry import _build_tool_args

        step_inputs = {
            "TVT_DEPT-001_PBM-001A_director_approve": {
                "APPROVED": True,
                "revision_notes": "Great post.",
                "jump_to_step": "",
            }
        }
        tool_cfg = {
            "name": "format_approval_decision",
            "args": {
                "source_key": "TVT_DEPT-001_PBM-001A_director_approve",
                "output_key": "approval",
            },
        }
        args = _build_tool_args(tool_cfg, step_inputs)
        self.assertIn("approval", args)
        self.assertNotIn("TVT_DEPT-001_PBM-001A_director_approve", args)
        self.assertTrue(args["approval"]["APPROVED"])

    def test_execute_pre_tools_with_compute_optimal_post_time(self):
        from src.orchestrator.tool_registry import execute_pre_tools

        step_def = {
            "step": "TVT_DEPT-003_PBM-004_distribution",
            "pre_tools": [{"name": "compute_optimal_post_time", "args": {}}],
        }
        step_inputs = {"request": {}}
        result = execute_pre_tools(step_def, step_inputs)
        self.assertIn("tool_results", result)
        self.assertIn("compute_optimal_post_time", result["tool_results"])
        self.assertIn("next_post_slot", result["tool_results"]["compute_optimal_post_time"])


if __name__ == "__main__":
    unittest.main()
