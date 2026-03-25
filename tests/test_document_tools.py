"""
tests/test_document_tools.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for document_tools.parse_document and reconstruct_document.
"""

import unittest

from src.orchestrator.tools.document_tools import parse_document, reconstruct_document


class ParseDocumentTests(unittest.TestCase):
    def test_splits_on_all_caps_headings(self):
        doc = "SUMMARY\nI am an engineer.\n\nEXPERIENCE\nWorked at Acme."
        result = parse_document(doc)
        headers = [s["header"] for s in result["sections"]]
        self.assertIn("SUMMARY", headers)
        self.assertIn("EXPERIENCE", headers)
        self.assertEqual(result["section_count"], 2)

    def test_splits_on_markdown_headings(self):
        doc = "# About Me\nSome text.\n\n## Skills\nPython, AWS"
        result = parse_document(doc)
        headers = [s["header"] for s in result["sections"]]
        self.assertIn("About Me", headers)
        self.assertIn("Skills", headers)

    def test_splits_on_colon_headings(self):
        doc = "Education:\nBSc Computer Science\n\nSkills:\nPython"
        result = parse_document(doc)
        headers = [s["header"] for s in result["sections"]]
        self.assertIn("Education", headers)
        self.assertIn("Skills", headers)

    def test_no_headings_returns_single_document_section(self):
        doc = "Just a plain paragraph with no headings at all."
        result = parse_document(doc)
        self.assertEqual(len(result["sections"]), 1)
        self.assertEqual(result["sections"][0]["header"], "DOCUMENT")
        self.assertIn("plain paragraph", result["sections"][0]["content"])

    def test_section_indexes_are_sequential(self):
        doc = "SUMMARY\nText.\n\nEXPERIENCE\nMore text.\n\nEDUCATION\nDegree."
        result = parse_document(doc)
        indexes = [s["index"] for s in result["sections"]]
        self.assertEqual(indexes, list(range(len(indexes))))

    def test_preserves_raw_text(self):
        doc = "SUMMARY\nI am an engineer."
        result = parse_document(doc)
        self.assertEqual(result["raw"], doc)

    def test_empty_string_returns_single_section(self):
        result = parse_document("")
        self.assertEqual(len(result["sections"]), 1)

    def test_content_trimmed_per_section(self):
        doc = "SUMMARY\n\n  I am a developer.  \n\nEXPERIENCE\nAcco Corp"
        result = parse_document(doc)
        summary = next(s for s in result["sections"] if s["header"] == "SUMMARY")
        self.assertEqual(summary["content"], "I am a developer.")


class ReconstructDocumentTests(unittest.TestCase):
    def test_reconstructs_with_headers(self):
        sections = [
            {"index": 0, "header": "SUMMARY", "content": "I am an engineer."},
            {"index": 1, "header": "EXPERIENCE", "content": "Worked at Acme."},
        ]
        result = reconstruct_document(sections)
        self.assertIn("SUMMARY", result["document_text"])
        self.assertIn("EXPERIENCE", result["document_text"])
        self.assertIn("I am an engineer.", result["document_text"])

    def test_sections_sorted_by_index(self):
        sections = [
            {"index": 1, "header": "EXPERIENCE", "content": "B"},
            {"index": 0, "header": "SUMMARY", "content": "A"},
        ]
        result = reconstruct_document(sections)
        summary_pos = result["document_text"].index("SUMMARY")
        experience_pos = result["document_text"].index("EXPERIENCE")
        self.assertLess(summary_pos, experience_pos)

    def test_section_count_returned(self):
        sections = [
            {"index": 0, "header": "A", "content": "text"},
            {"index": 1, "header": "B", "content": "more"},
        ]
        result = reconstruct_document(sections)
        self.assertEqual(result["section_count"], 2)

    def test_document_section_header_omitted_from_text(self):
        sections = [{"index": 0, "header": "DOCUMENT", "content": "Plain content here."}]
        result = reconstruct_document(sections)
        self.assertNotIn("DOCUMENT", result["document_text"])
        self.assertIn("Plain content here.", result["document_text"])

    def test_custom_separator(self):
        sections = [
            {"index": 0, "header": "A", "content": "first"},
            {"index": 1, "header": "B", "content": "second"},
        ]
        result = reconstruct_document(sections, separator="\n---\n")
        self.assertIn("\n---\n", result["document_text"])

    def test_empty_sections_list(self):
        result = reconstruct_document([])
        self.assertEqual(result["document_text"], "")
        self.assertEqual(result["section_count"], 0)

    def test_roundtrip_parse_then_reconstruct(self):
        original = "SUMMARY\nGreat engineer.\n\nSKILLS\nPython, AWS"
        parsed = parse_document(original)
        # Simulate rewritten sections keeping same structure
        rewritten = [
            {"index": s["index"], "header": s["header"], "content": s["content"]}
            for s in parsed["sections"]
        ]
        result = reconstruct_document(rewritten)
        self.assertIn("SUMMARY", result["document_text"])
        self.assertIn("SKILLS", result["document_text"])
        self.assertIn("Great engineer.", result["document_text"])


if __name__ == "__main__":
    unittest.main()
