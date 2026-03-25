"""
document_tools.py
~~~~~~~~~~~~~~~~~
Pure-Python tools for parsing and reconstructing plain-text documents.

These are used as pre/post tools in workflow step definitions:

  pre_tools:
    - name: parse_document
      args:
        source_key: request.document_text

  post_tools:
    - name: reconstruct_document
      args:
        source_key: formatter
"""

import re
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MARKDOWN_HEADING = re.compile(r"^(#{1,3})\s+(.+)$")
_ALL_CAPS_HEADING = re.compile(r"^[A-Z][A-Z\s\-/&]{2,}$")
_COLON_HEADING = re.compile(r"^([A-Z][A-Za-z\s]{1,40}):$")


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _MARKDOWN_HEADING.match(stripped):
        return True
    if _ALL_CAPS_HEADING.match(stripped):
        return True
    if _COLON_HEADING.match(stripped):
        return True
    return False


def _clean_heading(line: str) -> str:
    stripped = line.strip()
    m = _MARKDOWN_HEADING.match(stripped)
    if m:
        return m.group(2).strip()
    return stripped.rstrip(":")


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def parse_document(document_text: str) -> Dict[str, Any]:
    """
    Split a plain-text or markdown document into labelled sections.

    Returns:
        {
            "sections": [
                {"index": 0, "header": "SUMMARY", "content": "..."},
                ...
            ],
            "raw": "<original text>",
            "section_count": <int>
        }

    If no headings are detected the entire text is returned as a single
    section with header "DOCUMENT".
    """
    lines = document_text.splitlines()

    sections: List[Dict[str, Any]] = []
    current_header = "DOCUMENT"
    current_lines: List[str] = []
    index = 0

    for line in lines:
        if _is_heading(line):
            # Flush accumulated content
            content = "\n".join(current_lines).strip()
            if content or sections:
                sections.append({
                    "index": index,
                    "header": current_header,
                    "content": content,
                })
                index += 1
            current_header = _clean_heading(line)
            current_lines = []
        else:
            current_lines.append(line)

    # Flush final section
    content = "\n".join(current_lines).strip()
    sections.append({
        "index": index,
        "header": current_header,
        "content": content,
    })

    # If we ended up with a single empty DOCUMENT section, treat the whole
    # text as one section.
    if len(sections) == 1 and not sections[0]["content"]:
        sections[0]["content"] = document_text.strip()

    return {
        "sections": sections,
        "raw": document_text,
        "section_count": len(sections),
    }


def reconstruct_document(sections: List[Dict[str, Any]], separator: str = "\n\n") -> Dict[str, Any]:
    """
    Rebuild a document string from a list of section dicts.

    Each section should have at minimum ``header`` and ``content`` keys.
    Sections are sorted by ``index`` if present.

    Returns:
        {
            "document_text": "<reconstructed document>",
            "section_count": <int>
        }
    """
    sorted_sections = sorted(sections, key=lambda s: s.get("index", 0))
    parts: List[str] = []
    for sec in sorted_sections:
        header = sec.get("header", "").strip()
        content = sec.get("content", "").strip()
        if header and header.upper() != "DOCUMENT":
            parts.append(f"{header}\n{content}" if content else header)
        else:
            if content:
                parts.append(content)

    document_text = separator.join(p for p in parts if p)
    return {
        "document_text": document_text,
        "section_count": len(sorted_sections),
    }
