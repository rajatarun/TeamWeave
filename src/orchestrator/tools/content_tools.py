"""
content_tools.py
~~~~~~~~~~~~~~~~
Pre and post tools for the visibility team content marketing pipeline.

Pipeline integration summary
-----------------------------
director step (pre):
  extract_topic_keywords — normalises the raw topic into keyword signals
  so the Director agent grounds the brief in concrete terms.

editor step (pre):
  analyse_draft_quality — runs automated quality checks on the writer's
  drafts array before the editor sees them, injecting a quality report
  into context so the editor knows exactly what to fix.

distribution step (pre):
  compute_optimal_post_time — computes the next optimal LinkedIn posting
  slot (Tue/Thu, 8-10 am or 5-6 pm UTC) so the Distribution Manager
  gets a concrete timestamp rather than deriving one itself.

editor step (post):
  measure_post_quality — counts words, chars, and validates length of
  the polished final_post, adding post_metrics to the editor output.

distribution step (post):
  format_distribution_checklist — converts the checklist string array
  into a numbered plain-text list in checklist_formatted.

approval step (post):
  format_approval_decision — wraps APPROVED + revision_notes into a
  one-line decision_summary and adds a publish_ready boolean for
  downstream consumers.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 1. extract_topic_keywords  (pre-tool: director step)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the and or but in on at to of for with by from about as is are "
    "was were be been being have has had do does did will would could should "
    "may might i we you he she it they this that these those".split()
)

_TECH_DOMAINS = {
    "ai": "AI/ML",
    "ml": "AI/ML",
    "machine learning": "AI/ML",
    "llm": "AI/ML",
    "bedrock": "AWS",
    "aws": "AWS",
    "cloud": "Cloud",
    "serverless": "Cloud",
    "lambda": "AWS",
    "step functions": "AWS",
    "fintech": "Fintech",
    "governance": "Governance",
    "orchestration": "Orchestration",
    "agent": "Agentic AI",
    "multi-agent": "Agentic AI",
    "rag": "RAG/Vector",
    "vector": "RAG/Vector",
    "linkedin": "Social",
}


def extract_topic_keywords(topic: str) -> Dict[str, Any]:
    """
    Normalise a raw topic string into structured keyword signals.

    Returns:
        {
            "keywords": ["ai orchestration", "step functions", ...],
            "primary_entity": "AI Orchestration",
            "domain_tags": ["AWS", "Orchestration"],
            "word_count": 4
        }
    """
    if not topic or not isinstance(topic, str):
        return {"keywords": [], "primary_entity": "", "domain_tags": [], "word_count": 0}

    text = topic.lower().strip()
    # Remove punctuation except hyphens (preserve "multi-agent")
    text_clean = re.sub(r"[^\w\s-]", "", text)
    tokens = text_clean.split()

    # Multi-word phrase extraction (bigrams / known phrases)
    phrases: List[str] = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1:
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            if bigram in _TECH_DOMAINS:
                phrases.append(bigram)
                i += 2
                continue
        phrases.append(tokens[i])
        i += 1

    # Filter stopwords for keyword list
    keywords = [p for p in phrases if p not in _STOPWORDS]

    # Primary entity = first non-stopword token (title-cased)
    primary_entity = " ".join(
        t.capitalize() for t in (keywords[0].split() if keywords else [])
    )

    # Domain tag lookup
    domain_tags: List[str] = []
    lower_topic = topic.lower()
    for key, tag in _TECH_DOMAINS.items():
        if key in lower_topic and tag not in domain_tags:
            domain_tags.append(tag)

    return {
        "keywords": keywords,
        "primary_entity": primary_entity,
        "domain_tags": domain_tags,
        "word_count": len(tokens),
    }


# ---------------------------------------------------------------------------
# 2. analyse_draft_quality  (pre-tool: editor step)
# ---------------------------------------------------------------------------

_HYPE_WORDS = frozenset(
    "amazing incredible revolutionary game-changing groundbreaking "
    "unprecedented exceptional extraordinary fantastic brilliant superb "
    "game changer disruptive leverage synergy paradigm shift cutting-edge "
    "best-in-class world-class next-generation state-of-the-art".split()
)


def _get_post_text(drafts: List[Dict[str, Any]]) -> str:
    """Extract the first available post text from the drafts array."""
    if not drafts:
        return ""
    first = drafts[0] if isinstance(drafts, list) else {}
    return first.get("linkedin_post") or first.get("post") or ""


def analyse_draft_quality(drafts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run automated quality checks on the writer's drafts array.

    Injects a draft_quality_report into editor context so the editor
    knows exactly which issues to fix.

    Returns:
        {
            "draft_quality_report": {
                "word_count": int,
                "char_count": int,
                "hashtag_count": int,
                "hype_words_found": [...],
                "is_first_person": bool,
                "passes_length_check": bool,
                "issues": [...]
            }
        }
    """
    post_text = _get_post_text(drafts) if isinstance(drafts, list) else ""

    words = post_text.split()
    word_count = len(words)
    char_count = len(post_text)
    hashtags = re.findall(r"#\w+", post_text)
    hashtag_count = len(hashtags)

    lower_text = post_text.lower()
    hype_words_found = [w for w in _HYPE_WORDS if w in lower_text]

    first_person_markers = ["i ", "i've", "i'm", "my ", "we ", "our "]
    is_first_person = any(m in lower_text for m in first_person_markers)

    passes_length_check = 250 <= word_count <= 350
    passes_hashtag_check = 5 <= hashtag_count <= 6

    issues: List[str] = []
    if not passes_length_check:
        issues.append(f"word_count={word_count} (target 250-350)")
    if not passes_hashtag_check:
        issues.append(f"hashtag_count={hashtag_count} (target 5-6)")
    if hype_words_found:
        issues.append(f"hype_words={hype_words_found}")
    if not is_first_person:
        issues.append("missing first-person voice")

    return {
        "draft_quality_report": {
            "word_count": word_count,
            "char_count": char_count,
            "hashtag_count": hashtag_count,
            "hype_words_found": hype_words_found,
            "is_first_person": is_first_person,
            "passes_length_check": passes_length_check,
            "passes_hashtag_check": passes_hashtag_check,
            "issues": issues,
        }
    }


# ---------------------------------------------------------------------------
# 3. compute_optimal_post_time  (pre-tool: distribution step)
# ---------------------------------------------------------------------------

# LinkedIn peak slots: Tue/Thu, morning (09:00 UTC) or evening (17:00 UTC)
_OPTIMAL_DAYS = {1: "Tuesday", 3: "Thursday"}  # Monday=0
_MORNING_HOUR = 9
_EVENING_HOUR = 17


def compute_optimal_post_time(now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Compute the next optimal LinkedIn posting slot (Tue/Thu, 9am or 5pm UTC).

    An optional *now* parameter is accepted so tests can pass a fixed
    datetime without monkeypatching.

    Returns:
        {
            "next_post_slot": "2026-03-31T09:00:00+00:00",
            "day_of_week": "Tuesday",
            "time_slot": "morning_peak",
            "hours_from_now": 42
        }
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    # Start search from today at midnight so we always check both day-slots
    candidate = now.replace(hour=0, minute=0, second=0, microsecond=0)

    for _ in range(14):  # search up to 2 weeks ahead (1 iteration = 1 day)
        weekday = candidate.weekday()
        if weekday in _OPTIMAL_DAYS:
            # Try morning slot first, then evening
            for hour in (_MORNING_HOUR, _EVENING_HOUR):
                slot = candidate.replace(hour=hour)
                if slot > now:
                    delta_hours = max(1, int((slot - now).total_seconds() / 3600))
                    time_slot = "morning_peak" if hour == _MORNING_HOUR else "evening_peak"
                    return {
                        "next_post_slot": slot.isoformat(),
                        "day_of_week": _OPTIMAL_DAYS[weekday],
                        "time_slot": time_slot,
                        "hours_from_now": delta_hours,
                    }
        # Advance to the next calendar day
        candidate = candidate + timedelta(days=1)

    # Fallback: next Thursday morning (should never be reached)
    return {
        "next_post_slot": "",
        "day_of_week": "Thursday",
        "time_slot": "morning_peak",
        "hours_from_now": 0,
    }


# ---------------------------------------------------------------------------
# 4. measure_post_quality  (post-tool: editor step)
# ---------------------------------------------------------------------------


def measure_post_quality(final_post: str) -> Dict[str, Any]:
    """
    Measure key quality metrics of the polished final LinkedIn post.

    Returns:
        {
            "post_metrics": {
                "word_count": int,
                "char_count": int,
                "estimated_read_time_sec": int,
                "hashtag_count": int,
                "passes_length_check": bool,
                "passes_hashtag_check": bool,
                "quality_score": int   (0-100)
            }
        }
    """
    if not isinstance(final_post, str):
        final_post = str(final_post or "")

    words = final_post.split()
    word_count = len(words)
    char_count = len(final_post)
    hashtags = re.findall(r"#\w+", final_post)
    hashtag_count = len(hashtags)

    # LinkedIn average reading speed ~238 wpm
    estimated_read_time_sec = max(1, round((word_count / 238) * 60))

    passes_length = 250 <= word_count <= 350
    passes_hashtag = 5 <= hashtag_count <= 6

    # Simple quality score (0-100)
    score = 100
    if not passes_length:
        score -= 30
    if not passes_hashtag:
        score -= 20
    lower = final_post.lower()
    hype_hits = sum(1 for w in _HYPE_WORDS if w in lower)
    score -= min(hype_hits * 5, 25)
    score = max(0, score)

    return {
        "post_metrics": {
            "word_count": word_count,
            "char_count": char_count,
            "estimated_read_time_sec": estimated_read_time_sec,
            "hashtag_count": hashtag_count,
            "passes_length_check": passes_length,
            "passes_hashtag_check": passes_hashtag,
            "quality_score": score,
        }
    }


# ---------------------------------------------------------------------------
# 5. format_distribution_checklist  (post-tool: distribution step)
# ---------------------------------------------------------------------------


def format_distribution_checklist(checklist: List[str]) -> Dict[str, Any]:
    """
    Convert the checklist string array into a numbered plain-text list.

    Returns:
        {
            "checklist_formatted": "1. Reply to first 10 comments within 60 min\n2. ..."
        }
    """
    if not isinstance(checklist, list):
        checklist = []

    lines = [f"{i + 1}. {item.strip()}" for i, item in enumerate(checklist) if item]
    return {"checklist_formatted": "\n".join(lines)}


# ---------------------------------------------------------------------------
# 6. format_approval_decision  (post-tool: approval step)
# ---------------------------------------------------------------------------


def format_approval_decision(approval: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrap the director approval result in a concise decision summary.

    Returns:
        {
            "decision_summary": "[APPROVED] Ready to publish. ...",
            "publish_ready": bool
        }
    """
    if not isinstance(approval, dict):
        approval = {}

    approved = bool(approval.get("APPROVED", False))
    notes = (approval.get("revision_notes") or "").strip()
    status = "APPROVED" if approved else "REJECTED"
    summary = f"[{status}] {notes}" if notes else f"[{status}]"

    return {
        "decision_summary": summary,
        "publish_ready": approved,
    }
