"""Renders a sanitized `TicketAnalysis` into an MCP tool response.

Claude gets three parts of the same analysis: a directive telling it what to do
with them, the finished user-facing report inside `<report>` tags, and the
structured form inside `<ticket_analysis>` tags.

The directive matters as much as the markdown. An MCP client collapses tool
output and shows the user the assistant's own message instead, so without being
told otherwise Claude paraphrases this report into prose and the strict,
scannable shape is lost. The report is the deliverable, not raw material.
"""
import json
from typing import List, Optional

from core.ticket_analyzer import TicketAnalysis
from utils.jira_client import ATLASSIAN_INSTANCE_URL

REPORT_OPEN_TAG = "<report>"
REPORT_CLOSE_TAG = "</report>"
STRUCTURED_OPEN_TAG = "<ticket_analysis>"
STRUCTURED_CLOSE_TAG = "</ticket_analysis>"

RENDER_DIRECTIVE = (
    "INSTRUCTIONS FOR THE ASSISTANT — read before replying:\n"
    "1. Everything between <report> and </report> is a finished, user-facing report. "
    "Output it to the user verbatim: same sections, same order, same emoji headers, "
    "same code blocks. Do not summarize it, reorder it, merge sections, rewrite it as "
    "prose, or drop the log excerpts.\n"
    "2. The two sections that matter most to this user are 'Errors Found in Logs' and "
    "'Similar Tickets'. They sit directly under the summary deliberately. Reproduce "
    "them in full -- never trim, sample or paraphrase them.\n"
    "3. Keep your own analysis, caveats and next steps out of the report. Add them "
    "after </report> under a separate heading if you have something to say.\n"
    # The tag name is spelled without its angle brackets on purpose: callers
    # locate the structured block with a plain substring search for the opening
    # tag, and a literal one here would match this directive first.
    "4. The trailing ticket_analysis JSON block is context for answering follow-up "
    "questions without another tool call. Never show it to the user."
)

# A log excerpt is evidence, so it is reproduced whole in a fenced block rather
# than truncated into inline code. This bound only guards against a runaway log
# dump crowding out the rest of the report.
MAX_EXCERPT_CHARS = 2000


def _browse_url(ticket_key: str) -> Optional[str]:
    if not ATLASSIAN_INSTANCE_URL:
        return None
    base = ATLASSIAN_INSTANCE_URL.replace("/rest/api/3", "").rstrip("/")
    return f"{base}/browse/{ticket_key}"


def _ticket_link(ticket_key: str) -> str:
    url = _browse_url(ticket_key)
    return f"[{ticket_key}]({url})" if url else ticket_key


def _bullets(items: List[str], empty: str) -> List[str]:
    kept = [item.strip() for item in items if item and item.strip().lower() not in ("", "unknown")]
    return [f"- {item}" for item in kept] if kept else [f"_{empty}_"]


def _excerpt_is_redundant(error_lines: str, exception_line: Optional[str]) -> bool:
    """Whether the excerpt adds nothing over the exception line already shown.

    Mirrors the Streamlit rendering: a truncated traceback next to the exception
    it raised is noise, and the extractor writes the literal "undefined" when it
    found no lines at all.
    """
    if not error_lines:
        return True
    if error_lines.lower() == "undefined":
        return True
    return bool(exception_line) and (
        error_lines.startswith("Traceback") or error_lines.startswith('File "')
    )


def _error_section(analysis: TicketAnalysis) -> List[str]:
    lines = ["## 🚨 Errors Found in Logs", ""]
    if not analysis.error_log_highlights:
        lines.append("_No errors found in attached logs._")
        return lines

    for highlight in analysis.error_log_highlights:
        lines.append(f"- **Log file:** `{highlight.log_filename}`")
        if highlight.source_code_filename:
            lines.append(f"  - **File in code:** `{highlight.source_code_filename}`")
        if highlight.context:
            lines.append(f"  - **Context:** {highlight.context}")
        if highlight.exception_line:
            lines.append(f"  - **Exception:** `{highlight.exception_line}`")

        excerpt = (highlight.error_lines or "").strip()
        if not _excerpt_is_redundant(excerpt, highlight.exception_line):
            # Fence at column 0: indented fences render as literal text in some
            # markdown viewers, which would show the backticks to the user.
            lines += ["", "```", excerpt[:MAX_EXCERPT_CHARS], "```", ""]
    return lines


def _similar_section(analysis: TicketAnalysis) -> List[str]:
    lines = ["## 🎯 Similar Tickets", ""]
    if not analysis.similar_tickets:
        lines.append("_No similar historical tickets found._")
        return lines

    for ticket in analysis.similar_tickets:
        headline = f"- **{_ticket_link(ticket.key)}**: {ticket.title}"
        lines.append(headline)

        state = " · ".join(
            part for part in (
                ticket.status,
                ticket.resolution if ticket.resolution and ticket.resolution != "Unresolved" else None,
                ticket.issue_type,
            ) if part
        )
        if state:
            lines.append(f"  - **Status:** {state}")
        if ticket.relevance_score is not None:
            lines.append(f"  - **Relevance:** {ticket.relevance_score}/10")
        reason = ticket.similarity_reason or ticket.relevance_reason
        if reason:
            lines.append(f"  - **Why similar:** {reason}")
    return lines


def format_analysis(analysis: TicketAnalysis) -> str:
    """The directive, the report for the user, then the structured form for Claude."""
    heading = f"# {_ticket_link(analysis.ticket_key)}"
    if analysis.ticket_title:
        heading += f": {analysis.ticket_title}"
    report: List[str] = [heading]

    facts = [
        f"**Status:** {analysis.ticket_status}" if analysis.ticket_status else None,
        f"**Priority:** {analysis.ticket_priority}" if analysis.ticket_priority else None,
        f"**Created:** {analysis.ticket_created}" if analysis.ticket_created else None,
        f"**Labels:** {', '.join(analysis.ticket_labels)}" if analysis.ticket_labels else None,
    ]
    stated_facts = [fact for fact in facts if fact]
    if stated_facts:
        report += ["", " | ".join(stated_facts)]
    report += ["", "---"]

    # The summary orients the reader, then the two evidence sections an engineer
    # opens the report for, then the interpretation of that evidence.
    report += ["", "## 📋 Summary", "", analysis.ticket_summary or "_No summary available._"]

    report += [""] + _error_section(analysis)
    report += [""] + _similar_section(analysis)

    report += ["", "## 🔍 Key Issues", ""] + _bullets(
        analysis.key_issues, "No key issues identified."
    )
    report += ["", "## 🧩 Root Causes", ""] + _bullets(
        analysis.root_causes, "No root causes identified."
    )
    report += ["", "## 💡 Suggested Solutions", ""] + _bullets(
        analysis.suggested_solutions, "No suggested solutions."
    )
    report += ["", "## ⚠️ Important Notes", ""] + _bullets(
        analysis.important_notes, "No additional notes."
    )

    structured = json.dumps(analysis.model_dump(), indent=2, ensure_ascii=False)
    return "\n".join([
        RENDER_DIRECTIVE,
        "",
        REPORT_OPEN_TAG,
        *report,
        REPORT_CLOSE_TAG,
        "",
        STRUCTURED_OPEN_TAG,
        structured,
        STRUCTURED_CLOSE_TAG,
    ])
