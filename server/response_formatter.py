"""Renders a sanitized `TicketAnalysis` into an MCP tool response.

Claude gets both halves of the same analysis: markdown, because the user reads
the tool output directly, and the structured form inside `<ticket_analysis>`
tags, because follow-up questions are answered from Claude's context rather than
a second round trip to this server.
"""
import json
from typing import List, Optional

from core.ticket_analyzer import TicketAnalysis
from utils.jira_client import ATLASSIAN_INSTANCE_URL

STRUCTURED_OPEN_TAG = "<ticket_analysis>"
STRUCTURED_CLOSE_TAG = "</ticket_analysis>"


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


def format_analysis(analysis: TicketAnalysis) -> str:
    """Markdown for the reader, followed by the structured form for Claude."""
    heading = f"# {_ticket_link(analysis.ticket_key)}"
    if analysis.ticket_title:
        heading += f": {analysis.ticket_title}"
    lines: List[str] = [heading]

    facts = [
        f"**Status:** {analysis.ticket_status}" if analysis.ticket_status else None,
        f"**Priority:** {analysis.ticket_priority}" if analysis.ticket_priority else None,
        f"**Created:** {analysis.ticket_created}" if analysis.ticket_created else None,
        f"**Labels:** {', '.join(analysis.ticket_labels)}" if analysis.ticket_labels else None,
    ]
    stated_facts = [fact for fact in facts if fact]
    if stated_facts:
        lines += ["", " | ".join(stated_facts)]

    lines += ["", "## Summary", "", analysis.ticket_summary or "_No summary available._"]
    lines += ["", "## Key Issues", ""] + _bullets(analysis.key_issues, "No key issues identified.")
    lines += ["", "## Root Causes", ""] + _bullets(analysis.root_causes, "No root causes identified.")

    lines += ["", "## Errors Found in Logs", ""]
    if analysis.error_log_highlights:
        for highlight in analysis.error_log_highlights:
            lines.append(f"- **{highlight.log_filename}** — {highlight.context}")
            if highlight.source_code_filename:
                lines.append(f"  - Source file: `{highlight.source_code_filename}`")
            if highlight.exception_line:
                lines.append(f"  - Exception: `{highlight.exception_line}`")
            if highlight.error_lines:
                lines.append(f"  - Log excerpt: `{highlight.error_lines[:500]}`")
    else:
        lines.append("_No errors found in attached logs._")

    lines += ["", "## Similar Tickets", ""]
    if analysis.similar_tickets:
        for ticket in analysis.similar_tickets:
            headline = f"- **{_ticket_link(ticket.key)}**: {ticket.title}"
            if ticket.status:
                headline += f" ({ticket.status})"
            lines.append(headline)
            if ticket.similarity_reason:
                lines.append(f"  - Why similar: {ticket.similarity_reason}")
            elif ticket.relevance_reason:
                lines.append(f"  - Why similar: {ticket.relevance_reason}")
    else:
        lines.append("_No similar historical tickets found._")

    lines += ["", "## Suggested Solutions", ""] + _bullets(
        analysis.suggested_solutions, "No suggested solutions."
    )
    lines += ["", "## Important Notes", ""] + _bullets(
        analysis.important_notes, "No additional notes."
    )

    structured = json.dumps(analysis.model_dump(), indent=2, ensure_ascii=False)
    lines += [
        "",
        STRUCTURED_OPEN_TAG,
        structured,
        STRUCTURED_CLOSE_TAG,
    ]
    return "\n".join(lines)
