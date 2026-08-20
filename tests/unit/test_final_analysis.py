"""
Unit tests for the final_analysis LLM call.

Tests the FinalAnalysisOutput generation that combines the current ticket
with similar tickets to produce the final analysis.
"""
import pytest

from tests.conftest import dataset_range
from langchain_core.messages import SystemMessage, HumanMessage


# ===========================================================================
# Deterministic tests
# ===========================================================================
class TestFinalAnalysisDeterministic:

    @pytest.mark.deterministic
    def test_final_analysis_output_schema(self):
        from core.ticket_analyzer import FinalAnalysisOutput, SimilarTicketInfo

        output = FinalAnalysisOutput(
            ticket_summary="Test summary",
            key_issues=["Issue 1"],
            root_causes=["Cause 1"],
            similar_tickets=[
                SimilarTicketInfo(
                    key="TEST-1",
                    summary="Similar ticket",
                    similarity_reason="Same error",
                    status="Resolved",
                    title="Similar title",
                    score=0.85,
                    issue_type="Bug",
                )
            ],
            suggested_solutions=["Fix the config"],
            important_notes=["Check DB logs"],
        )
        assert output.ticket_summary == "Test summary"
        assert len(output.similar_tickets) == 1
        assert output.similar_tickets[0].key == "TEST-1"

    @pytest.mark.deterministic
    def test_final_analysis_empty_similar_tickets(self):
        from core.ticket_analyzer import FinalAnalysisOutput

        output = FinalAnalysisOutput(
            ticket_summary="No similar tickets found",
            key_issues=["Issue 1"],
            root_causes=["unknown"],
            similar_tickets=[],
            suggested_solutions=[],
            important_notes=[],
        )
        assert len(output.similar_tickets) == 0


# ===========================================================================
# LLM eval tests
# ===========================================================================
class TestFinalAnalysisLLMEval:

    @pytest.mark.llm_eval
    @pytest.mark.parametrize("case_idx", dataset_range("final_analysis"))
    def test_final_analysis(self, llm_chat, final_analysis_dataset, case_idx):
        """Test final analysis generation with real LLM."""
        from core.ticket_analyzer import FinalAnalysisOutput

        case = final_analysis_dataset[case_idx]
        current = case["input"]["current_ticket"]
        similar_tickets = case["input"]["similar_tickets"]
        expected = case["expected"]

        # Build similar tickets text
        similar_details = []
        for idx, ticket in enumerate(similar_tickets, 1):
            similar_details.append(f"""
        Ticket {idx}:
        - Key: {ticket['key']}
        - Title: {ticket['title']}
        - Summary: {ticket['summary'][:2400]}
        - Status: {ticket['status']}
        - Resolution: {ticket.get('resolution', 'Unresolved')}
        - Labels: {ticket.get('labels', [])}
        - Issue Type: {ticket.get('issue_type', 'Unknown')}
        """)
        similar_formatted = "\n".join(similar_details).strip() or "NONE"

        system_prompt = """You are an expert Jira analyst. Analyze a Jira ticket with similar tickets and produce a structured analysis.
Rules: Base conclusions ONLY on provided info. Do NOT hallucinate. Be concise and technical."""

        input_prompt = f"""
        Current Jira ticket:
        - Key: {current['key']}
        - Summary: {current['summary'][:2400]}
        - Description: {current['description'][:1600]}
        - Status: {current['status']}
        - Priority: {current['priority']}
        - Labels: {current['labels']}

        Similar tickets (reranked, {len(similar_tickets)} total):
        {similar_formatted}

        RULES:
        - Return similar_tickets with exactly the tickets provided. If none, return empty list.
        - Do NOT invent ticket keys.
        """

        structured_llm = llm_chat.with_structured_output(FinalAnalysisOutput)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=input_prompt),
        ]
        result = structured_llm.invoke(messages)

        assert isinstance(result, FinalAnalysisOutput)

        # Has summary
        if expected.get("has_ticket_summary"):
            assert len(result.ticket_summary) > 10

        # Key issues count
        if "min_key_issues" in expected:
            assert len(result.key_issues) >= expected["min_key_issues"]

        # Root causes count
        if "min_root_causes" in expected:
            assert len(result.root_causes) >= expected["min_root_causes"]

        # Similar tickets validation
        result_keys = {t.key for t in result.similar_tickets}
        if "similar_ticket_keys" in expected:
            for key in expected["similar_ticket_keys"]:
                assert key in result_keys, (
                    f"[{case['test_id']}] Expected similar ticket {key} not in output: {result_keys}"
                )

        if expected.get("similar_tickets_empty"):
            assert len(result.similar_tickets) == 0, (
                f"[{case['test_id']}] Expected empty similar_tickets but got {len(result.similar_tickets)}"
            )

        if expected.get("must_not_invent_tickets"):
            allowed_keys = set(expected.get("similar_ticket_keys", []))
            allowed_keys.add(current["key"])
            for t in result.similar_tickets:
                assert t.key in allowed_keys, (
                    f"[{case['test_id']}] Hallucinated ticket key: {t.key}"
                )

        # Summary keywords
        if "summary_must_contain" in expected:
            summary_lower = result.ticket_summary.lower()
            for kw in expected["summary_must_contain"]:
                assert kw.lower() in summary_lower, (
                    f"[{case['test_id']}] Expected '{kw}' in summary: {result.ticket_summary}"
                )
