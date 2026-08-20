"""
Unit tests for the reranking LLM call.

Deterministic tests validate the Pydantic schema.
LLM eval tests validate that the LLM correctly scores similar tickets.
"""
import pytest
from pydantic import ValidationError

from tests.conftest import dataset_range
from langchain_core.messages import SystemMessage, HumanMessage


# ===========================================================================
# Deterministic tests
# ===========================================================================
class TestRerankingDeterministic:

    @pytest.mark.deterministic
    def test_rerank_output_schema(self):
        """Validate RerankOutput Pydantic model."""
        # Import here to avoid Streamlit initialization
        from core.ticket_analyzer import RerankOutput, RerankTicketScore

        output = RerankOutput(
            scored_tickets=[
                RerankTicketScore(key="TEST-1", relevance_score=8, reason="Same error pattern"),
                RerankTicketScore(key="TEST-2", relevance_score=2, reason="Unrelated issue"),
            ]
        )
        assert len(output.scored_tickets) == 2
        assert output.scored_tickets[0].relevance_score == 8

    @pytest.mark.deterministic
    def test_rerank_ticket_score_bounds(self):
        """Scores should be integers (model defines int, no explicit bounds in Pydantic)."""
        from core.ticket_analyzer import RerankTicketScore

        score = RerankTicketScore(key="TEST-1", relevance_score=10, reason="Identical issue")
        assert score.relevance_score == 10

        score = RerankTicketScore(key="TEST-2", relevance_score=0, reason="Completely unrelated")
        assert score.relevance_score == 0


# ===========================================================================
# LLM eval tests
# ===========================================================================
class TestRerankingLLMEval:

    @pytest.mark.llm_eval
    @pytest.mark.parametrize("case_idx", dataset_range("reranking"))
    def test_reranking(self, llm_chat, reranking_dataset, case_idx):
        """Test reranking with real LLM against dataset cases."""
        from core.ticket_analyzer import RerankOutput

        case = reranking_dataset[case_idx]
        current = case["input"]["current_ticket"]
        candidates = case["input"]["candidates"]
        expected = case["expected"]

        candidate_details = "\n".join(
            f"- Key: {t['key']}, Title: {t['title']}, Summary: {t['summary'][:500]}"
            for t in candidates
        )

        rerank_prompt = f"""You are a Jira ticket similarity scorer. Given a current ticket and a list of candidate tickets,
score each candidate on a 0-10 relevance scale based on:
- Same error patterns or error messages
- Same component/service affected
- Same symptoms described
- Same root cause

Score guide: 0 = completely unrelated, 5 = moderately similar, 10 = nearly identical issue.

Current ticket:
- Key: {current['key']}
- Summary: {current['summary'][:1000]}
- Description: {current['description'][:1000]}

Candidate tickets:
{candidate_details}

Score each candidate. Return ALL candidates with their scores."""

        structured_llm = llm_chat.with_structured_output(RerankOutput)
        messages = [
            SystemMessage(content="You are an expert at comparing Jira tickets for similarity."),
            HumanMessage(content=rerank_prompt),
        ]
        result = structured_llm.invoke(messages)

        assert isinstance(result, RerankOutput)

        # All candidates scored
        scored_keys = {s.key for s in result.scored_tickets}
        expected_keys = {c["key"] for c in candidates}
        if expected["all_candidates_scored"]:
            assert scored_keys == expected_keys, (
                f"[{case['test_id']}] Not all candidates scored. "
                f"Expected {expected_keys}, got {scored_keys}"
            )

        # Scores in range
        if expected["scores_are_0_to_10"]:
            for s in result.scored_tickets:
                assert 0 <= s.relevance_score <= 10, (
                    f"[{case['test_id']}] Score out of range: {s.key} = {s.relevance_score}"
                )

        # Relative ordering
        score_map = {s.key: s.relevance_score for s in result.scored_tickets}
        if "highest_score_key" in expected and "lowest_score_key" in expected:
            high_key = expected["highest_score_key"]
            low_key = expected["lowest_score_key"]
            assert score_map.get(high_key, 0) > score_map.get(low_key, 10), (
                f"[{case['test_id']}] Expected {high_key} ({score_map.get(high_key)}) > "
                f"{low_key} ({score_map.get(low_key)})"
            )
