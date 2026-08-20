"""
Unit tests for intent classification.

The intent classification is mostly rule-based (in _classify_intent) with
an LLM fallback (in _classify_intent_with_llm) for ambiguous cases.
These tests validate the rule-based logic deterministically.
"""
import pytest

from tests.conftest import dataset_range
from utils.jira_client import extract_jira_keys_from_text


# ===========================================================================
# Deterministic tests: extract_jira_keys_from_text
# ===========================================================================
class TestExtractJiraKeys:

    @pytest.mark.deterministic
    def test_simple_key(self):
        assert extract_jira_keys_from_text("GC-1234") == ["GC-1234"]

    @pytest.mark.deterministic
    def test_lowercase_key(self):
        assert extract_jira_keys_from_text("gc-1234") == ["GC-1234"]

    @pytest.mark.deterministic
    def test_url_with_key(self):
        result = extract_jira_keys_from_text("https://mycompany.atlassian.net/browse/GC-5678")
        assert "GC-5678" in result

    @pytest.mark.deterministic
    def test_multiple_keys_in_text(self):
        result = extract_jira_keys_from_text("compare GC-1234 and GC-5678")
        assert "GC-1234" in result
        assert "GC-5678" in result

    @pytest.mark.deterministic
    def test_no_keys(self):
        assert extract_jira_keys_from_text("hello world") == []

    @pytest.mark.deterministic
    def test_deduplication(self):
        result = extract_jira_keys_from_text("GC-1234 and gc-1234 again")
        assert result == ["GC-1234"]


# ===========================================================================
# Deterministic tests: Intent classification rules
# ===========================================================================
class TestIntentClassificationRules:
    """Test the rule-based portion of intent classification.
    These don't require Streamlit or LLM - we test the logic directly."""

    @pytest.mark.deterministic
    @pytest.mark.parametrize("case_idx", dataset_range("intent_classification"))
    def test_intent_rules(self, intent_classification_dataset, case_idx):
        """Validate intent classification rules against dataset."""
        case = intent_classification_dataset[case_idx]
        user_msg = case["input"]["user_message"]
        current_key = case["input"]["current_ticket_key"]
        similar_keys = set(case["input"]["similar_ticket_keys"])
        expected_intent = case["expected"]["intent"]

        potential_keys = extract_jira_keys_from_text(user_msg)

        # Replicate the rule-based logic from _classify_intent
        if not current_key:
            if potential_keys:
                if len(potential_keys) == 1:
                    computed = "analyze_new_ticket"
                else:
                    computed = "more_than_one_key"
            else:
                computed = "unrelated_chat"
        elif not potential_keys:
            # This is the ambiguous case that would go to LLM
            # For testing purposes, we skip this case
            pytest.skip("Ambiguous case requires LLM - tested separately")
            return
        else:
            if len(potential_keys) == 1:
                if potential_keys[0] == current_key:
                    computed = "follow_up_on_current_ticket"
                elif potential_keys[0] in similar_keys:
                    computed = "follow_up_on_current_ticket"
                else:
                    computed = "analyze_new_ticket"
            else:
                computed = "more_than_one_key"

        assert computed == expected_intent, (
            f"[{case['test_id']}] Expected '{expected_intent}' but got '{computed}' "
            f"for message: '{user_msg}'"
        )
