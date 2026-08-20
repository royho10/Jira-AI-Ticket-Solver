"""
Unit tests for utility/helper functions that don't involve LLM calls.
"""
import pytest

from utils.jira_client import (
    JiraIssue,
    JiraComment,
    JiraAttachment,
    JiraRelatedIssue,
    extract_text_from_adf,
    extract_jira_keys_from_text,
)


class TestExtractTextFromADF:

    @pytest.mark.deterministic
    def test_simple_paragraph(self):
        adf = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "Hello world"}],
                }
            ],
        }
        assert extract_text_from_adf(adf).strip() == "Hello world"

    @pytest.mark.deterministic
    def test_code_block(self):
        adf = {
            "type": "doc",
            "content": [
                {
                    "type": "codeBlock",
                    "content": [{"type": "text", "text": "print('hello')"}],
                }
            ],
        }
        result = extract_text_from_adf(adf)
        assert "print('hello')" in result
        assert "```" in result

    @pytest.mark.deterministic
    def test_empty_node(self):
        assert extract_text_from_adf({}) == ""
        assert extract_text_from_adf(None) == ""

    @pytest.mark.deterministic
    def test_nested_content(self):
        adf = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": "Error: "},
                        {"type": "text", "text": "NullPointerException"},
                    ],
                }
            ],
        }
        result = extract_text_from_adf(adf)
        assert "Error: NullPointerException" in result


class TestJiraIssueFromDict:

    @pytest.mark.deterministic
    def test_basic_issue_parsing(self):
        raw = {
            "key": "GC-100",
            "fields": {
                "summary": "Test issue",
                "description": {"type": "doc", "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "Bug description"}]}
                ]},
                "labels": ["bug"],
                "status": {"name": "Open"},
                "priority": {"name": "High"},
                "issuetype": {"name": "Bug"},
                "created": "2024-01-01T00:00:00.000+0000",
                "comment": {"comments": []},
                "attachment": [],
                "issuelinks": [],
                "components": [{"name": "backend"}],
            },
        }
        issue = JiraIssue.from_dict(raw)
        assert issue.key == "GC-100"
        assert issue.summary == "Test issue"
        assert "Bug description" in issue.description
        assert issue.status == "Open"
        assert issue.priority == "High"
        assert issue.issue_type == "Bug"
        assert "backend" in issue.components

    @pytest.mark.deterministic
    def test_issue_with_comments(self):
        raw = {
            "key": "GC-101",
            "fields": {
                "summary": "With comments",
                "description": None,
                "labels": [],
                "status": {"name": "Open"},
                "created": "2024-01-01",
                "comment": {
                    "comments": [
                        {
                            "id": "1",
                            "author": {"displayName": "Alice"},
                            "body": {"type": "doc", "content": [
                                {"type": "paragraph", "content": [{"type": "text", "text": "Found the issue"}]}
                            ]},
                            "created": "2024-01-02",
                            "updated": "2024-01-02",
                        }
                    ]
                },
                "attachment": [],
                "issuelinks": [],
            },
        }
        issue = JiraIssue.from_dict(raw)
        assert len(issue.comments) == 1
        assert issue.comments[0].author_display_name == "Alice"
        assert "Found the issue" in issue.comments[0].body


class TestFileUtils:

    @pytest.mark.deterministic
    def test_extract_from_zip(self):
        import io
        import zipfile
        from utils.file_utils import extract_content_from_zip

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("app.log", "2024-01-01 ERROR something failed\n")
            zf.writestr("readme.md", "# Docs\n")  # Should be skipped
        buf.seek(0)

        result = extract_content_from_zip(buf.read())
        assert len(result) == 1
        assert result[0][1] == "app.log"
        assert "ERROR" in result[0][0]
