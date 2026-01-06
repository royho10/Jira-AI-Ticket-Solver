import re
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv
import os
from typing import Any, Dict, List, Optional, Tuple
import requests
from requests.auth import HTTPBasicAuth


load_dotenv()
ATLASSIAN_INSTANCE_URL = os.environ["ATLASSIAN_INSTANCE_URL"]
ATLASSIAN_EMAIL = os.environ["ATLASSIAN_EMAIL"]
ATLASSIAN_TOKEN = os.environ["ATLASSIAN_API_TOKEN"]
GUARDICORE_PROJECT_NAME = "Ticket Solver"


@dataclass
class JiraComment:
    id: str
    author_display_name: str
    body: str
    created: str
    updated: str

    @classmethod
    def from_dict(cls, comment_data: Dict[str, Any]) -> "JiraComment":
        author = comment_data.get("author", {})
        return cls(
            id=comment_data.get("id", ""),
            author_display_name=author.get("displayName", ""),
            body=comment_data.get("body", {}).get("content", [])[0].get("content", [])[0].get("text", ""),
            created=comment_data.get("created", ""),
            updated=comment_data.get("updated", ""),
        )


@dataclass
class JiraAttachment:
    id: str
    filename: str
    size: int
    mime_type: str
    content_url: str
    created: str
    author_display_name: str

    @classmethod
    def from_dict(cls, attachment_data: Dict[str, Any]) -> "JiraAttachment":
        author = attachment_data.get("author", {})
        return cls(
            id=attachment_data.get("id", ""),
            filename=attachment_data.get("filename", ""),
            size=attachment_data.get("size", 0),
            mime_type=attachment_data.get("mimeType", ""),
            content_url=attachment_data.get("content", ""),
            created=attachment_data.get("created", ""),
            author_display_name=author.get("displayName", ""),
        )


@dataclass
class JiraRelatedIssue:
    key: str
    relation_type: str  # e.g., "Duplicate", "Relates to", etc.
    direction: Optional[str] = None  # "inward" or "outward"

    @classmethod
    def from_dict(cls, issue_link_data: Dict[str, Any]) -> "JiraRelatedIssue":
        relation_type = issue_link_data.get("type", {}).get("name", "")
        outward_issue = issue_link_data.get("outwardIssue", {})
        inward_issue = issue_link_data.get("inwardIssue", {})
        if outward_issue:
            return cls(
                key=outward_issue.get("key", ""),
                relation_type=relation_type,
                direction="outward"
            )
        elif inward_issue:
            return cls(
                key=inward_issue.get("key", ""),
                relation_type=relation_type,
                direction="inward"
            )
        else:
            return cls(key="", relation_type=relation_type)


@dataclass
class JiraIssue:
    key: str
    summary: str
    description: Optional[str]
    labels: List[str]
    comments: List[JiraComment]
    attachments: Optional[List[JiraAttachment]]
    related_issues: Optional[List[JiraRelatedIssue]]
    priority: Optional[str]
    issue_type: Optional[str]
    components: Optional[List[str]]
    created: str
    status: Optional[str]

    @classmethod
    def from_dict(cls, issue_data: Dict[str, Any]) -> "JiraIssue":
        fields = issue_data.get("fields", {})

        # Parse comments
        comments_data = fields.get("comment", {}).get("comments", [])
        comments = [JiraComment.from_dict(comment) for comment in comments_data]

        # Parse attachments
        attachments_data = fields.get("attachment", [])
        attachments = [JiraAttachment.from_dict(att) for att in attachments_data]

        # Parse related issues
        issue_links = fields.get("issuelinks", [])
        related_issues = [JiraRelatedIssue.from_dict(issue_link) for issue_link in issue_links]

        # Parse description
        description = extract_text_from_adf(fields.get("description", {}))

        # Parse additional fields
        priority = fields.get("priority", {}).get("name") if fields.get("priority") else None
        issue_type = fields.get("issuetype", {}).get("name") if fields.get("issuetype", {}) else None
        created = fields.get("created", "")
        components = [comp.get("name") if comp else None for comp in fields.get("components", [])]
        status = fields.get("status", {}).get("name") if fields.get("status", {}) else None

        return cls(
            key=issue_data.get("key", ""),
            summary=fields.get("summary", ""),
            description=description,
            labels=fields.get("labels", []),
            comments=comments,
            attachments=attachments,
            priority=priority,
            issue_type=issue_type,
            related_issues=related_issues,
            created=created,
            components=components if components else None,
            status=status,
        )

    def __str__(self) -> str:
        return f"JiraIssue(key='{self.key}', summary='{self.summary}', status='{self.status}')"

    def __repr__(self) -> str:
        return self.__str__()

    # def to_dict(self) -> Dict[str, Any]:
    #     """Convert JiraIssue object to dictionary format for compatibility with existing code."""
    #     return {
    #         "key": self.key,
    #         "fields": {
    #             "summary": self.summary,
    #             "description": self.description,
    #             "status": {"name": self.status.name},
    #             "resolution": {"name": self.resolution.name} if self.resolution else None,
    #             "comment": {
    #                 "comments": [{"body": comment.body} for comment in self.comments]
    #             },
    #             "attachment": [
    #                 {
    #                     "id": att.id,
    #                     "filename": att.filename,
    #                     "mimeType": att.mime_type,
    #                     "content": att.content_url,
    #                     "size": att.size
    #                 } for att in self.attachments
    #             ],
    #             "project": {"key": self.project_key},
    #             "priority": {"name": self.priority} if self.priority else None,
    #             "issuetype": {"name": self.issue_type} if self.issue_type else None,
    #         },
    #     }


class JiraClient:
    JIRA_ISSUES_URL = "/rest/api/3/search/jql"
    JIRA_ATTACHMENT_CONTENT_URL = "/rest/api/3/attachment/content"

    def __init__(self):
        self.base_url = ATLASSIAN_INSTANCE_URL
        self.auth = HTTPBasicAuth(ATLASSIAN_EMAIL, ATLASSIAN_TOKEN)
        self.headers = {"Accept": "application/json"}

    def fetch_issues(
        self, jql="order by updated desc", max_results=50, next_page_token=None
    ) -> Tuple[List[JiraIssue], Optional[str]]:
        url = f"{self.base_url}{self.JIRA_ISSUES_URL}"
        params = {
            "jql": jql,
            "maxResults": max_results,
            "nextPageToken": next_page_token,
            "fields": "summary,description,comment,labels,status,resolution,attachment,project,priority,issuetype",
            "expand": "attachment",
            "project": GUARDICORE_PROJECT_NAME
        }
        response = requests.get(
            url, params=params, auth=self.auth, headers=self.headers
        )
        response.raise_for_status()

        data = response.json()
        issues = []
        for issue_data in data.get("issues", []):
            issues.append(JiraIssue.from_dict(issue_data))

        next_page_token = data.get("nextPageToken", None)

        return issues, next_page_token

    def fetch_issue_by_key(self, issue_key) -> JiraIssue:
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        fields = "summary,description,comment,labels,status,resolution,attachment,priority,issuetype,components,created"
        params = {
            "fields": fields,
            "expand": "attachment"
        }

        r = requests.get(url, params=params, auth=self.auth, headers=self.headers)
        r.raise_for_status()

        return JiraIssue.from_dict(r.json())

    def download_attachment(self, attachment_id: str) -> bytes:
        """Download attachment content"""
        response = requests.get(f'{self.base_url}{self.JIRA_ATTACHMENT_CONTENT_URL}/{attachment_id}', auth=self.auth)
        response.raise_for_status()
        return response.content


def extract_text_from_adf(node: Dict[str, Any]) -> str:
    """Recursively extract plain text from Jira ADF (Atlassian Document Format)."""
    if not node:
        return ""

    node_type = node.get("type")

    # leaf text node
    if node_type == "text":
        return node.get("text", "")

    # container nodes with children in "content"
    text_parts = []
    for child in node.get("content", []):
        text_parts.append(extract_text_from_adf(child))

    # formatting based on type
    if node_type == "paragraph":
        return "".join(text_parts) + "\n"
    if node_type == "codeBlock":
        return "```\n" + "".join(text_parts) + "\n```\n"
    if node_type == "heading":
        return "# " + "".join(text_parts) + "\n"

    # default
    return "".join(text_parts)


def extract_jira_keys_from_text(url_or_key: str) -> List[str]:
    """Extract JIRA keys from URL, exact key, or text containing keys."""
    stripped_url_or_key = url_or_key.strip()
    keys = []

    # Check if it's already a valid key (case-insensitive)
    if re.match(r"^[A-Z]+-\d+$", stripped_url_or_key, re.IGNORECASE):
        return [stripped_url_or_key.upper()]

    # Try to extract from URL
    try:
        parsed = urlparse(stripped_url_or_key)
        match = re.search(r"/browse/([A-Z]+-\d+)", parsed.path, re.IGNORECASE)
        if match:
            keys.append(match.group(1).upper())
    except Exception:
        pass

    # Find all keys anywhere in the text (case-insensitive)
    matches = re.findall(r'\b([A-Z]+-\d+)\b', stripped_url_or_key, re.IGNORECASE)
    keys.extend([m.upper() for m in matches])

    # Remove duplicates while preserving order
    return list(dict.fromkeys(keys))
