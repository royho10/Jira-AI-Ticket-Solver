from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional
import requests

from dotenv import load_dotenv


load_dotenv()
ATLASSIAN_INSTANCE_URL = os.environ["ATLASSIAN_INSTANCE_URL"]
ATLASSIAN_EMAIL = os.environ["ATLASSIAN_EMAIL"]
ATLASSIAN_TOKEN = os.environ["ATLASSIAN_API_TOKEN"]


@dataclass
class JiraComment:
    id: str
    author_display_name: str
    author_email: str
    body: str
    created: str
    updated: str

    @classmethod
    def from_dict(cls, comment_data: Dict[str, Any]) -> "JiraComment":
        author = comment_data.get("author", {})
        return cls(
            id=comment_data.get("id", ""),
            author_display_name=author.get("displayName", ""),
            author_email=author.get("emailAddress", ""),
            body=comment_data.get("body", ""),
            created=comment_data.get("created", ""),
            updated=comment_data.get("updated", ""),
        )


@dataclass
class JiraStatus:
    id: str
    name: str
    category: str

    @classmethod
    def from_dict(cls, status_data: Dict[str, Any]) -> "JiraStatus":
        status_category = status_data.get("statusCategory", {})
        return cls(
            id=status_data.get("id", ""),
            name=status_data.get("name", ""),
            category=status_category.get("name", ""),
        )


@dataclass
class JiraResolution:
    id: str
    name: str
    description: str

    @classmethod
    def from_dict(cls, resolution_data: Dict[str, Any]) -> "JiraResolution":
        return cls(
            id=resolution_data.get("id", ""),
            name=resolution_data.get("name", ""),
            description=resolution_data.get("description", ""),
        )


@dataclass
class JiraIssue:
    key: str
    summary: str
    description: Optional[str]
    labels: List[str]
    status: JiraStatus
    resolution: Optional[JiraResolution]
    comments: List[JiraComment]

    @classmethod
    def from_dict(cls, issue_data: Dict[str, Any]) -> "JiraIssue":
        fields = issue_data.get("fields", {})

        # Parse comments
        comments_data = fields.get("comment", {}).get("comments", [])
        comments = [JiraComment.from_dict(comment) for comment in comments_data]

        # Parse status
        status_data = fields.get("status", {})
        status = JiraStatus.from_dict(status_data)

        # Parse resolution (can be None)
        resolution_data = fields.get("resolution")
        resolution = (
            JiraResolution.from_dict(resolution_data) if resolution_data else None
        )

        return cls(
            key=issue_data.get("key", ""),
            summary=fields.get("summary", ""),
            description=fields.get("description", ""),
            labels=fields.get("labels", []),
            status=status,
            resolution=resolution,
            comments=comments,
        )

    def __str__(self) -> str:
        return f"JiraIssue(key='{self.key}', summary='{self.summary}', status='{self.status.name}')"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert JiraIssue object to dictionary format for compatibility with existing code."""
        return {
            "key": self.key,
            "fields": {
                "summary": self.summary,
                "description": self.description,
                "status": {"name": self.status.name},
                "resolution": {"name": self.resolution.name}
                if self.resolution
                else None,
                "comment": {
                    "comments": [{"body": comment.body} for comment in self.comments]
                },
            },
        }


class JiraClient:
    def __init__(self):
        self.base_url = ATLASSIAN_INSTANCE_URL
        self.auth = (ATLASSIAN_EMAIL, ATLASSIAN_TOKEN)
        self.headers = {"Accept": "application/json"}

    def fetch_issues(
        self, jql="order by updated desc", max_results=50, start_at=0
    ) -> List[JiraIssue]:
        url = f"{self.base_url}/rest/api/3/search"
        params = {
            "jql": jql,
            "maxResults": max_results,
            "startAt": start_at,
            "fields": "summary,description,comment,labels,status,resolution",
        }
        response = requests.get(
            url, params=params, auth=self.auth, headers=self.headers
        )
        response.raise_for_status()

        data = response.json()
        issues = []
        for issue_data in data.get("issues", []):
            issues.append(JiraIssue.from_dict(issue_data))

        return issues

    def fetch_issue_by_key(self, issue_key) -> JiraIssue:
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        params = {"fields": "summary,description,comment,labels,status,resolution"}
        r = requests.get(url, params=params, auth=self.auth, headers=self.headers)
        r.raise_for_status()

        return JiraIssue.from_dict(r.json())
