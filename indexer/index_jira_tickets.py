from enum import Enum
from typing import List

import weaviate
import weaviate.classes as wvc

from langchain_community.embeddings import OllamaEmbeddings
from weaviate.classes.config import Property, DataType
from weaviate.classes.data import DataObject

from utils.jira_client import JiraClient, JiraIssue
from utils.jira_ticket_processing import JiraIssueLLMProcessor

# -------------------------------
# CONFIG
# -------------------------------
EMBEDDING_MODEL_NAME = "nomic-embed-text"
VLM_MODEL_NAME = "qwen2.5vl:7b"
LLM_MODEL_NAME = "llama3"
MAX_LOG_FILES_TO_PROCESS = 20  # Max number of log files to process within a zip
MAX_LINES_PER_LOG = 500  # hard cap on lines per log file to process
CONTEXT_LINES_BEFORE_ERROR = 10  # Number of context lines to keep before errors in logs
CONTEXT_LINES_AFTER_ERROR = 20  # Number of context lines to keep after errors in logs
WEAVIATE_BATCH_SIZE = 100  # Number of objects to batch insert into Weaviate at once


class ContentType(Enum):
    SUMMARY = "summary"
    DESCRIPTION = "description"
    COMMENT = "comment"
    ATTACHMENT_TEXT = "attachment_text"
    ATTACHMENT_LOG = "attachment_log"
    ATTACHMENT_IMAGE_TEXT = "attachment_image_text"


class Componenets(Enum):
    MANAGEMENT = "management"
    AGGREGATOR = "aggregator"
    AGENT = "agent"


class JiraIndexer:
    def __init__(self):
        self.db_client = weaviate.connect_to_local()
        self.jira_client = JiraClient()
        self.embedding_model = OllamaEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            base_url="http://localhost:11434"  # Explicitly set Ollama URL
        )

        # Setup collection
        self._setup_collection()

    def _setup_collection(self):
        """Setup Weaviate collection schema if not exists"""
        if self.db_client.collections.exists("JiraCollection"):
            self.jira_collection = self.db_client.collections.get("JiraCollection")
            return
        else:
            collection_name = "JiraTicketChunk"

            # Create new collection
            self.jira_collection = self.db_client.collections.create(
                name=collection_name,
                properties=[
                    # Core fields
                    Property(name="issue_key", data_type=DataType.TEXT),
                    Property(name="summary", data_type=DataType.TEXT),
                    Property(name="clean_description", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),

                    # OPTIONAL FILTER FIELDS
                    Property(name="issue_type", data_type=DataType.TEXT),
                    Property(name="priority", data_type=DataType.TEXT),
                    Property(name="project_key", data_type=DataType.TEXT),
                    Property(name="labels", data_type=DataType.TEXT_ARRAY),
                    Property(name="squad", data_type=DataType.TEXT),
                    Property(name="components", data_type=DataType.TEXT_ARRAY),
                    Property(name="created", data_type=DataType.DATE),
                    Property(name="status", data_type=DataType.TEXT),
                ],
                # Configure for similarity search
                vector_config=wvc.config.Configure.VectorIndex.none(),
            )

    def _insert_issues_data_objects_to_db(self, issues_data_objects: List[DataObject]) -> int:
        total_chunks = 0
        """Batch insert issues into weaviate DB in WEAVIATE_BATCH_SIZE batches"""
        for i in range(0, len(issues_data_objects), WEAVIATE_BATCH_SIZE):
            batch = issues_data_objects[i:i + WEAVIATE_BATCH_SIZE]
            try:
                self.jira_collection.data.insert_many(batch)
                total_chunks += 1
            except Exception as e:
                print(f"   ❌ Error inserting issue batch to jira collection. error: {e}")

        return total_chunks

    @staticmethod
    def _prepare_issues_for_inserting_to_db(issues: List[JiraIssue]) -> List[DataObject]:
        issues_data_objects = []
        for jira_issue in issues:
            try:
                jira_issue_processor = JiraIssueLLMProcessor(
                    LLM_MODEL_NAME, VLM_MODEL_NAME, EMBEDDING_MODEL_NAME)
                issue_summary, _ = jira_issue_processor.process_issue(jira_issue)
                base_props = {
                    "issue_key": jira_issue.key,
                    "summary": issue_summary,
                    "clean_description": jira_issue.description or "",
                    "title": jira_issue.summary or "",
                    # OPTIONAL FILTER FIELDS
                    "issue_type": jira_issue.issue_type or "",
                    "priority": jira_issue.priority or "",
                    "labels": jira_issue.labels or [],
                    "squad": "",  # TODO: implement squad extraction
                    "components": [],  # TODO: implement components extraction
                    "created": jira_issue.created or None,
                    "status": jira_issue.status or None,
                }

                issues_data_objects.append(jira_issue_processor.create_data_object(base_props, issue_summary))
            except Exception as e:
                print(f"   ❌ Error processing issue {jira_issue.key}, moving to the next issue. error: {e}")
                continue

        return issues_data_objects

    def index_all(self, page_size: int = 200, jql: str = "order by updated desc"):
        """Index all Jira issues into Weaviate"""
        print(f"🚀 Starting Jira indexing...")

        total_chunks = 0
        total_issues = 0

        jira_client = JiraClient()

        next_page_token = None
        while True:
            try:
                issues, next_page_token = jira_client.fetch_issues(
                    jql=jql,
                    max_results=page_size,
                    next_page_token=next_page_token
                )

                if not issues:
                    break
            except Exception as e:
                print(f"   ❌ Error fetching jira tickets. error: {e}")
                break

            issues_data_objects = self._prepare_issues_for_inserting_to_db(issues)
            total_issues += len(issues_data_objects)

            total_chunks += self._insert_issues_data_objects_to_db(issues_data_objects)

            # Check if we've reached the end
            if not next_page_token:
                print("🏁 Reached end of issues")
                break

        # TODO: add pydentic

        print(f"\n🎉 Indexing complete!")
        print(f"📊 Total issues processed: {total_issues}")
        print(f"📊 Total chunks created: {total_chunks}")
        print(f"📊 Collection count: {len(self.jira_collection)}")


def main():
    indexer = JiraIndexer()
    try:
        jql_query = "created >= -730d order by updated desc"
        indexer.index_all(page_size=1, jql=jql_query)
    finally:
        indexer.db_client.close()


if __name__ == "__main__":
    main()
