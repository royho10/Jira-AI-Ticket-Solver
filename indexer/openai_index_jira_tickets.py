import time
from threading import local, Lock

import requests
import weaviate
import weaviate.classes as wvc
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import AzureOpenAIEmbeddings
from pathlib import Path
from typing import List, Set
from weaviate.classes.config import Property, DataType
from weaviate.classes.data import DataObject

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_LLM_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    OPENAI_JIRA_COLLECTION_NAME,
    MAX_EMBEDDINGS_INPUT_CHARS,
)
from utils.jira_client import JiraClient, JiraIssue
from utils.openai_jira_ticket_processing import OpenAIJiraIssueLLMProcessor

# Indexer-specific constants
WEAVIATE_BATCH_SIZE = 100
MAX_PROCESS_TICKET_WORKERS = 5


class OpenAIJiraIndexer:
    """
    Batch indexer for Jira tickets using OpenAI embeddings.
    Uses a separate Weaviate collection (JiraCollectionOpenAI) due to different embedding dimensions.
    """

    def __init__(self):
        self.db_client = weaviate.connect_to_local()
        self.jira_client = JiraClient()
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            openai_api_version=AZURE_OPENAI_API_VERSION,
        )

        # Setup collection
        self._setup_collection()

    def close(self) -> None:
        if getattr(self, "db_client", None):
            self.db_client.close()

    def _setup_collection(self):
        """Setup Weaviate collection schema if not exists"""
        if self.db_client.collections.exists(OPENAI_JIRA_COLLECTION_NAME):
            self.jira_collection = self.db_client.collections.get(OPENAI_JIRA_COLLECTION_NAME)
            return
        else:
            # Create new collection
            self.jira_collection = self.db_client.collections.create(
                name=OPENAI_JIRA_COLLECTION_NAME,
                # Configure for similarity search
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
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
                ]
            )

    def _get_existing_issue_keys(self) -> Set[str]:
        """Fetch all existing issue keys from the collection to avoid duplicates."""
        existing_keys = set()
        try:
            for item in self.jira_collection.iterator():
                issue_key = item.properties.get("issue_key")
                if issue_key:
                    existing_keys.add(issue_key)
            print(f"📋 Found {len(existing_keys)} existing tickets in database")
        except Exception as e:
            print(f"   ⚠️ Could not fetch existing keys: {e}")
        return existing_keys

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
        thread_local = local()

        total = len(issues)
        counter_lock = Lock()
        counter = {"n": 0}  # mutable holder

        def next_ticket_number() -> int:
            with counter_lock:
                counter["n"] += 1
                return counter["n"]

        def get_processor() -> OpenAIJiraIssueLLMProcessor:
            # Reuse one processor per worker thread to avoid repeated socket/client creation
            if not hasattr(thread_local, "processor"):
                thread_local.processor = OpenAIJiraIssueLLMProcessor(
                    AZURE_OPENAI_LLM_DEPLOYMENT,
                    AZURE_OPENAI_LLM_DEPLOYMENT,
                    AZURE_OPENAI_EMBEDDING_DEPLOYMENT
                )
            return thread_local.processor

        def process_single_issue(jira_issue):
            ticket_no = next_ticket_number()
            try:
                print(f"[{ticket_no}/{total}] Processing {jira_issue.key} ...")

                jira_issue_processor = get_processor()
                issue_summary, _ = jira_issue_processor.process_issue(jira_issue)
                base_props = {
                    "issue_key": jira_issue.key,
                    "summary": issue_summary,
                    "clean_description": jira_issue.description or "",
                    "title": jira_issue.summary or "",
                    "issue_type": jira_issue.issue_type or "",
                    "priority": jira_issue.priority or "",
                    "labels": jira_issue.labels or [],
                    "squad": "",
                    "components": [],
                    "created": jira_issue.created or None,
                    "status": jira_issue.status or None,
                }

                print(f"[{ticket_no}/{total}] Done {jira_issue.key}")

                # Embed both summary and description for better similarity matching
                content_to_embed = (issue_summary + "\n\n" + (jira_issue.description or ""))[:MAX_EMBEDDINGS_INPUT_CHARS]
                return jira_issue_processor.create_data_object(base_props, content_to_embed)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                print(f"OpenAI API unavailable, moving to the next issue. error: {e}")
            except Exception as e:
                print(f"   ❌ Error processing issue {jira_issue.key}, moving to the next issue. error: {e}")
                return None

        issues_data_objects = []
        with ThreadPoolExecutor(max_workers=MAX_PROCESS_TICKET_WORKERS) as executor:  # Adjust max_workers as needed
            futures = [executor.submit(process_single_issue, issue) for issue in issues]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    issues_data_objects.append(result)
        return issues_data_objects

    def index_all(self, page_size: int = 200, jql: str = "order by updated desc"):
        """Index all Jira issues into Weaviate"""
        print(f"🚀 Starting Jira indexing with OpenAI embeddings...")

        # Fetch existing keys to avoid duplicates
        existing_keys = self._get_existing_issue_keys()

        total_chunks = 0
        total_issues = 0
        skipped_issues = 0

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

            # Filter out already indexed issues
            new_issues = [issue for issue in issues if issue.key not in existing_keys]
            skipped_in_batch = len(issues) - len(new_issues)
            skipped_issues += skipped_in_batch

            if skipped_in_batch > 0:
                print(f"   ⏭️ Skipping {skipped_in_batch} already indexed tickets")

            if not new_issues:
                if not next_page_token:
                    print("🏁 Reached end of issues")
                    break
                continue

            issues_data_objects = self._prepare_issues_for_inserting_to_db(new_issues)
            total_issues += len(issues_data_objects)

            total_chunks += self._insert_issues_data_objects_to_db(issues_data_objects)

            # Add newly indexed keys to existing_keys set to handle duplicates within the same run
            for issue in new_issues:
                existing_keys.add(issue.key)

            # Check if we've reached the end
            if not next_page_token:
                print("🏁 Reached end of issues")
                break

        print(f"\n🎉 Indexing complete!")
        print(f"📊 Total issues processed: {total_issues}")
        print(f"📊 Total issues skipped (already indexed): {skipped_issues}")
        print(f"📊 Total chunks created: {total_chunks}")
        print(f"📊 Collection count: {len(self.jira_collection)}")


def main():
    indexer = OpenAIJiraIndexer()
    start = time.perf_counter()

    try:
        jql_query = "project IN ('GC') AND issuetype = Bug AND created >= -1825d AND 'Is Field issue' = No AND component = 'Aggregator / Collector [Collection]' AND status In (CLOSED, Done)"
        indexer.index_all(page_size=20, jql=jql_query)
    finally:
        elapsed = time.perf_counter() - start
        print(f"\n⏱️ Indexing took: {elapsed:.2f}s")
        indexer.db_client.close()


if __name__ == "__main__":
    main()
