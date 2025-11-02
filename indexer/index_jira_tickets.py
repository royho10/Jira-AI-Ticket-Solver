import os

from dotenv import load_dotenv
from typing import Dict, List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from utils.jira_client import JiraClient

load_dotenv()
CHROMA_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")

# LangChain/OpenAI embeddings wrapper
EMBEDDINGS = OpenAIEmbeddings(
    model="text-embedding-3-large", openai_api_key=os.environ["OPENAI_API_KEY"]
)

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


class JiraIndexer:
    @staticmethod
    def issue_to_docs(issue: Dict) -> List[Document]:
        fields = issue.get("fields", {})
        summary = fields.get("summary", "") or ""

        # Handle description field - can be string or dict
        description = fields.get("description") or ""
        if isinstance(description, dict):
            # Extract text from Atlassian Document Format
            description = JiraIndexer._extract_text_from_adf(description)

        comments = []
        if fields.get("comment") and fields["comment"].get("comments"):
            for comment in fields["comment"]["comments"]:
                comment_body = comment.get("body", "")
                if isinstance(comment_body, dict):
                    # Extract text from ADF format
                    comment_text = JiraIndexer._extract_text_from_adf(comment_body)
                else:
                    comment_text = str(comment_body)
                comments.append(comment_text)

        raw = summary + "\n\n" + str(description) + "\n\n" + "\n\n".join(comments)
        chunks = TEXT_SPLITTER.split_text(raw)
        docs = []
        for i, chunk in enumerate(chunks):
            meta = {
                "ticket_key": issue.get("key"),
                "summary": summary,
                "status": fields.get("status", {}).get("name"),
                "resolution": fields.get("resolution", {}).get("name")
                if fields.get("resolution")
                else None,
                "chunk_index": i,
            }
            docs.append(Document(page_content=chunk, metadata=meta))
        return docs

    @staticmethod
    def _extract_text_from_adf(adf_content):
        """Extract plain text from Atlassian Document Format"""
        if not isinstance(adf_content, dict):
            return str(adf_content)

        text_parts = []
        content = adf_content.get("content", [])

        for item in content:
            if item.get("type") == "paragraph":
                paragraph_content = item.get("content", [])
                for text_item in paragraph_content:
                    if text_item.get("type") == "text":
                        text_parts.append(text_item.get("text", ""))

        return " ".join(text_parts)

    @classmethod
    def index_all(cls, max_pages=5, page_size=50):
        print(f"🗄️ Initializing Chroma database at: {CHROMA_DIR}")
        chroma_collection = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=EMBEDDINGS,
            collection_name="jira_tickets",
        )

        print(f"📊 Current collection count: {chroma_collection._collection.count()}")

        start_at = 0
        total_docs_added = 0

        for page in range(max_pages):
            print(f"\n📄 Processing page {page + 1}/{max_pages}...")
            issues = JiraClient().fetch_issues(max_results=page_size, start_at=start_at)
            if not issues:
                print("❌ No more issues found")
                break

            print(f"✅ Found {len(issues)} issues")
            docs = []
            for issue in issues:
                issue_docs = cls.issue_to_docs(issue.to_dict())
                docs.extend(issue_docs)
                print(f"   📝 {issue.key}: {len(issue_docs)} document chunks")

            print(f"💾 Adding {len(docs)} documents to Chroma...")
            # add to chroma
            chroma_collection.add_documents(docs)
            total_docs_added += len(docs)

            start_at += len(issues)
            if len(issues) < page_size:
                print("🏁 Reached end of issues")
                break

        print(f"\n🎉 Indexing complete!")
        print(f"📊 Total documents added: {total_docs_added}")
        print(f"📊 Final collection count: {chroma_collection._collection.count()}")

        # persist (Chroma persist happens automatically for local)
        chroma_collection.persist()
        print("💾 Database persisted successfully")


if __name__ == "__main__":
    indexer = JiraIndexer()
    indexer.index_all()
