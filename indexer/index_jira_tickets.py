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
        description = fields.get("description") or ""
        comments = []
        if fields.get("comment") and fields["comment"].get("comments"):
            comments = [c.get("body", "") for c in fields["comment"]["comments"]]
        raw = summary + "\n\n" + description + "\n\n" + "\n\n".join(comments)
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

    @classmethod
    def index_all(cls, max_pages=5, page_size=50):
        chroma_collection = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=EMBEDDINGS,
            collection_name="jira_tickets",
        )
        start_at = 0
        for _ in range(max_pages):
            issues = JiraClient().fetch_issues(max_results=page_size, start_at=start_at)
            if not issues:
                break
            docs = []
            for issue in issues:
                docs.extend(cls.issue_to_docs(issue.to_dict()))
            # add to chroma
            chroma_collection.add_documents(docs)
            start_at += len(issues)
            if len(issues) < page_size:
                break
        # persist (Chroma persist happens automatically for local)
        chroma_collection.persist()
        print("Indexing complete.")


# tune for your Jira size
