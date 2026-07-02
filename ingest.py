"""
Ingests LangGraph + LangSmith documentation into a local Chroma vector store.

Run this once (and again any time you want to refresh the knowledge base):
    python ingest.py

It fetches each URL below, strips HTML down to readable text, splits it into
chunks, embeds them with OpenAI, and persists everything to ./chroma_db.
"""

import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PERSIST_DIR = "./chroma_db"

# Curated set of LangGraph + LangSmith doc pages. Add more URLs here any time
# to expand the knowledge base — the ingest script re-splits and re-embeds
# everything on each run.
DOC_URLS = [
    "https://docs.langchain.com/oss/python/langgraph/overview",
    "https://docs.langchain.com/oss/python/langgraph/persistence",
    "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "https://docs.langchain.com/oss/python/concepts/memory",
    "https://docs.langchain.com/oss/python/langchain/overview",
    "https://docs.langchain.com/oss/python/langchain/models",
    "https://docs.langchain.com/oss/python/langchain/tools",
    "https://docs.langchain.com/langsmith/home",
    "https://docs.langchain.com/langsmith/observability",
    "https://docs.langchain.com/langsmith/trace-with-langchain",
    "https://docs.langchain.com/langsmith/deployment",
    "https://docs.langchain.com/langsmith/engine",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAGIngestBot/1.0)"}


def fetch_page_text(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ! skipped {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def build_index():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_chunks = []

    print(f"Fetching {len(DOC_URLS)} pages...")
    for url in DOC_URLS:
        text = fetch_page_text(url)
        if not text:
            continue
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk, metadata={"source": url}))
        print(f"  + {url} -> {len(chunks)} chunks")

    if not all_chunks:
        raise RuntimeError("No documents were successfully fetched. Check your network connection.")

    print(f"\nEmbedding {len(all_chunks)} chunks into Chroma at {PERSIST_DIR}...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(all_chunks, embeddings, persist_directory=PERSIST_DIR)
    print("Done. Vector store is ready.")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your .env file before running ingest.py")
    build_index()
