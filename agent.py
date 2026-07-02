"""
Agentic RAG Assistant — LangGraph state machine.

Flow:
    retrieve -> grade_relevance -> [route]
                                     ├── "sufficient"   -> generate_from_docs -> END
                                     └── "insufficient" -> web_search -> generate_from_web -> END

Retrieves from a local Chroma vector store first. An LLM call judges whether
the retrieved chunks actually answer the question. If not, it falls back to
a live Tavily web search instead of forcing an answer from irrelevant docs.
"""

import os
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from tavily import TavilyClient

load_dotenv()

MODEL_NAME = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
PERSIST_DIR = "./chroma_db"
TOP_K = 4


class AgentState(TypedDict):
    question: str
    documents: List[dict]       # [{"content": str, "source": str}, ...]
    relevant: Optional[bool]
    web_results: Optional[str]
    answer: str
    source_type: str            # "docs" or "web"


def _get_llm():
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


def _get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def make_retrieve(vectorstore=None):
    def retrieve(state: AgentState) -> AgentState:
        store = vectorstore or _get_vectorstore()
        results = store.similarity_search(state["question"], k=TOP_K)
        state["documents"] = [
            {"content": doc.page_content, "source": doc.metadata.get("source", "unknown")}
            for doc in results
        ]
        return state

    return retrieve


def make_grade_relevance(llm=None):
    llm = llm or _get_llm()

    def grade_relevance(state: AgentState) -> AgentState:
        if not state["documents"]:
            state["relevant"] = False
            return state

        context = "\n\n".join(d["content"][:400] for d in state["documents"])
        prompt = (
            f"Question: {state['question']}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Does this context contain enough information to accurately answer "
            "the question? Reply with exactly one word: YES or NO."
        )
        response = llm.invoke(prompt)
        state["relevant"] = "yes" in response.content.strip().lower()
        return state

    return grade_relevance


def route(state: AgentState) -> str:
    return "sufficient" if state["relevant"] else "insufficient"


def make_generate_from_docs(llm=None):
    llm = llm or _get_llm()

    def generate_from_docs(state: AgentState) -> AgentState:
        context = "\n\n".join(
            f"[Source: {d['source']}]\n{d['content']}" for d in state["documents"]
        )
        prompt = (
            f"Answer the question using ONLY the context below. Cite the source "
            f"URL(s) you used at the end.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}"
        )
        response = llm.invoke(prompt)
        state["answer"] = response.content
        state["source_type"] = "docs"
        return state

    return generate_from_docs


def make_web_search(tavily_client=None):
    def web_search(state: AgentState) -> AgentState:
        client = tavily_client or TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        results = client.search(state["question"], max_results=4)
        formatted = "\n\n".join(
            f"[{r['url']}]\n{r['content']}" for r in results.get("results", [])
        )
        state["web_results"] = formatted
        return state

    return web_search


def make_generate_from_web(llm=None):
    llm = llm or _get_llm()

    def generate_from_web(state: AgentState) -> AgentState:
        prompt = (
            f"Answer the question using the live web search results below. "
            f"Cite the source URL(s) you used at the end.\n\n"
            f"Web results:\n{state['web_results']}\n\n"
            f"Question: {state['question']}"
        )
        response = llm.invoke(prompt)
        state["answer"] = response.content
        state["source_type"] = "web"
        return state

    return generate_from_web


def build_graph(vectorstore=None, llm=None, tavily_client=None):
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", make_retrieve(vectorstore))
    graph.add_node("grade_relevance", make_grade_relevance(llm))
    graph.add_node("generate_from_docs", make_generate_from_docs(llm))
    graph.add_node("web_search", make_web_search(tavily_client))
    graph.add_node("generate_from_web", make_generate_from_web(llm))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_conditional_edges(
        "grade_relevance",
        route,
        {"sufficient": "generate_from_docs", "insufficient": "web_search"},
    )
    graph.add_edge("generate_from_docs", END)
    graph.add_edge("web_search", "generate_from_web")
    graph.add_edge("generate_from_web", END)

    return graph.compile()


def initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "documents": [],
        "relevant": None,
        "web_results": None,
        "answer": "",
        "source_type": "",
    }
