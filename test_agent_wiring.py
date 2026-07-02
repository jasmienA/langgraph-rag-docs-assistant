"""
Verifies the RAG agent graph wiring works end-to-end WITHOUT calling OpenAI
or Tavily. Tests BOTH branches: docs-sufficient and web-fallback.
"""

from agent import build_graph, initial_state


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeVectorStoreGood:
    """Returns a chunk that actually answers the question."""

    def similarity_search(self, query, k=4):
        class D:
            page_content = "LangGraph's StateGraph lets you define nodes and conditional edges for agent workflows."
            metadata = {"source": "https://docs.langchain.com/oss/python/langgraph/overview"}
        return [D()]


class FakeVectorStoreEmpty:
    """Simulates a knowledge base with nothing relevant."""

    def similarity_search(self, query, k=4):
        return []


class FakeLLM:
    def __init__(self, grade_answer="YES"):
        self.grade_answer = grade_answer

    def invoke(self, prompt):
        if "Reply with exactly one word" in prompt:
            return FakeResponse(self.grade_answer)
        if "Web results" in prompt:
            return FakeResponse("Per live web search: the answer is X. Source: https://example.com")
        return FakeResponse("StateGraph organizes nodes and edges for agent workflows. Source: docs.langchain.com")


class FakeTavily:
    def search(self, query, max_results=4):
        return {"results": [{"url": "https://example.com", "content": "Some fresh web content."}]}


def test_docs_sufficient_branch():
    graph = build_graph(
        vectorstore=FakeVectorStoreGood(),
        llm=FakeLLM(grade_answer="YES"),
        tavily_client=FakeTavily(),
    )
    result = graph.invoke(initial_state("What is a StateGraph?"))
    assert result["source_type"] == "docs", f"expected docs branch, got {result['source_type']}"
    assert result["web_results"] is None, "web search should not have run"
    print("docs-sufficient branch: OK ->", result["answer"][:60])


def test_web_fallback_branch():
    graph = build_graph(
        vectorstore=FakeVectorStoreEmpty(),
        llm=FakeLLM(grade_answer="NO"),
        tavily_client=FakeTavily(),
    )
    result = graph.invoke(initial_state("What's the weather in Tokyo right now?"))
    assert result["source_type"] == "web", f"expected web branch, got {result['source_type']}"
    assert result["web_results"] is not None, "web search should have run"
    print("web-fallback branch:   OK ->", result["answer"][:60])


if __name__ == "__main__":
    test_docs_sufficient_branch()
    test_web_fallback_branch()
    print("\nAll checks passed: both routing branches work correctly.")
