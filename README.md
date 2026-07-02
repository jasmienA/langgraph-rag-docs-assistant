# LangGraph & LangSmith Docs Assistant (Agentic RAG + Web Fallback)

Answers questions about LangGraph and LangSmith from their official docs.
When the docs don't cover something, it automatically falls back to a live
web search instead of guessing — and tells you which source it used.

## Architecture

```
retrieve -> grade_relevance -> [route]
                                 ├── sufficient   -> generate_from_docs -> END
                                 └── insufficient -> web_search -> generate_from_web -> END
```

- **retrieve** — pulls the top 4 most similar chunks from a local Chroma
  vector store (built from LangGraph/LangSmith docs).
- **grade_relevance** — an LLM call that judges whether the retrieved
  chunks actually answer the question (not just "did search return
  something," but "is it actually useful"). This is the key agentic
  decision point.
- **generate_from_docs** — answers using only the retrieved chunks, cites
  the source URL(s).
- **web_search** — falls back to Tavily for a live web search when the
  docs don't cover the question.
- **generate_from_web** — answers from web results, clearly flagged in the
  UI as "answered from live web search" rather than the docs.


## Setup

```bash
git clone <your-repo-url>
cd rag-assistant
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in both keys:
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

Get a free Tavily key at [tavily.com](https://tavily.com) (1,000
searches/month free, no credit card).

## Build the knowledge base

```bash
python ingest.py
```

This fetches the curated LangGraph/LangSmith doc pages listed at the top of
`ingest.py`, chunks them, embeds them with OpenAI, and saves everything to
`./chroma_db`. Takes about a minute. Re-run it any time you add more URLs
to the `DOC_URLS` list.


Try it live : https://langgraph-rag-docs-assistant-kappsu9xdi3afbducugs3wd.streamlit.app/

DEMO:
Returning the data found in documents
<img width="797" height="510" alt="image" src="https://github.com/user-attachments/assets/f29839ad-7fce-4a7b-9909-feb10759b599" />

Returning data from teh web search:
<img width="827" height="327" alt="image" src="https://github.com/user-attachments/assets/ad52b342-6f32-437e-bd60-338376511465" />










## Suggested resume line

> Built an agentic RAG assistant (LangGraph + Chroma + Tavily) that grades
> its own retrieval quality and falls back to live web search when a
> private knowledge base is insufficient. [Live demo](your-streamlit-url)
