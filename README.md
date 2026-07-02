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

This is a different LangGraph *shape* than the Data Analyst Agent project —
conditional routing based on a relevance judgment, rather than a
generate/execute/retry loop. Worth mentioning together on a resume to show
you understand more than one agent pattern.

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

## Verify the graph wiring (no API calls, free)

```bash
python test_agent_wiring.py
```

This tests **both** branches — docs-sufficient and web-fallback — using
fake components, so you can confirm the routing logic works before
spending real API calls on it.

## Run it

```bash
streamlit run app.py
```

Try:
- *"What is a StateGraph in LangGraph?"* → should answer from docs
- *"What's the latest LangGraph release version?"* → likely falls back to
  web search, since doc snapshots go stale but the web doesn't
- Open **🔍 What the agent retrieved** to see the actual chunks it pulled

## Deploying to Streamlit Community Cloud

1. Push to GitHub (`.env` and `chroma_db/` are gitignored — they won't be
   committed, which is intentional: don't commit API keys or a large binary
   vector store).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `app.py`.
3. In **Secrets**, add both:
   ```
   OPENAI_API_KEY = "sk-..."
   TAVILY_API_KEY = "tvly-..."
   ```
4. Deploy. On first load, the app automatically builds the vector store
   itself (see the `if not os.path.exists("./chroma_db")` block in
   `app.py`) — no manual `ingest.py` step needed on the server.

## Known limitations (good interview talking points)

- **Chroma storage isn't persistent across Streamlit Cloud redeploys** —
  each redeploy rebuilds the index from scratch on first load. Fine for a
  demo; a production version would use a hosted vector DB or persistent
  volume.
- **The relevance grader is a single LLM call with no confidence score** —
  it's a binary yes/no, not a nuanced ranking. A more advanced version
  could re-rank retrieved chunks or ask the grader to cite *which* chunk
  helped.
- **No conversation memory** — every question is independent; a follow-up
  like "can you give an example of that" won't know what "that" refers to.
- **The curated URL list is small on purpose** (~12 pages) to keep the demo
  fast and cheap. Expanding `DOC_URLS` in `ingest.py` deepens the knowledge
  base at the cost of a longer ingest run.

## Suggested resume line

> Built an agentic RAG assistant (LangGraph + Chroma + Tavily) that grades
> its own retrieval quality and falls back to live web search when a
> private knowledge base is insufficient. [Live demo](your-streamlit-url)
