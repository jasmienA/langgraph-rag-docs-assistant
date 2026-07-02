import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent import build_graph, initial_state

st.set_page_config(page_title="LangChain Docs Assistant", page_icon="🔎", layout="centered")
st.title("🔎 LangGraph & LangSmith Docs Assistant")
st.caption("Answers from the docs first, falls back to live web search when needed.")

missing = [k for k in ("OPENAI_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)]
if missing:
    st.warning(f"Missing: {', '.join(missing)}. Add these to your .env file.")

if not os.path.exists("./chroma_db"):
    if missing:
        st.error("No vector store found, and API keys are missing — can't build it. Add your keys to .env and rerun.")
        st.stop()
    with st.spinner("First run: building the knowledge base from LangGraph/LangSmith docs (~1 min)..."):
        import ingest
        ingest.build_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("badge"):
            st.caption(msg["badge"])

question = st.chat_input("Ask a question about LangGraph or LangSmith")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the docs..."):
            graph = build_graph()
            result = graph.invoke(initial_state(question))

        st.write(result["answer"])

        if result["source_type"] == "docs":
            badge = "📚 Answered from LangGraph/LangSmith docs"
        else:
            badge = "🌐 Docs didn't cover this — answered from live web search"
        st.caption(badge)

        with st.expander("🔍 What the agent retrieved"):
            if result["documents"]:
                for d in result["documents"]:
                    st.markdown(f"**{d['source']}**")
                    st.text(d["content"][:300] + "...")
            else:
                st.text("No relevant documents found in the knowledge base.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "badge": badge,
    })
