"""Streamlit UI — talks to the FastAPI service, never to the RAG core directly.
Run the API first, then:  streamlit run ui/streamlit_app.py
"""
import os

import requests
import streamlit as st

API = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="🔎")
st.title("🔎 Enterprise RAG Knowledge Assistant")
st.caption("Upload company documents and ask questions — every answer comes with its sources.")

with st.sidebar:
    st.header("Add documents")
    uploaded = st.file_uploader("PDF, TXT or Markdown", type=["pdf", "txt", "md"])
    if uploaded and st.button("Index document"):
        r = requests.post(f"{API}/ingest",
                          files={"file": (uploaded.name, uploaded.getvalue())}, timeout=60)
        st.success(r.json() if r.ok else r.text)
    try:
        st.json(requests.get(f"{API}/stats", timeout=5).json())
    except requests.RequestException:
        st.warning("API is not reachable — start it with `uvicorn app.main:app --port 8000`.")

question = st.text_input("Ask a question", placeholder="How many vacation days do employees get?")
if st.button("Ask") and question:
    r = requests.post(f"{API}/ask", json={"question": question, "top_k": 3}, timeout=30)
    if r.ok:
        data = r.json()
        st.write(data["answer"])
        st.subheader("Sources")
        for c in data["citations"]:
            st.markdown(f"**{c['source']}** · chunk {c['chunk_id']} · score {c['score']}")
            st.caption(c["excerpt"])
    else:
        st.error(r.text)
