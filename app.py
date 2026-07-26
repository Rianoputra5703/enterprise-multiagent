"""
Antarmuka utama sistem multi-agent enterprise, dibangun dengan Streamlit.
Jalankan: streamlit run app.py
"""
import streamlit as st
from orchestrator import route_query
from evaluator import timed_route

st.set_page_config(page_title="Enterprise Multi-Agent Assistant", layout="wide")
st.title("Asisten Multi-Divisi Enterprise")
st.caption("Customer Service • Finance • HR — didukung RAG + Multi-Agent LLM")

with st.sidebar:
    st.subheader("Status agent")
    st.write("🟢 Customer Service Agent")
    st.write("🟢 Finance Agent")
    st.write("🟢 HR Agent")
    st.markdown("---")
    st.caption(
        "Pastikan sudah menjalankan `python -m rag.ingest` "
        "dan melatih model (`models/train_finance_model.py`, `models/train_hr_model.py`) "
        "sebelum memakai aplikasi ini."
    )

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Tanyakan sesuatu ke sistem (mis. 'bagaimana cara cancel order saya?')")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Orchestrator sedang routing ke agent yang tepat..."):
            result, scores = timed_route(route_query, query)

        st.markdown(result["answer"])

        with st.expander("Detail agent & skor evaluasi"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Agent yang menjawab", result.get("routed_intent", "-"))
            col2.metric("Latency (detik)", scores["latency_seconds"])
            col3.metric("Faithfulness score", scores["faithfulness_score"])

            st.write("**Dokumen/sumber yang dirujuk:**")
            st.json(result.get("sources", []))

    st.session_state.history.append({"role": "assistant", "content": result["answer"]})
