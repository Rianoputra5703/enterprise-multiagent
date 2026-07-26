# --- Patch wajib untuk kompatibilitas ChromaDB di Streamlit Community Cloud ---
# Streamlit Cloud pakai SQLite versi lama, sedangkan ChromaDB butuh SQLite >= 3.35.
# Baris ini HARUS ada di paling atas, sebelum import lain apa pun.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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


@st.cache_resource(show_spinner="Menyiapkan basis data pengetahuan (hanya sekali saat aplikasi start)...")
def ensure_data_ingested():
    """
    Cek apakah vector store sudah berisi data. Kalau kosong (misalnya karena
    file binary hasil ingest lokal tidak kompatibel dengan versi chromadb di
    lingkungan cloud), jalankan ulang proses ingest secara otomatis.
    Di-cache dengan st.cache_resource supaya hanya berjalan SEKALI selama
    aplikasi ini hidup, bukan setiap kali ada pertanyaan baru.
    """
    from rag.vector_store import get_collection
    from rag.ingest import ingest_customer_service, ingest_finance, ingest_hr

    checks = {
        "customer_service_kb": ingest_customer_service,
        "finance_kb": ingest_finance,
        "hr_kb": ingest_hr,
    }

    for collection_name, ingest_fn in checks.items():
        try:
            collection = get_collection(collection_name)
            count = collection.count()
        except Exception:
            count = 0

        if count == 0:
            ingest_fn()

    return True


# Jalankan pengecekan/ingest otomatis sebelum UI utama dimuat
ensure_data_ingested()

with st.sidebar:
    st.subheader("Status agent")
    st.write("🟢 Customer Service Agent")
    st.write("🟢 Finance Agent")
    st.write("🟢 HR Agent")
    st.markdown("---")
    st.caption(
        "Basis data pengetahuan (RAG) disiapkan otomatis saat aplikasi "
        "pertama kali dijalankan di server ini."
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