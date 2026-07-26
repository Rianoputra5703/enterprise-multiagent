"""
Customer Service Agent: menjawab pertanyaan pelanggan berbasis RAG dari dataset Bitext.
"""
from rag.vector_store import search
from agents.llm_client import call_llm
from config import TOP_K_RETRIEVAL

SYSTEM_PROMPT = (
    "Kamu adalah agent Customer Service yang ramah dan solutif. "
    "Jawab pertanyaan pelanggan dalam Bahasa Indonesia berdasarkan konteks yang diberikan. "
    "Jika konteks tidak relevan, jawab dengan pengetahuan umum yang wajar."
)


def handle(query: str) -> dict:
    results = search("customer_service_kb", query, top_k=TOP_K_RETRIEVAL)
    context = "\n\n".join([doc for doc, _ in results])

    prompt = f"Konteks referensi:\n{context}\n\nPertanyaan pelanggan: {query}\n\nJawaban:"
    answer = call_llm(prompt, system=SYSTEM_PROMPT)

    return {
        "agent": "Customer Service",
        "answer": answer,
        # Simpan teks dokumen asli (bukan skor), supaya:
        # 1. Faithfulness score di evaluator.py bisa dihitung dengan benar
        # 2. UI bisa menampilkan potongan dokumen yang dirujuk
        "sources": [doc for doc, _ in results],
        # Simpan skor similarity terpisah, kalau suatu saat dibutuhkan untuk debugging/UI
        "similarity_scores": [score for _, score in results],
    }