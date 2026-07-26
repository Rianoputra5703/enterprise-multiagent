"""
Finance Agent: menjawab pertanyaan seputar sentimen keuangan.
Menggabungkan RAG (konteks historis) + model klasifikasi sentimen (fine-tuned).
"""
import os
import joblib
from rag.vector_store import search
from agents.llm_client import call_llm
from config import FINANCE_MODEL_PATH, TOP_K_RETRIEVAL

SYSTEM_PROMPT = (
    "Kamu adalah agent Finance yang menganalisis sentimen berita/laporan keuangan. "
    "Jawab dalam Bahasa Indonesia, jelaskan alasan di balik sentimen tersebut."
)

_model = None


def _load_model():
    global _model
    if _model is None and os.path.exists(FINANCE_MODEL_PATH):
        _model = joblib.load(FINANCE_MODEL_PATH)
    return _model


def predict_sentiment(text: str) -> str:
    """Prediksi langsung pakai model fine-tuned (TF-IDF + Logistic Regression)."""
    model = _load_model()
    if model is None:
        return "model belum dilatih (jalankan models/train_finance_model.py)"
    return model.predict([text])[0]


def handle(query: str) -> dict:
    results = search("finance_kb", query, top_k=TOP_K_RETRIEVAL)
    context = "\n\n".join([doc for doc, _ in results])
    predicted_sentiment = predict_sentiment(query)

    prompt = (
        f"Konteks kalimat keuangan serupa:\n{context}\n\n"
        f"Prediksi model sentimen untuk teks ini: {predicted_sentiment}\n\n"
        f"Pertanyaan/teks pengguna: {query}\n\nJawaban:"
    )
    answer = call_llm(prompt, system=SYSTEM_PROMPT)

    return {
        "agent": "Finance",
        "answer": answer,
        "predicted_sentiment": predicted_sentiment,
        # Simpan teks dokumen asli (bukan skor), supaya faithfulness score
        # bisa dihitung dengan benar dan UI bisa menampilkan sumber yang dirujuk.
        "sources": [doc for doc, _ in results],
        "similarity_scores": [score for _, score in results],
    }