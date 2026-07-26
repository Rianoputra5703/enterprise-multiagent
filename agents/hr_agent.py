"""
HR Agent: menjawab pertanyaan seputar risiko attrition karyawan.
Menggabungkan RAG (profil karyawan mirip) + model prediksi attrition (fine-tuned).
"""
import os
import joblib
import pandas as pd
from rag.vector_store import search
from agents.llm_client import call_llm
from config import HR_MODEL_PATH, TOP_K_RETRIEVAL

SYSTEM_PROMPT = (
    "Kamu adalah agent HR yang membantu menganalisis risiko attrition (resign) karyawan. "
    "Jawab dalam Bahasa Indonesia, berikan insight yang actionable bagi tim HR."
)

_bundle = None


def _load_model():
    global _bundle
    if _bundle is None and os.path.exists(HR_MODEL_PATH):
        _bundle = joblib.load(HR_MODEL_PATH)
    return _bundle


def predict_attrition(employee: dict) -> str:
    """
    Prediksi attrition dari data satu karyawan (dict).
    employee harus berisi field yang sama dengan kolom dataset (kecuali Attrition).
    """
    bundle = _load_model()
    if bundle is None:
        return "model belum dilatih (jalankan models/train_hr_model.py)"

    model, encoders, feature_order = bundle["model"], bundle["encoders"], bundle["feature_order"]
    row = pd.DataFrame([employee])[feature_order]
    for col, le in encoders.items():
        if col in row.columns:
            row[col] = le.transform(row[col].astype(str))

    pred = model.predict(row)[0]
    return "Yes" if pred == 1 else "No"


def handle(query: str) -> dict:
    """Mode tanya-jawab umum berbasis RAG (tanpa input structured employee)."""
    results = search("hr_kb", query, top_k=TOP_K_RETRIEVAL)
    context = "\n\n".join([doc for doc, _ in results])

    prompt = f"Konteks profil karyawan serupa:\n{context}\n\nPertanyaan: {query}\n\nJawaban:"
    answer = call_llm(prompt, system=SYSTEM_PROMPT)

    return {
        "agent": "HR",
        "answer": answer,
        # Simpan teks dokumen asli (bukan metadata dict), supaya faithfulness
        # score bisa dihitung dengan benar dan UI bisa menampilkan sumber yang dirujuk.
        "sources": [doc for doc, _ in results],
        "similarity_scores": [score for _, score in results],
    }