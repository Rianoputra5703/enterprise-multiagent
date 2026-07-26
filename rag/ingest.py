"""
Jalankan sekali di awal untuk memasukkan dataset ke vector DB:
    python -m rag.ingest
"""
import pandas as pd
from config import CS_DATASET_PATH, FINANCE_DATASET_PATH, HR_DATASET_PATH
from rag.vector_store import add_documents

BATCH_SIZE = 500  # supaya tidak sekali embed puluhan ribu baris (lambat)


def ingest_customer_service(limit: int = 3000):
    """Masukkan pasangan instruction-response Bitext ke collection 'customer_service_kb'."""
    df = pd.read_csv(CS_DATASET_PATH).dropna(subset=["instruction", "response"]).head(limit)
    documents = [f"Q: {row.instruction}\nA: {row.response}" for row in df.itertuples()]
    metadatas = [{"intent": row.intent, "category": row.category} for row in df.itertuples()]
    ids = [f"cs-{i}" for i in range(len(documents))]

    for i in range(0, len(documents), BATCH_SIZE):
        add_documents(
            "customer_service_kb",
            documents[i:i + BATCH_SIZE],
            metadatas[i:i + BATCH_SIZE],
            ids[i:i + BATCH_SIZE],
        )
    print(f"[CS] {len(documents)} dokumen dimasukkan ke customer_service_kb")


def ingest_finance(limit: int = 3000):
    """Masukkan kalimat Financial PhraseBank ke collection 'finance_kb'."""
    df = pd.read_csv(
        FINANCE_DATASET_PATH, header=None, names=["sentiment", "sentence"], encoding="latin-1"
    ).dropna().head(limit)
    documents = df["sentence"].tolist()
    metadatas = [{"sentiment": s} for s in df["sentiment"].tolist()]
    ids = [f"fin-{i}" for i in range(len(documents))]

    for i in range(0, len(documents), BATCH_SIZE):
        add_documents(
            "finance_kb",
            documents[i:i + BATCH_SIZE],
            metadatas[i:i + BATCH_SIZE],
            ids[i:i + BATCH_SIZE],
        )
    print(f"[Finance] {len(documents)} dokumen dimasukkan ke finance_kb")


def ingest_hr():
    """Masukkan ringkasan tiap karyawan (dalam bentuk teks) ke collection 'hr_kb'."""
    df = pd.read_csv(HR_DATASET_PATH)
    documents = []
    metadatas = []
    for row in df.itertuples():
        text = (
            f"Karyawan usia {row.Age} tahun, departemen {row.Department}, "
            f"jabatan {row.JobRole}, kepuasan kerja {row.JobSatisfaction}/4, "
            f"lembur: {row.OverTime}, lama kerja {row.YearsAtCompany} tahun, "
            f"status attrition: {row.Attrition}."
        )
        documents.append(text)
        metadatas.append({"attrition": row.Attrition, "department": row.Department})
    ids = [f"hr-{i}" for i in range(len(documents))]

    for i in range(0, len(documents), BATCH_SIZE):
        add_documents(
            "hr_kb",
            documents[i:i + BATCH_SIZE],
            metadatas[i:i + BATCH_SIZE],
            ids[i:i + BATCH_SIZE],
        )
    print(f"[HR] {len(documents)} dokumen dimasukkan ke hr_kb")


if __name__ == "__main__":
    ingest_customer_service()
    ingest_finance()
    ingest_hr()
    print("Selesai ingest semua dataset ke vector DB.")
