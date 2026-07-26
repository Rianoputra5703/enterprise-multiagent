# Enterprise Multi-Agent LLM — Starter Project

Project ini adalah implementasi sistem multi-agent (Customer Service, Finance, HR)
menggunakan RAG + Vector DB + model fine-tuned, dengan antarmuka Streamlit.

## 1. Setup di VS Code

```bash
# buka folder ini di VS Code (File > Open Folder)
# lalu buat virtual environment
python -m venv venv

# aktifkan venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# install semua dependency
pip install -r requirements.txt
```

## 2. Siapkan dataset

Salin 3 file datasetmu ke folder `data/`:
```
data/
├── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
├── all-data.csv
└── WA_Fn-UseC_-HR-Employee-Attrition.csv
```

## 3. (Opsional) Setup LLM lokal via Ollama

Kalau mau jawaban di-generate oleh LLM (bukan cuma retrieval mentah):
1. Install Ollama dari https://ollama.com
2. Jalankan: `ollama pull llama3.1`
3. Pastikan Ollama berjalan di background (`ollama serve`)

Kalau belum mau setup Ollama, aplikasi tetap jalan — hanya jawabannya akan pakai
fallback konteks RAG mentah tanpa LLM generatif.

## 4. Ingest data ke vector database

```bash
python -m rag.ingest
```
Ini akan mengubah 3 dataset menjadi embedding dan menyimpannya di ChromaDB lokal
(folder `vectorstore/`), masing-masing di collection terpisah per divisi.

## 5. Latih model ML (Finance & HR)

```bash
python -m models.train_finance_model
python -m models.train_hr_model
```
Akan muncul laporan akurasi (precision/recall/F1) di terminal, dan model
tersimpan di folder `models/` sebagai file `.joblib`.

## 6. Jalankan aplikasi

```bash
streamlit run app.py
```
Buka browser ke `http://localhost:8501`.

## Struktur project

```
enterprise_multiagent/
├── app.py                     # Antarmuka Streamlit (entry point)
├── orchestrator.py            # Routing query ke agent yang tepat
├── evaluator.py                # Evaluasi accuracy/efficiency/hallucination
├── config.py                  # Semua konfigurasi path & model
├── agents/
│   ├── customer_service_agent.py
│   ├── finance_agent.py
│   ├── hr_agent.py
│   └── llm_client.py           # Koneksi ke LLM lokal (Ollama)
├── rag/
│   ├── embeddings.py           # Model embedding (multilingual-MiniLM)
│   ├── vector_store.py         # ChromaDB wrapper
│   └── ingest.py                # Script memasukkan dataset ke vector DB
├── models/
│   ├── train_finance_model.py  # Fine-tune sentiment classifier
│   └── train_hr_model.py       # Fine-tune attrition classifier
├── data/                        # Taruh 3 dataset CSV di sini
└── vectorstore/                 # Auto-generated oleh ChromaDB
```

## Catatan untuk laporan tugas

- **Fine-tuning**: dilakukan di `train_finance_model.py` (TF-IDF + Logistic Regression)
  dan `train_hr_model.py` (Random Forest). Untuk fine-tuning LLM generatif yang
  sesungguhnya (LoRA/QLoRA), perlu GPU dan library `peft` + `transformers` —
  bisa dijalankan terpisah di Google Colab jika laptop tidak punya GPU.
- **RAG**: ada di `rag/ingest.py` (indexing) dan `rag/vector_store.py` (retrieval).
- **Embedding**: `rag/embeddings.py`, pakai model multibahasa ringan agar jalan di CPU.
- **Vector DB**: ChromaDB, collection terpisah per divisi (`customer_service_kb`,
  `finance_kb`, `hr_kb`).
- **Evaluator**: `evaluator.py` — accuracy dari `classification_report` saat training,
  faithfulness/latency dihitung real-time di `evaluate_response()`.
