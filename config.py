"""
Konfigurasi global untuk sistem multi-agent enterprise.
Sesuaikan path file & nama model di sini.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Path dataset (letakkan 3 file CSV kamu di folder data/) ---
CS_DATASET_PATH = os.path.join(DATA_DIR, "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
FINANCE_DATASET_PATH = os.path.join(DATA_DIR, "all-data.csv")
HR_DATASET_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")

# --- Model embedding (multibahasa, ringan, jalan di CPU) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- LLM generatif via Ollama (jalankan `ollama pull llama3.1` dulu) ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1"

# --- Model ML yang disimpan setelah training ---
FINANCE_MODEL_PATH = os.path.join(MODELS_DIR, "finance_sentiment_model.joblib")
HR_MODEL_PATH = os.path.join(MODELS_DIR, "hr_attrition_model.joblib")

TOP_K_RETRIEVAL = 3
