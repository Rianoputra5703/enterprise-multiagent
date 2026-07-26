"""
Helper untuk memanggil LLM via Groq API (cloud, gratis, cepat).
Digunakan sebagai pengganti Ollama saat aplikasi di-deploy ke Streamlit Cloud,
karena Ollama tidak bisa dijalankan di lingkungan cloud tersebut.

Prasyarat:
1. Daftar API key gratis di https://console.groq.com
2. Simpan API key sebagai secret di Streamlit Cloud dengan nama GROQ_API_KEY
   (Settings -> Secrets, format: GROQ_API_KEY = "isi_api_key_kamu")
3. Untuk testing lokal, buat file .streamlit/secrets.toml dengan isi yang sama.
"""
import requests
import streamlit as st

# Model Llama gratis di Groq (cek console.groq.com/docs/models untuk daftar
# model terbaru jika model ini sudah deprecated di kemudian hari)
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def call_llm(prompt: str, system: str = "") -> str:
    """Kirim prompt ke Groq API, kembalikan teks jawaban."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return (
            "[LLM tidak aktif] GROQ_API_KEY belum diset. Tambahkan API key di "
            "Streamlit Cloud Secrets atau file .streamlit/secrets.toml. "
            "Untuk sementara, ini jawaban berbasis konteks RAG saja:\n\n"
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return (
            "[LLM tidak aktif] Tidak bisa terhubung ke Groq API. Cek koneksi "
            "internet. Untuk sementara, ini jawaban berbasis konteks RAG saja:\n\n"
        )
    except requests.exceptions.ReadTimeout:
        return (
            "[LLM lambat merespons] Permintaan ke Groq API melebihi batas "
            "waktu. Coba kirim ulang pertanyaan Anda.\n\n"
        )
    except requests.exceptions.HTTPError as e:
        return (
            f"[Error API] Groq API mengembalikan error: {e}. "
            "Cek apakah API key valid dan belum melebihi limit.\n\n"
        )