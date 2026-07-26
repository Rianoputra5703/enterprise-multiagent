"""
Orchestrator Agent: menentukan agent divisi mana yang menangani query pengguna.
Pendekatan sederhana: keyword-based routing (bisa diganti classifier/LLM zero-shot).
"""
from agents import customer_service_agent, finance_agent, hr_agent


# Kata kunci khusus untuk ANALISIS SENTIMEN berita/laporan keuangan perusahaan
# (dataset finance_kb berisi berita finansial, bukan data transaksi personal pengguna)
FINANCE_KEYWORDS = [
    "saham", "sentimen", "laba", "profit", "pendapatan perusahaan", "revenue",
    "laporan keuangan", "emiten", "obligasi", "kredit korporasi",
    "analisis keuangan", "berita ekonomi", "pasar modal",
]

HR_KEYWORDS = [
    "karyawan", "resign", "attrition", "hr", "sdm", "kepuasan kerja",
    "gaji", "promosi", "lembur", "cuti", "izin", "kontrak kerja",
    "rekrutmen", "pelatihan", "tunjangan", "bpjs", "absensi",
    "onboarding", "penilaian kinerja", "pensiun",
]

# Termasuk hal-hal seputar transaksi/pembayaran PERSONAL pengguna
# (dataset customer_service_kb -- Bitext -- lebih relevan untuk ini)
CS_KEYWORDS = [
    "pesanan", "order", "refund", "retur", "pengiriman", "akun",
    "pembatalan", "keluhan pelanggan", "cancel", "produk", "barang",
    "komplain", "garansi", "pelacakan", "resi", "ganti alamat",
    "transaksi", "pembayaran", "tagihan", "billing", "saldo",
    "rekening", "cicilan", "riwayat pembelian",
]


def classify_intent(query: str) -> str:
    """
    Klasifikasi sederhana berbasis keyword.
    Ganti dengan LLM/classifier untuk hasil lebih akurat di masa depan.
    """
    q = query.lower()

    if any(keyword in q for keyword in FINANCE_KEYWORDS):
        return "finance"
    if any(keyword in q for keyword in HR_KEYWORDS):
        return "hr"
    if any(keyword in q for keyword in CS_KEYWORDS):
        return "customer_service"

    return "customer_service"  # default fallback


def route_query(query: str) -> dict:
    """Fungsi utama: terima query pengguna, kembalikan hasil dari agent yang relevan."""
    intent = classify_intent(query)

    if intent == "finance":
        result = finance_agent.handle(query)
    elif intent == "hr":
        result = hr_agent.handle(query)
    else:
        result = customer_service_agent.handle(query)

    result["routed_intent"] = intent
    return result