"""
Modul vector database menggunakan Chroma (open source, jalan lokal).
Setiap divisi punya collection terpisah -> isolasi data antar-agent.
"""
import chromadb
from config import VECTORSTORE_DIR
from rag.embeddings import embed_texts, embed_query

_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)


def get_collection(name: str):
    """Ambil (atau buat baru) collection sesuai nama divisi."""
    return _client.get_or_create_collection(name=name)


def add_documents(collection_name: str, documents: list[str], metadatas: list[dict], ids: list[str]):
    """Simpan dokumen (sudah di-embed otomatis) ke collection tertentu."""
    collection = get_collection(collection_name)
    embeddings = embed_texts(documents)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def search(collection_name: str, query: str, top_k: int = 3):
    """Cari dokumen paling relevan dari collection tertentu berdasarkan query."""
    collection = get_collection(collection_name)
    query_embedding = embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return list(zip(docs, metas))
