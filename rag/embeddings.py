"""
Modul embedding: mengubah teks jadi vektor menggunakan sentence-transformers.
"""
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

_model = None


def get_embedding_model():
    """Load model embedding sekali saja (singleton) supaya hemat memori."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Ubah list teks menjadi list vektor embedding."""
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=True).tolist()


def embed_query(query: str) -> list[float]:
    """Ubah satu query menjadi vektor embedding."""
    model = get_embedding_model()
    return model.encode([query])[0].tolist()
