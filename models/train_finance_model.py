"""
Melatih model klasifikasi sentimen keuangan (Financial PhraseBank).
Jalankan: python -m models.train_finance_model
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from config import FINANCE_DATASET_PATH, FINANCE_MODEL_PATH


def main():
    df = pd.read_csv(
        FINANCE_DATASET_PATH, header=None, names=["sentiment", "sentence"], encoding="latin-1"
    ).dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        df["sentence"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("=== Laporan evaluasi model sentimen Finance ===")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, FINANCE_MODEL_PATH)
    print(f"Model disimpan di: {FINANCE_MODEL_PATH}")


if __name__ == "__main__":
    main()
