"""
Melatih model prediksi attrition karyawan (IBM HR Analytics).
Jalankan: python -m models.train_hr_model
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from config import HR_DATASET_PATH, HR_MODEL_PATH


def main():
    df = pd.read_csv(HR_DATASET_PATH)

    # Drop kolom yang tidak informatif (nilainya konstan/ID)
    drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df["Attrition"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Attrition"])

    # Encode kolom kategorikal (mis. Department, JobRole, OverTime)
    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=== Laporan evaluasi model attrition HR ===")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

    joblib.dump({"model": model, "encoders": encoders, "feature_order": X.columns.tolist()}, HR_MODEL_PATH)
    print(f"Model disimpan di: {HR_MODEL_PATH}")


if __name__ == "__main__":
    main()
