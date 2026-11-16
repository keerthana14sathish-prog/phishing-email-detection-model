import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src_features import EmailFeatureExtractor

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "emails.csv"
MODEL_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "phishing_email_model.joblib"


def build_pipeline():
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            max_features=12000, sublinear_tf=True, min_df=1
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), max_features=18000,
            sublinear_tf=True, min_df=1
        )),
        ("email_metadata", Pipeline([
            ("extract", EmailFeatureExtractor()),
            ("vectorize", DictVectorizer(sparse=True))
        ]))
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ))
    ])


def main():
    parser = argparse.ArgumentParser(description="Train a phishing email classifier.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV with email_text,label columns")
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required = {"email_text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain columns named email_text and label.")
    df = df.dropna(subset=["email_text", "label"]).copy()
    df["email_text"] = df["email_text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.title()
    df = df[(df["email_text"] != "") & df["label"].isin(["Phishing", "Safe"])]
    if df["label"].nunique() != 2 or len(df) < 10:
        raise ValueError("Provide at least 10 labelled rows and both classes: Phishing and Safe.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["email_text"], df["label"], test_size=0.20,
        random_state=42, stratify=df["label"]
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    labels = ["Phishing", "Safe"]
    cm = confusion_matrix(y_test, predictions, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Phishing Email Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    metrics = {
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_phishing": float(precision_score(y_test, predictions, pos_label="Phishing", zero_division=0)),
        "recall_phishing": float(recall_score(y_test, predictions, pos_label="Phishing", zero_division=0)),
        "f1_phishing": float(f1_score(y_test, predictions, pos_label="Phishing", zero_division=0)),
        "classification_report": classification_report(y_test, predictions, labels=labels, zero_division=0, output_dict=True),
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
        "dataset_note": "The bundled dataset is synthetic demonstration data; metrics are not a real-world performance estimate."
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\\nMODEL EVALUATION\\n----------------")
    print(f"Dataset rows : {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"Precision    : {metrics['precision_phishing']:.4f}")
    print(f"Recall       : {metrics['recall_phishing']:.4f}")
    print(f"F1-score     : {metrics['f1_phishing']:.4f}")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm)
    print(f"\\nSaved model: {MODEL_PATH}")
    print(f"Saved confusion matrix: {MODEL_DIR / 'confusion_matrix.png'}")
    print(f"Saved metrics: {MODEL_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
