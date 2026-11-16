import argparse
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "phishing_email_model.joblib"


def predict_email(email_text):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]
    classes = list(model.named_steps["classifier"].classes_)
    confidence = float(probabilities[classes.index(prediction)])
    return prediction, confidence


def main():
    parser = argparse.ArgumentParser(description="Classify one email as Phishing or Safe.")
    parser.add_argument("--text", help="Email text to classify")
    args = parser.parse_args()
    email_text = args.text if args.text else input("Paste the email text: ").strip()
    if not email_text:
        print("Please provide email text.")
        return
    label, confidence = predict_email(email_text)
    print("\\nPREDICTION\\n----------")
    print(f"Classification: {label}")
    print(f"Model confidence: {confidence:.2%}")
    print("\\nNote: This is a machine-learning estimate, not a guarantee. Do not click suspicious links.")


if __name__ == "__main__":
    main()
