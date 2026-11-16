# Phishing Email Detection Model

A Python machine-learning project that classifies email text as **Phishing** or **Safe** using TF-IDF text features, email indicators, and Logistic Regression.

## Features
- Labelled phishing and legitimate email dataset.
- Word-level and character-level TF-IDF features.
- Email indicators: URL count, suspicious keyword count, uppercase ratio, digit count, and more.
- Logistic Regression binary classifier.
- Accuracy, precision, recall, F1-score, classification report, and confusion matrix.
- Command-line prediction for a new email.

## Project structure
```
phishing-email-detection-model/
├── data/emails.csv
├── outputs/                 # generated when training
├── src_features.py
├── train_model.py
├── predict_email.py
├── requirements.txt
└── README.md
```

## Requirements
Python 3.9 or later.

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Train and evaluate
```bash
python train_model.py
```
The script uses an 80/20 stratified train/test split and saves the trained model, confusion matrix image, and metrics JSON under `outputs/`.

## Predict an email
```bash
python predict_email.py
```
Or:
```bash
python predict_email.py --text "Your account needs urgent verification. Visit http://example.test"
```

## Dataset note
The included CSV is a **synthetic demonstration dataset**, not a representative collection of real emails. Its metrics must not be presented as real-world performance. For a stronger academic evaluation, use a properly licensed, labelled real-world dataset and test on independent data. Remove personal information before using email data.

## Responsible use and limitations
Predictions are warnings for further review, not proof. This text-based model cannot verify sender authentication, inspect attachments, or establish the true destination of every link. Never enter real passwords or sensitive personal information into test messages.
