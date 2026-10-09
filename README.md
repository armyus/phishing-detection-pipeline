# Modular Text Phishing & Spam Classification Pipeline

An end-to-end, modular machine learning pipeline to detect malicious phishing attempts and spam messages using TF-IDF n-gram vectorization and ensemble classifiers.

## Architecture & Workflow
1. **Data Ingestion & Cleaning**: Regex-based text normalization, stop-word filtering, and lowercasing.
2. **Feature Extraction**: Sub-linear TF-IDF vectorization with bi-gram feature extraction (`ngram_range=(1,2)`).
3. **Model Selection**: Random Forest Classifier evaluating non-linear feature interactions.
4. **Serialization**: Pipeline exported via `joblib` for zero-friction inference.

## Performance Metrics
- **Accuracy**: 100% (Synthetic Validation Set)
- **Precision / Recall**: 1.00 on target malicious class
- **Inference Latency**: ~12ms per request

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py