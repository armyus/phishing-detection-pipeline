# Phishing Detection Pipeline

A lightweight phishing and spam detection project that trains a text-classification model, saves the model artifact, and exposes it through a Flask web app and API. The project uses a TF-IDF + n-gram pipeline with a Random Forest classifier to classify incoming text as legitimate or malicious.

## Project Structure

```text
phishing-detection-pipeline/
├── app.py                  # Flask app and prediction API
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── data/                   # Data assets (if used in future expansions)
├── models/                 # Saved trained model artifacts
├── src/
│   ├── data_loader.py      # Synthetic dataset generation and cleaning
│   ├── predict.py          # CLI prediction utility
│   └── train.py            # Model training and artifact export
├── templates/
│   └── index.html          # Web dashboard UI
└── .venv/                  # Local virtual environment
```

## How It Works

1. Data is generated in `src/data_loader.py` as a small synthetic phishing/spam dataset.
2. Text is normalized by lowercasing and stripping punctuation.
3. `TfidfVectorizer` converts the cleaned text into features using unigrams and bigrams.
4. A `RandomForestClassifier` is trained to classify text as:
   - `0` = legitimate / ham
   - `1` = phishing / spam
5. The trained pipeline is serialized to `models/spam_detector.joblib`.
6. The Flask app loads the model and exposes it via a browser dashboard and `/api/predict` endpoint.

## Model Details

- Feature extraction: TF-IDF with `ngram_range=(1, 2)`
- Model: Random Forest
- Target labels:
  - `0` = legitimate
  - `1` = threat

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install Flask
```

## Train the Model

```bash
python src/train.py
```

This creates the serialized model at:

```text
models/spam_detector.joblib
```

## Run the Web App

```bash
python app.py
```

Then open:

```text
http://localhost:5000/
```

The homepage renders a simple dashboard for pasting text and checking whether it looks like phishing or spam.

## Predict from the Command Line

```bash
python src/predict.py
```

This runs a few sample messages and prints the classification result.

## API Usage

Send a POST request to the prediction endpoint:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Your account is locked. Click here to verify immediately."}'
```

Example response:

```json
{
  "is_threat": true,
  "confidence": 99.3,
  "prob_safe": 0.7,
  "prob_threat": 99.3
}
```

## Notes

- The current dataset is synthetic and intentionally compact for demonstration purposes.
- The project is structured to be easy to extend with a larger labeled dataset, additional models, or a production deployment pipeline.
- The web app currently loads the model once at startup and serves predictions without requiring a separate API service.
