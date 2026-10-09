import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_data

def train_pipeline():
    df = load_data()
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Modular ML Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("Training model pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    preds = pipeline.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # Save artifact
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "spam_detector.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model artifact saved to {model_path}")

if __name__ == "__main__":
    train_pipeline()