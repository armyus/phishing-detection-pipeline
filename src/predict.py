import os
import joblib

def predict_message(text: str):
    model_path = os.path.join("models", "spam_detector.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model artifact not found. Run train.py first.")
    
    pipeline = joblib.load(model_path)
    clean = text.lower()
    pred = pipeline.predict([clean])[0]
    prob = pipeline.predict_proba([clean])[0]
    
    label = "SPAM / PHISHING" if pred == 1 else "HAM / LEGITIMATE"
    confidence = prob[pred]
    return label, round(confidence * 100, 2)

if __name__ == "__main__":
    test_samples = [
        "Your account is locked. Click link to verify immediately.",
        "Hey team, see you in the conference room in 5 mins."
    ]
    for sample in test_samples:
        lbl, conf = predict_message(sample)
        print(f"Input: '{sample}'\n-> Result: {lbl} (Confidence: {conf}%)\n")