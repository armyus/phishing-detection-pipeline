import os
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model artifact
MODEL_PATH = os.path.join("models", "spam_detector.joblib")
pipeline = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    
    if not pipeline:
        return jsonify({"error": "Model not loaded"}), 500

    clean_text = text.lower()
    pred = int(pipeline.predict([clean_text])[0])
    prob = pipeline.predict_proba([clean_text])[0]

    return jsonify({
        "is_threat": bool(pred == 1),
        "confidence": round(float(prob[pred]) * 100, 2),
        "prob_safe": round(float(prob[0]) * 100, 1),
        "prob_threat": round(float(prob[1]) * 100, 1)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)