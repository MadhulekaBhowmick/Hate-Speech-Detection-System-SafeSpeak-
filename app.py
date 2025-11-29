from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import os
import traceback

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "hate_speech_model_only.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer_only.joblib")

print("Model path:", MODEL_PATH)
print("Vectorizer path:", VECTORIZER_PATH)
print("CWD:", os.getcwd())

# Load model + vectorizer once, on startup
model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model and vectorizer loaded successfully.")

        # Optional: sanity checks
        if hasattr(vectorizer, "vocabulary_"):
            print("Vectorizer vocabulary size:", len(vectorizer.vocabulary_))
        if hasattr(model, "classes_"):
            print("Model classes:", model.classes_)
    except Exception as e:
        print("Error loading artifacts:", e)
        print(traceback.format_exc())
else:
    print("Error: Model or vectorizer file not found under /models.")

@app.after_request
def add_header(response):
    # Stop aggressive caching during dev
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    response.expires = 0
    return response

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detection")
def detection():
    print("Detection page accessed")
    return render_template("detection.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or vectorizer is None:
            return jsonify({"error": "Model/vectorizer not loaded"}), 500

        data = request.get_json(silent=True) or {}
        tweet = data.get("tweet", "").strip()

        if not tweet:
            return jsonify({"error": "Empty input"}), 400

        # Transform input
        tweet_vec = vectorizer.transform([tweet])
        print("Input vector shape:", tweet_vec.shape)

        # Predict
        pred_idx = model.predict(tweet_vec)[0]
        proba = model.predict_proba(tweet_vec)[0].tolist()

        # If your model.classes_ == [0,1,2] map accordingly
        # If you trained with names, use them directly.
        label_map = {
            0: ("Hate Speech", "black"),
            1: ("Offensive Language", "red"),
            2: ("Neither", "green")
        }

        # If model.classes_ are not [0,1,2], remap with classes_
        # (Assumes three classes; adjust if you changed that.)
        if hasattr(model, "classes_") and list(model.classes_) != [0, 1, 2]:
            # Build a dict from class value to human label/color above
            translated = {}
            for i, cls_val in enumerate(model.classes_):
                # fall back to string label if your classes_ are strings
                if cls_val in label_map:
                    translated[i] = label_map[cls_val]
                else:
                    # default for unknown label
                    translated[i] = (str(cls_val), "blue")
            # predicted index currently corresponds to classes_ order; get label from that
            pred_label, color = translated[list(model.classes_).index(pred_idx)]
        else:
            pred_label, color = label_map.get(pred_idx, ("Unknown", "blue"))

        return jsonify({
            "result": pred_label,
            "color": color,
            "probabilities": proba,
            "classes": getattr(model, "classes_", [0,1,2])  # for the frontend to align bars
        })

    except Exception as e:
        print("Error during prediction:", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Optional: serve static for debug prints
@app.route("/static/<path:filename>")
def send_static(filename):
    print(f"Static file requested: {filename}")
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    # You said you want to keep http://127.0.0.1:5000/
    app.run(debug=True, host="127.0.0.1", port=5000)
