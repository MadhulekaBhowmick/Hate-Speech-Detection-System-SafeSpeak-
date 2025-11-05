from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import os
import traceback

app = Flask(__name__, static_url_path='/static')

# The base directory of the app
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer from the models directory
model_path = os.path.join(base_dir, 'models', 'hate_speech_model_only.joblib')
vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer_only.joblib')

print("Model path:", model_path)
print("Vectorizer path:", vectorizer_path)

# Debugging statements
print("Current Working Directory:", os.getcwd())

# Load the model and vectorizer
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Debugging: Check if the vectorizer has been loaded properly
    print("Model and vectorizer loaded successfully.")
    
    # Check the contents of the vectorizer
    if hasattr(vectorizer, 'vocabulary_'):
        print("Vectorizer vocabulary loaded. Vocabulary size:", len(vectorizer.vocabulary_))
        print("Vectorizer vocabulary:", vectorizer.vocabulary_)  # Add this line
    else:
        print("Error: Vectorizer does not have a vocabulary attribute.")
else:
    print("Error: Model or vectorizer not found.")

@app.after_request
def add_header(response):
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    response.expires = 0
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    print("Detection page accessed")
    return render_template('detection.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tweet = data['tweet']
        
        # Vectorize the input tweet using the vectorizer
        tweet_vector = vectorizer.transform([tweet])
        print("Shape of input vector:", tweet_vector.shape)  # Log the shape

        # Predict the label and get probabilities
        prediction = model.predict(tweet_vector)[0]
        prediction_proba = model.predict_proba(tweet_vector)[0]
        print("Prediction probabilities:", prediction_proba)  # Print probability for each class

        # Map the prediction to a result
        if prediction == 0:  # Assuming 0 is "Hate Speech"
            result = "Hate Speech"
            color = "black"
        elif prediction == 1:  # Assuming 1 is "Offensive Language"
            result = "Offensive Language"
            color = "red"
        else:  # Assuming 2 is "Neither"
            result = "Neither"
            color = "green"
        
        return jsonify({"result": result, "color": color, "probabilities": prediction_proba.tolist()})

    except Exception as e:
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())  # Log the full traceback
        return jsonify({"error": str(e)}), 500

# Route to serve static files for debugging
@app.route('/static/<path:filename>')
def send_static(filename):
    print(f"Static file requested: {filename}")
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)



