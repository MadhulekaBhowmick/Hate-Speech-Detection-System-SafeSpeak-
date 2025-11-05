from joblib import load
# Importing the pickle module
import pickle  

# Load the hate speech model
try:
    model = load('hate_speech_model.joblib')  
    print("Model loaded successfully.")
except Exception as e: 
    print(f"Error loading model: {e}")

# Load the vectorizer
try:
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading vectorizer: {e}")
