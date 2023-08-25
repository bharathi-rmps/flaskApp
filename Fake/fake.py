from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model and vectorizer
try:
    clf = load('model.joblib')
    vectorizer = load('vectorizer.joblib')
except Exception as e:
    print("Error loading model or vectorizer:", str(e))
    exit(1)

def predict_fake_text():
    try:
        text = request.json.get('text')
        if text is None:
            return jsonify({'error': 'No text provided'}), 400

        text_tfidf = vectorizer.transform([text])
        prediction = clf.predict(text_tfidf)
        label = "fake" if prediction[0] == 1 else "real"
        return jsonify({'text': text, 'prediction': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_fake_file():
    try:
        file = request.files.get('file')
        if file is None:
            return jsonify({'error': 'No file provided'}), 400

        df = pd.read_csv(file)
        text_tfidf = vectorizer.transform(df['text'])
        predictions = clf.predict(text_tfidf)
        df['prediction'] = ['fake' if pred == 1 else 'real' for pred in predictions]

        return jsonify(df.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500