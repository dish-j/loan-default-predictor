# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/loan_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        features = [float(x) for x in request.form.values()]
        final_features = scaler.transform([features])
        prediction = model.predict(final_features)[0]
        output = "Will Default" if prediction == 1 else "Will Not Default"

        return render_template('index.html', prediction_text=f"Prediction: {output}")

if __name__ == "__main__":
    app.run(debug=True)
