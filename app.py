from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)
model = None

# Load model
if os.path.exists("heart_model.pkl"):
    model = joblib.load("heart_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json()
    age = data["age"]
    bmi = data["bmi"]
    chol = data["chol"]

    input_data = np.array([[age, bmi, chol]])
    prediction = model.predict(input_data)[0]
    label = "🚨 High Risk" if prediction == 1 else "✅ Low Risk"

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
