from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os

application = Flask(__name__)
app = application

# Load model and preprocessor
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
preprocessor = pickle.load(open(PREPROCESSOR_PATH, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [x for x in request.form.values()]
        columns = list(request.form.keys())
        df = pd.DataFrame([data], columns=columns)
        transformed = preprocessor.transform(df)
        prediction = model.predict(transformed)
        return render_template(
            "index.html",
            prediction_text=f"Predicted Value: {prediction[0]:.2f}"
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Prediction Failed: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
