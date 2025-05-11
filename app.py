from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
app = Flask(__name__)

with open("AD.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/")
def about():
    return render_template("about.html")

@app.route("/predict-page")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        sex = float(request.form["sex"])
        cp = float(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = float(request.form["fbs"])
        restecg = float(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = float(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = float(request.form["slope"])
        ca = float(request.form["ca"])
        thal = float(request.form["thal"])

        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        prediction = model.predict(features)

        return jsonify({"prediction": "Positive" if prediction[0] == 1 else "Negative"})
    except Exception as e:
        return jsonify  ({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)