from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# Use absolute path for reliability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained models
model_eff = pickle.load(open(os.path.join(BASE_DIR, "model_efficiency.pkl"), "rb"))
model_bd = pickle.load(open(os.path.join(BASE_DIR, "model_blowdown.pkl"), "rb"))
model_mu = pickle.load(open(os.path.join(BASE_DIR, "model_makeup.pkl"), "rb"))

@app.route("/", methods=["GET"])
def home():
    return "Cooling Tower ML API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Ensure input keys match the names expected by the model
        X = np.array([[
            float(data["hot"]), 
            float(data["cold"]), 
            float(data["wbt"]), 
            float(data["flow"]), 
            float(data["coc"])
        ]])

        return jsonify({
            "efficiency": round(float(model_eff.predict(X)[0]), 2),
            "blowdown": round(float(model_bd.predict(X)[0]), 2),
            "makeup": round(float(model_mu.predict(X)[0]), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)