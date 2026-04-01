from flask import Flask, request, jsonify
import pickle
import numpy as np
import warnings

# Ignore the valid feature names warning to keep Render logs clean
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Helper function to load a set of 3 models for a specific algorithm
def load_models(prefix):
    return {
        "eff": pickle.load(open(f"{prefix}_eff.pkl", "rb")),
        "bd": pickle.load(open(f"{prefix}_bd.pkl", "rb")),
        "mu": pickle.load(open(f"{prefix}_mu.pkl", "rb"))
    }

# Load all 15 models into memory
models_dict = {
    "Original Baseline": load_models("model"),
    "Linear Regression": load_models("lr"),
    "Random Forest": load_models("rf"),
    "Gradient Boosting": load_models("gb"),
    "Support Vector": load_models("svr")
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract inputs from the mobile app
        hot = float(data["hot"])
        cold = float(data["cold"])
        wbt = float(data["wbt"])
        flow = float(data["flow"])
        coc = float(data["coc"])
        
        # Format for the ML models
        X = np.array([[hot, cold, wbt, flow, coc]])
        
        # Calculate Evaporation Loss mathematically based on the report
        evaporation_loss = 0.00085 * 1.8 * flow * (hot - cold)

        results = {}
        
        # Loop through all 5 algorithms and generate predictions
        for algo_name, alg_models in models_dict.items():
            results[algo_name] = {
                "efficiency": round(float(alg_models["eff"].predict(X)[0]), 2),
                "blowdown": round(float(alg_models["bd"].predict(X)[0]), 2),
                "water_makeup": round(float(alg_models["mu"].predict(X)[0]), 2),
                "evaporation_loss": round(evaporation_loss, 2)
            }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
