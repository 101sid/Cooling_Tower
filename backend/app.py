@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract values for both ML and manual calculations
        hot = float(data["hot"])
        cold = float(data["cold"])
        wbt = float(data["wbt"])
        flow = float(data["flow"])
        coc = float(data["coc"])
        
        # Prepare input for ML models
        X = np.array([[hot, cold, wbt, flow, coc]])

        # Calculate Evaporation Loss manually based on the report's formula
        # E = 0.00085 * 1.8 * C * (T1 - T2)
        evaporation_loss = 0.00085 * 1.8 * flow * (hot - cold)

        # Return updated JSON
        return jsonify({
            "efficiency": round(float(model_eff.predict(X)[0]), 2),
            "blowdown": round(float(model_bd.predict(X)[0]), 2),
            "water_makeup": round(float(model_mu.predict(X)[0]), 2), # Renamed
            "evaporation_loss": round(evaporation_loss, 2) # Added
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
