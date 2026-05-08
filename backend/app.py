"""
Flask REST API — Urinalysis UTI/Kidney Disease Predictor
Endpoints:
  POST /predict         — predict from form values
  GET  /metadata        — model info & comparison metrics
  GET  /eda/<plot_name> — serve EDA plot images
  GET  /health          — health check
"""

import os, pickle, json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

BASE   = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")
PLOTS  = os.path.join(BASE, "eda_plots")

app = Flask(__name__)
CORS(app)  # Allow React frontend

# ── Load artefacts ────────────────────────────────────────
with open(os.path.join(MODELS, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODELS, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODELS, "feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)
with open(os.path.join(MODELS, "metadata.json")) as f:
    METADATA = json.load(f)

print(f"✅ Model loaded  : {METADATA['best_model_name']}")
print(f"   Features      : {len(FEATURE_NAMES)}")


def engineer_features(raw: dict) -> np.ndarray:
    """
    Apply the same feature engineering as in ml_pipeline.py
    Input: dict with the 14 original feature values
    Output: scaled feature array for prediction
    """
    age               = float(raw.get("Age", 0))
    glucose           = float(raw.get("Glucose", 0))
    protein           = float(raw.get("Protein", 0))
    ph                = float(raw.get("pH", 7))
    sg                = float(raw.get("Specific Gravity", 1.015))
    wbc               = float(raw.get("WBC", 0))
    rbc               = float(raw.get("RBC", 0))
    epi               = float(raw.get("Epithelial Cells", 0))
    mucous            = float(raw.get("Mucous Threads", 0))
    amorph            = float(raw.get("Amorphous Urates", 0))
    bacteria          = float(raw.get("Bacteria", 0))
    gender            = float(raw.get("Gender", 0))
    color             = float(raw.get("Color", 3))
    transparency      = float(raw.get("Transparency", 1))

    # Derived features (must match ml_pipeline.py)
    infection_score   = wbc + bacteria + rbc
    sediment_score    = epi + mucous + amorph
    ph_abnormal       = 1 if (ph < 4.5 or ph > 8.5) else 0
    sg_abnormal       = 1 if (sg < 1.005 or sg > 1.030) else 0
    has_protein       = 1 if protein > 0 else 0
    has_glucose       = 1 if glucose > 0 else 0
    if age <= 12:   age_group = 0
    elif age <= 18: age_group = 1
    elif age <= 40: age_group = 2
    elif age <= 65: age_group = 3
    else:           age_group = 4

    feature_vector = [
        age, gender, color, transparency, glucose, protein, ph, sg,
        wbc, rbc, epi, mucous, amorph, bacteria,
        infection_score, sediment_score, ph_abnormal, sg_abnormal,
        has_protein, has_glucose, age_group
    ]
    return np.array(feature_vector).reshape(1, -1)


# ── ROUTES ───────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": METADATA["best_model_name"]})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON body with urinalysis values.
    Returns prediction, probability, and clinical interpretation.
    """
    try:
        data = request.get_json(force=True)

        # Build + scale feature vector
        X_raw    = engineer_features(data)
        X_df     = pd.DataFrame(X_raw, columns=FEATURE_NAMES)
        X_scaled = scaler.transform(X_df)

        pred     = int(model.predict(X_scaled)[0])
        proba    = float(model.predict_proba(X_scaled)[0][pred])
        pos_prob = float(model.predict_proba(X_scaled)[0][1])

        label = "Positive" if pred == 1 else "Negative"

        # Risk level based on positive probability
        if pos_prob >= 0.80:   risk = "High Risk"
        elif pos_prob >= 0.50: risk = "Moderate Risk"
        elif pos_prob >= 0.30: risk = "Low Risk"
        else:                  risk = "Minimal Risk"

        # Clinical flags
        flags = []
        if float(data.get("WBC", 0)) > 5:
            flags.append("Elevated WBC (possible infection)")
        if float(data.get("Bacteria", 0)) >= 4:
            flags.append("High bacteria count")
        if float(data.get("Protein", 0)) > 0:
            flags.append("Proteinuria detected")
        if float(data.get("Glucose", 0)) > 0:
            flags.append("Glucosuria detected")
        ph = float(data.get("pH", 7))
        if ph < 4.5 or ph > 8.5:
            flags.append(f"Abnormal pH ({ph})")
        sg = float(data.get("Specific Gravity", 1.015))
        if sg < 1.005 or sg > 1.030:
            flags.append(f"Abnormal Specific Gravity ({sg})")

        return jsonify({
            "prediction"         : pred,
            "label"              : label,
            "confidence"         : round(proba * 100, 2),
            "positive_probability": round(pos_prob * 100, 2),
            "risk_level"         : risk,
            "clinical_flags"     : flags,
            "disclaimer"         : (
                "⚠️ This is an AI-assisted screening tool only. "
                "Results must be confirmed by a licensed medical professional."
            )
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/metadata")
def metadata():
    return jsonify(METADATA)


@app.route("/eda/<plot_name>")
def eda_plot(plot_name):
    """Serve EDA plot images."""
    path = os.path.join(PLOTS, plot_name)
    if not os.path.exists(path):
        return jsonify({"error": "Plot not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/eda_list")
def eda_list():
    plots = sorted(os.listdir(PLOTS))
    return jsonify({"plots": plots})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
