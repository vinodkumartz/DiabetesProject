from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load(r"D:\DiabetesProject\DiabetesProject\backend\models\XGBoost.pkl")

# **Selected Important Features (Based on Model Importance)**
important_features = [
    "race", "gender", "age", "admission_type_id", "num_lab_procedures", 
    "num_medications", "number_inpatient", "insulin", "change", "diabetesMed"
]

# Define categorical mappings
categorical_mappings = {
    "race": {"Caucasian": 0, "AfricanAmerican": 1, "Asian": 2, "Hispanic": 3, "Other": 4},
    "gender": {"Male": 0, "Female": 1},
    "insulin": {"No": 0, "Up": 1, "Down": 2, "Steady": 3},
    "change": {"No": 0, "Ch": 1},
    "diabetesMed": {"No": 0, "Yes": 1}
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    print("Received data:", data)  # Debugging

    try:
        features = np.array(data["features"])
        
        # Handle missing values
        if len(features) < 177:
            features = np.pad(features, (0, 177 - len(features)), 'constant', constant_values=0)

        features = features.reshape(1, -1)  # Ensure correct shape
        prediction = model.predict(features)[0]
        print("Raw Prediction:", prediction)  # Debugging

        explanation_map = {
            "<30": "High readmission risk. Frequent inpatient visits, high medication count, or severe condition.",
            "1": "Moderate readmission risk. Ensure proper diabetes management, follow-ups, and medication adherence.",
            "0": "Low readmission risk. Maintain current treatment plan and lifestyle improvements."
        }

        prediction_str = str(prediction) if prediction in ["<30", "1", "0"] else str(int(prediction))
        explanation = explanation_map.get(prediction_str, "Unknown prediction result.")

        return jsonify({"prediction": str(prediction), "explanation": explanation})

    
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)


