from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

print("Loading model...")
model = pickle.load(open('model.pkl', 'rb'))
transformer = pickle.load(open('transformer.pkl', 'rb'))
print("Model loaded successfully.")

EXPECTED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

NUMERIC_COLS = [
    "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "MonthlyCharges", "TotalCharges"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        input_df = pd.DataFrame([data])
        input_df = input_df[EXPECTED_COLUMNS]

        for col in NUMERIC_COLS:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        transformed = transformer.transform(input_df)
        transformed = np.array(transformed, dtype=float)

        prediction = model.predict(transformed)
        probability = model.predict_proba(transformed)[0][1]

        return jsonify({
            'prediction': int(prediction[0]),
            'churn_probability': round(float(probability), 4),
            'status': 'success'
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)