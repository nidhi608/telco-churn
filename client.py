import requests

url = "http://127.0.0.1:5000/predict"

# ✅ Categorical columns must be STRINGS matching the training data
data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": 1,                        # Already encoded to int during training
    "Dependents": 0,                     # Already encoded to int during training
    "tenure": 12,
    "PhoneService": 1,                   # Already encoded to int during training
    "MultipleLines": 0,                  # Already encoded to int during training
    "InternetService": "Fiber optic",    # ← string, NOT 2
    "OnlineSecurity": "No",              # ← string, NOT 0
    "OnlineBackup": "Yes",               # ← string
    "DeviceProtection": "No",            # ← string
    "TechSupport": "No",                 # ← string
    "StreamingTV": "Yes",                # ← string
    "StreamingMovies": "No",             # ← string
    "Contract": "Month-to-month",        # ← string, NOT 0
    "PaperlessBilling": "Yes",           # ← string
    "PaymentMethod": "Electronic check", # ← string, NOT 0
    "MonthlyCharges": 70.0,
    "TotalCharges": 800.0
}

try:
    response = requests.post(url, json=data, timeout=5)
    print("Status code:", response.status_code)
    print("Response:", response.json())
except requests.exceptions.ConnectionError:
    print("Error: Could not connect. Run 'python app.py' first!")
except Exception as e:
    print("Error:", e)