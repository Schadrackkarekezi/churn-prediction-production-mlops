import joblib
import pandas as pd
from src.config import MODEL_PATH


class ChurnPredictor:
    """Churn prediction model wrapper."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        self.model = joblib.load(model_path)

    def predict(self, data: pd.DataFrame) -> list:
        """Predict churn (0 or 1)."""
        predictions = self.model.predict(data)
        return predictions.tolist()

    def predict_proba(self, data: pd.DataFrame) -> list:
        """Predict churn probability."""
        probabilities = self.model.predict_proba(data)[:, 1]
        return probabilities.tolist()


def predict_single(customer_data: dict) -> dict:
    """Make prediction for a single customer."""
    predictor = ChurnPredictor()
    df = pd.DataFrame([customer_data])

    prediction = predictor.predict(df)[0]
    probability = predictor.predict_proba(df)[0]

    return {
        "churn": bool(prediction),
        "churn_probability": round(probability, 4),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    }


# Example usage
if __name__ == "__main__":
    # Example customer
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.2
    }

    result = predict_single(sample_customer)
    print(f"Prediction: {result}")
