# Customer Churn Prediction

End-to-end machine learning pipeline for predicting customer churn.

## Project Structure

```
churn-prediction/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── artifacts/              # Saved models
├── notebook/               # EDA notebooks
├── templates/              # Web UI
├── application.py          # FastAPI app
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python -m src.pipeline.train_pipeline
```

## Run API

```bash
python application.py
```

API available at `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

## API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Docker

```bash
docker build -t churn-prediction .
docker run -p 8000:8000 churn-prediction
```

## Features

- SMOTE for class imbalance
- Feature engineering (tenure groups, service counts)
- Multiple models (GradientBoosting, XGBoost, RandomForest)
- Grid search hyperparameter tuning
- FastAPI with automatic docs

## Tech Stack

- scikit-learn, XGBoost
- Pandas, NumPy
- FastAPI, Uvicorn
- Docker
