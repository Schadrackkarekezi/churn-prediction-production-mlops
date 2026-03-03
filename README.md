# Customer Churn Prediction — Production MLOps Pipeline

An end-to-end machine learning system that predicts telecom customer churn and serves real-time predictions through a FastAPI REST API, containerized with Docker and deployed to AWS ECS via CI/CD.

## Overview

Customer churn costs telecom companies significantly more than retention. This project builds a production-ready ML pipeline that:
- Ingests and transforms raw customer data (7,043 records, 26.5% churn rate)
- Engineers features and handles class imbalance with SMOTE
- Trains and compares 4 ML models with hyperparameter tuning
- Serves predictions via a FastAPI API with an interactive web UI
- Deploys automatically to AWS ECS through GitHub Actions CI/CD

## Architecture

```
Raw Data → Data Ingestion → Feature Engineering → Model Training → Artifact Storage
                                                                         ↓
                                                                  Prediction Pipeline
                                                                         ↓
                                                                   FastAPI Server
                                                                         ↓
                                                               Web UI + REST API
                                                                         ↓
                                                              Docker → AWS ECR/ECS
```

## Demo

![Churn Prediction Demo](assets/demo.gif)

## Key Features

| Feature | Details |
|---------|---------|
| **ML Models** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| **Hyperparameter Tuning** | GridSearchCV with 3-fold CV, optimized on F1-score |
| **Class Imbalance** | SMOTE oversampling for balanced training |
| **Feature Engineering** | Tenure groups, average monthly charges, total services count |
| **Preprocessing** | StandardScaler (numerical) + OneHotEncoder (categorical) via sklearn Pipeline |
| **API** | FastAPI with Swagger docs, health check, and web UI |
| **Deployment** | Docker + GitHub Actions CI/CD → AWS ECR/ECS |

## Project Structure






```
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # Data loading and train/test split
│   │   ├── data_transformation.py   # Feature engineering and preprocessing
│   │   └── model_trainer.py         # Model training and evaluation
│   ├── pipeline/
│   │   ├── train_pipeline.py        # Orchestrates full training workflow
│   │   └── predict_pipeline.py      # Loads model and generates predictions
│   ├── exception.py                 # Custom exception handling
│   ├── logger.py                    # Logging configuration
│   └── utils.py                     # Utility functions
├── artifacts/                       # Trained model and preprocessor (.pkl)
├── data/raw/                        # Raw dataset
├── notebook/                        # Exploratory Data Analysis
├── templates/                       # Web UI (HTML/JS)
├── application.py                   # FastAPI application
├── Dockerfile                       # Container configuration
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
└── .github/workflows/aws.yml       # CI/CD pipeline
```

## Getting Started

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)

### Installation

```bash
git clone https://github.com/yourusername/churn-prediction-production-mlops.git
cd churn-prediction-production-mlops
pip install -r requirements.txt
```

### Train the Model

```bash
python -m src.pipeline.train_pipeline
```

### Run the API

```bash
python application.py
```

The API will be available at:
- Web UI: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### Docker Deployment

```bash
docker build -t churn-prediction .
docker run -p 8000:8000 churn-prediction
```

## API Usage

**POST** `/predict`

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

**Response:**
```json
{
  "churn": true,
  "churn_probability": 0.73,
  "risk_level": "High"
}
```

## Tech Stack

- **ML**: scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Data**: Pandas, NumPy
- **API**: FastAPI, Uvicorn, Pydantic
- **Frontend**: HTML, CSS, JavaScript
- **DevOps**: Docker, GitHub Actions
- **Cloud**: AWS ECR, AWS ECS
