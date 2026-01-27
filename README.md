# Customer Churn Prediction - Production MLOps Project

A production-ready, end-to-end machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset.

## Key Features

- **Data Validation**: Automated data quality checks and schema validation
- **Feature Engineering**: 13+ engineered features for improved model performance
- **Hyperparameter Tuning**: RandomizedSearchCV with cross-validation
- **Multiple Models**: Support for GradientBoosting, XGBoost, LightGBM, RandomForest
- **Class Imbalance Handling**: SMOTE oversampling for balanced training
- **Model Evaluation**: Comprehensive metrics, ROC curves, confusion matrices
- **REST API**: FastAPI-based prediction service with Swagger docs
- **Docker Support**: Containerized deployment
- **Logging**: Production-ready logging throughout the pipeline
- **Testing**: Unit tests with pytest

## Project Structure

```
churn-prediction-production-mlops/
├── data/
│   ├── raw/                       # Raw data
│   └── processed/                 # Processed data
├── models/                        # Trained models & metadata
├── reports/
│   └── figures/                   # Evaluation plots
├── logs/                          # Training logs
├── src/
│   ├── config.py                  # Configuration settings
│   ├── data_preprocessing.py      # Data cleaning functions
│   ├── data_validation.py         # Data quality validation
│   ├── feature_engineering.py     # Feature creation
│   ├── train.py                   # Training pipeline
│   ├── tuning.py                  # Hyperparameter tuning
│   ├── evaluation.py              # Model evaluation & plots
│   ├── predict.py                 # Prediction utilities
│   └── logger.py                  # Logging configuration
├── api/
│   └── app.py                     # FastAPI application
├── tests/                         # Unit tests
├── notebooks/                     # Jupyter notebooks for EDA
├── Dockerfile
├── requirements.txt
├── pytest.ini
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/churn-prediction-production-mlops.git
cd churn-prediction-production-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Full training with hyperparameter tuning (recommended)
python -m src.train

# Quick training without tuning
python -m src.train --no-tuning

# Train with specific model
python -m src.train --model XGBoost

# See all options
python -m src.train --help
```

**Training Options:**
- `--model`: GradientBoosting, XGBoost, LightGBM, RandomForest
- `--no-smote`: Disable SMOTE oversampling
- `--no-feature-engineering`: Use only original features
- `--no-tuning`: Skip hyperparameter tuning
- `--n-iter`: Number of tuning iterations (default: 50)

### 3. Run the API

```bash
uvicorn api.app:app --reload
```

The API will be available at `http://localhost:8000`

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 4. Make Predictions

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

### 5. Run Tests

```bash
pytest
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status |
| `/health` | GET | Health check with model status |
| `/predict` | POST | Make churn prediction |
| `/docs` | GET | Swagger documentation |

## Training Pipeline

The training pipeline follows these steps:

1. **Data Loading & Validation** - Load raw data and validate schema/quality
2. **Data Preprocessing** - Clean data, handle missing values
3. **Feature Engineering** - Create 13+ new features
4. **Data Splitting** - Stratified train/test split
5. **Model Training** - Train with optional hyperparameter tuning
6. **Model Evaluation** - Generate metrics and visualization
7. **Model Saving** - Save model and metadata

## Engineered Features

| Feature | Description |
|---------|-------------|
| `tenure_group` | Tenure categorized into groups |
| `avg_monthly_charges` | TotalCharges / (tenure + 1) |
| `total_services` | Count of active services |
| `contract_value` | MonthlyCharges × contract_months |
| `has_partner_dependents` | Both partner and dependents |
| `no_protection` | No security/backup services |
| `auto_payment` | Uses automatic payment method |
| ... | And more |

## Docker Deployment

```bash
# Build image
docker build -t churn-prediction-api .

# Run container
docker run -p 8000:8000 churn-prediction-api

# With environment variables
docker run -p 8000:8000 -e LOG_LEVEL=DEBUG churn-prediction-api
```

## Model Performance

After hyperparameter tuning:

| Metric | Score |
|--------|-------|
| F1 Score (Churn) | ~0.65-0.68 |
| Recall (Churn) | ~0.70-0.75 |
| Precision (Churn) | ~0.60-0.65 |
| ROC-AUC | ~0.82-0.85 |

*Actual scores depend on tuning iterations and random state.*

## Tech Stack

- **ML Framework**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: Pandas, NumPy
- **API**: FastAPI, Pydantic, Uvicorn
- **Visualization**: Matplotlib, Seaborn
- **Testing**: pytest
- **Deployment**: Docker
- **Logging**: Python logging

## Project Highlights (For Resume)

- End-to-end ML pipeline from data validation to deployment
- Production-ready code with logging, error handling, and tests
- Hyperparameter tuning with cross-validation
- Class imbalance handling with SMOTE
- Feature engineering for improved model performance
- RESTful API with automatic documentation
- Containerized deployment with Docker
- Modular, maintainable code structure

## License

MIT
