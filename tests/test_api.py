"""
Unit tests for FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


class TestAPI:
    """Tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked model."""
        # Mock the model before importing app
        with patch('api.app.MODEL_PATH') as mock_path:
            mock_path.exists.return_value = False
            from api.app import app
            return TestClient(app)

    @pytest.fixture
    def sample_customer(self):
        """Sample customer data for testing."""
        return {
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

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct response."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert response.json()["status"] == "running"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
        assert "model_loaded" in response.json()

    def test_predict_without_model(self, client, sample_customer):
        """Test prediction endpoint returns 503 when model not loaded."""
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_predict_missing_fields(self, client):
        """Test prediction with missing required fields."""
        incomplete_data = {"gender": "Female", "tenure": 12}
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_types(self, client, sample_customer):
        """Test prediction with invalid data types."""
        sample_customer["tenure"] = "invalid"
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 422


class TestPredictionWithModel:
    """Tests for prediction endpoint with mocked model."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.predict.return_value = np.array([1])
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        return model

    @pytest.fixture
    def client_with_model(self, mock_model):
        """Create test client with mocked model loaded."""
        with patch('api.app.model', mock_model):
            from api.app import app
            return TestClient(app)

    @pytest.fixture
    def sample_customer(self):
        """Sample customer data."""
        return {
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

    def test_predict_returns_correct_structure(self, client_with_model, sample_customer):
        """Test prediction returns correct response structure."""
        # Patch model at module level
        import api.app as app_module

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        app_module.model = mock_model

        client = TestClient(app_module.app)
        response = client.post("/predict", json=sample_customer)

        assert response.status_code == 200
        data = response.json()
        assert "churn" in data
        assert "churn_probability" in data
        assert "risk_level" in data

    def test_risk_level_high(self, client_with_model, sample_customer):
        """Test high risk level for high probability."""
        import api.app as app_module

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        app_module.model = mock_model

        client = TestClient(app_module.app)
        response = client.post("/predict", json=sample_customer)

        assert response.json()["risk_level"] == "High"

    def test_risk_level_low(self, client_with_model, sample_customer):
        """Test low risk level for low probability."""
        import api.app as app_module

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        app_module.model = mock_model

        client = TestClient(app_module.app)
        response = client.post("/predict", json=sample_customer)

        assert response.json()["risk_level"] == "Low"
