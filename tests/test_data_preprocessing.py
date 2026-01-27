"""
Unit tests for data preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import load_data, clean_data, prepare_features, split_data


class TestDataPreprocessing:
    """Tests for data preprocessing functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return pd.DataFrame({
            'customerID': ['1', '2', '3', '4', '5'],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0, 0, 1],
            'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Dependents': ['No', 'No', 'Yes', 'No', 'Yes'],
            'tenure': [1, 12, 24, 0, 48],
            'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
            'MultipleLines': ['No', 'Yes', 'No phone service', 'No', 'Yes'],
            'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic'],
            'OnlineSecurity': ['No', 'No', 'No internet service', 'Yes', 'Yes'],
            'OnlineBackup': ['Yes', 'No', 'No internet service', 'Yes', 'No'],
            'DeviceProtection': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'TechSupport': ['No', 'No', 'No internet service', 'Yes', 'Yes'],
            'StreamingTV': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'StreamingMovies': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'Two year'],
            'PaperlessBilling': ['Yes', 'No', 'No', 'Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                            'Credit card (automatic)', 'Mailed check'],
            'MonthlyCharges': [29.85, 56.95, 20.00, 45.25, 89.50],
            'TotalCharges': ['29.85', '683.40', '480.00', '', '4296.00'],
            'Churn': ['No', 'No', 'No', 'Yes', 'No']
        })

    def test_clean_data_converts_total_charges(self, sample_data):
        """Test that TotalCharges is converted to numeric."""
        df = clean_data(sample_data)
        assert df['TotalCharges'].dtype in ['float64', 'float32']

    def test_clean_data_fills_missing_total_charges(self, sample_data):
        """Test that missing TotalCharges are filled with 0."""
        df = clean_data(sample_data)
        assert df['TotalCharges'].isna().sum() == 0
        # The row with empty string should now be 0
        assert df.loc[3, 'TotalCharges'] == 0

    def test_clean_data_drops_customer_id(self, sample_data):
        """Test that customerID is dropped."""
        df = clean_data(sample_data)
        assert 'customerID' not in df.columns

    def test_prepare_features_separates_x_y(self, sample_data):
        """Test that features and target are separated correctly."""
        df = clean_data(sample_data)
        X, y = prepare_features(df)

        assert 'Churn' not in X.columns
        assert len(y) == len(X)
        assert set(y.unique()) == {0, 1}

    def test_split_data_stratified(self, sample_data):
        """Test that data is split with stratification."""
        df = clean_data(sample_data)
        X, y = prepare_features(df)

        # Need more data for stratified split
        X = pd.concat([X] * 20, ignore_index=True)
        y = pd.concat([y] * 20, ignore_index=True)

        X_train, X_test, y_train, y_test = split_data(X, y)

        # Check sizes
        assert len(X_train) > len(X_test)
        assert len(X_train) + len(X_test) == len(X)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_clean_data_handles_all_missing_total_charges(self):
        """Test handling when all TotalCharges are missing."""
        df = pd.DataFrame({
            'customerID': ['1'],
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['Yes'],
            'Dependents': ['No'],
            'tenure': [0],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['DSL'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['Yes'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No'],
            'Contract': ['Month-to-month'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [29.85],
            'TotalCharges': [''],
            'Churn': ['No']
        })

        cleaned = clean_data(df)
        assert cleaned['TotalCharges'].iloc[0] == 0
