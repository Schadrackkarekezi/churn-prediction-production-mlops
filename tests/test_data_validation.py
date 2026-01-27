"""
Unit tests for data validation module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_validation import DataValidator, validate_data


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def valid_data(self):
        """Create valid test data."""
        return pd.DataFrame({
            'gender': ['Male', 'Female'],
            'SeniorCitizen': [0, 1],
            'Partner': ['Yes', 'No'],
            'Dependents': ['No', 'Yes'],
            'tenure': [12, 24],
            'PhoneService': ['Yes', 'No'],
            'MultipleLines': ['No', 'No phone service'],
            'InternetService': ['DSL', 'Fiber optic'],
            'OnlineSecurity': ['Yes', 'No'],
            'OnlineBackup': ['No', 'Yes'],
            'DeviceProtection': ['Yes', 'No'],
            'TechSupport': ['No', 'Yes'],
            'StreamingTV': ['Yes', 'No'],
            'StreamingMovies': ['No', 'Yes'],
            'Contract': ['Month-to-month', 'One year'],
            'PaperlessBilling': ['Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check'],
            'MonthlyCharges': [50.0, 75.0],
            'TotalCharges': [600.0, 1800.0],
            'Churn': ['Yes', 'No']
        })

    def test_valid_data_passes(self, valid_data):
        """Test that valid data passes validation."""
        is_valid, report = validate_data(valid_data, is_training=True)
        assert is_valid is True
        assert len(report['errors']) == 0

    def test_missing_columns_fails(self, valid_data):
        """Test that missing required columns fails validation."""
        df = valid_data.drop('tenure', axis=1)
        is_valid, report = validate_data(df, is_training=True)
        assert is_valid is False
        assert any('Missing required columns' in err for err in report['errors'])

    def test_missing_target_for_training_fails(self, valid_data):
        """Test that missing Churn column fails for training data."""
        df = valid_data.drop('Churn', axis=1)
        is_valid, report = validate_data(df, is_training=True)
        assert is_valid is False

    def test_missing_target_ok_for_inference(self, valid_data):
        """Test that missing Churn column is OK for inference data."""
        df = valid_data.drop('Churn', axis=1)
        is_valid, report = validate_data(df, is_training=False)
        assert is_valid is True

    def test_invalid_categorical_values_warning(self, valid_data):
        """Test that invalid categorical values generate warning."""
        df = valid_data.copy()
        df.loc[0, 'gender'] = 'Unknown'
        is_valid, report = validate_data(df, is_training=True)
        # Should pass but with warnings
        assert len(report['warnings']) > 0

    def test_out_of_range_numerical_warning(self, valid_data):
        """Test that out-of-range numerical values generate warning."""
        df = valid_data.copy()
        df.loc[0, 'tenure'] = 200  # Above expected range
        is_valid, report = validate_data(df, is_training=True)
        assert len(report['warnings']) > 0

    def test_non_numeric_charges_fails(self, valid_data):
        """Test that non-numeric charges fail validation."""
        df = valid_data.copy()
        df['MonthlyCharges'] = df['MonthlyCharges'].astype(str)
        is_valid, report = validate_data(df, is_training=True)
        assert is_valid is False


class TestValidationReport:
    """Tests for validation report structure."""

    @pytest.fixture
    def valid_data(self):
        """Create valid test data."""
        return pd.DataFrame({
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['Yes'],
            'Dependents': ['No'],
            'tenure': [12],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['DSL'],
            'OnlineSecurity': ['Yes'],
            'OnlineBackup': ['No'],
            'DeviceProtection': ['Yes'],
            'TechSupport': ['No'],
            'StreamingTV': ['Yes'],
            'StreamingMovies': ['No'],
            'Contract': ['Month-to-month'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [50.0],
            'TotalCharges': [600.0],
            'Churn': ['Yes']
        })

    def test_report_structure(self, valid_data):
        """Test that validation report has correct structure."""
        is_valid, report = validate_data(valid_data, is_training=True)

        assert 'is_valid' in report
        assert 'errors' in report
        assert 'warnings' in report
        assert 'row_count' in report
        assert 'column_count' in report

    def test_report_counts(self, valid_data):
        """Test that report counts are correct."""
        is_valid, report = validate_data(valid_data, is_training=True)

        assert report['row_count'] == len(valid_data)
        assert report['column_count'] == len(valid_data.columns)
