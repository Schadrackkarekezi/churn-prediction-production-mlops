import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    pass


class DataValidator:
    REQUIRED_COLUMNS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]

    CATEGORICAL_VALUES = {
        'gender': ['Male', 'Female'],
        'Partner': ['Yes', 'No'],
        'Dependents': ['Yes', 'No'],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['Yes', 'No', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['Yes', 'No', 'No internet service'],
        'TechSupport': ['Yes', 'No', 'No internet service'],
        'StreamingTV': ['Yes', 'No', 'No internet service'],
        'StreamingMovies': ['Yes', 'No', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ]
    }

    NUMERICAL_RANGES = {
        'SeniorCitizen': (0, 1),
        'tenure': (0, 100),
        'MonthlyCharges': (0, 500),
        'TotalCharges': (0, 10000)
    }

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[bool, Dict]:
        self.errors = []
        self.warnings = []

        self._check_required_columns(df, is_training)
        self._check_missing_values(df)
        self._check_categorical_values(df)
        self._check_numerical_ranges(df)
        self._check_data_types(df)
        self._check_duplicates(df)

        is_valid = len(self.errors) == 0

        report = {
            'is_valid': is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'row_count': len(df),
            'column_count': len(df.columns)
        }

        if not is_valid:
            logger.error(f"Validation failed: {len(self.errors)} errors")
            for error in self.errors:
                logger.error(f"  - {error}")

        for warning in self.warnings:
            logger.warning(f"  - {warning}")

        return is_valid, report

    def _check_required_columns(self, df, is_training):
        required = self.REQUIRED_COLUMNS.copy()
        if is_training:
            required.append('Churn')

        missing = set(required) - set(df.columns)
        if missing:
            self.errors.append(f"Missing columns: {missing}")

    def _check_missing_values(self, df):
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    pct = (missing_count / len(df)) * 100
                    if pct > 5:
                        self.errors.append(f"'{col}' has {pct:.1f}% missing")
                    else:
                        self.warnings.append(f"'{col}' has {missing_count} missing ({pct:.1f}%)")

    def _check_categorical_values(self, df):
        for col, valid_values in self.CATEGORICAL_VALUES.items():
            if col in df.columns:
                invalid = set(df[col].dropna().unique()) - set(valid_values)
                if invalid:
                    self.warnings.append(f"'{col}' has unexpected values: {invalid}")

    def _check_numerical_ranges(self, df):
        for col, (min_val, max_val) in self.NUMERICAL_RANGES.items():
            if col in df.columns:
                if df[col].min() < min_val:
                    self.warnings.append(f"'{col}' below {min_val}: min={df[col].min()}")
                if df[col].max() > max_val:
                    self.warnings.append(f"'{col}' above {max_val}: max={df[col].max()}")

    def _check_data_types(self, df):
        for col in ['MonthlyCharges', 'TotalCharges']:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                self.errors.append(f"'{col}' should be numeric, got {df[col].dtype}")

    def _check_duplicates(self, df):
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            self.warnings.append(f"Found {duplicates} duplicate rows")


def validate_data(df: pd.DataFrame, is_training: bool = True) -> Tuple[bool, Dict]:
    validator = DataValidator()
    return validator.validate(df, is_training)
