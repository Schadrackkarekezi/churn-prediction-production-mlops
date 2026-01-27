import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import (
    RAW_DATA_PATH, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    TARGET, TEST_SIZE, RANDOM_STATE
)


def load_data(path=None):
    if path is None:
        path = RAW_DATA_PATH
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    return df


def prepare_features(df):
    X = df.drop(TARGET, axis=1)
    y = df[TARGET].map({'Yes': 1, 'No': 0})
    return X, y


def get_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def get_full_pipeline(model, use_smote=True):
    preprocessor = get_preprocessor()

    if use_smote:
        return ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('classifier', model)
        ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
