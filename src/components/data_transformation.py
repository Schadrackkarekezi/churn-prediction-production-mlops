import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

        self.numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_columns = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]

    def clean_data(self, df):
        df = df.drop('customerID', axis=1, errors='ignore')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        return df

    def create_features(self, df):
        df = df.copy()
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])
        df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)

        services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['total_services'] = df[services].apply(lambda x: sum(x.isin(['Yes', 'Fiber optic', 'DSL'])), axis=1)

        return df

    def get_preprocessor(self):
        num_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, self.numerical_columns),
            ('cat', cat_pipeline, self.categorical_columns)
        ])

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Read train and test data")

            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)
            logger.info("Data cleaning completed")

            train_df = self.create_features(train_df)
            test_df = self.create_features(test_df)
            logger.info("Feature engineering completed")

            self.categorical_columns.extend(['SeniorCitizen', 'tenure_group'])
            self.numerical_columns.append('avg_monthly_charges')

            target_column = 'Churn'
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column].map({'Yes': 1, 'No': 0})
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column].map({'Yes': 1, 'No': 0})

            logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")

            preprocessor = self.get_preprocessor()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
            logger.info(f"After SMOTE - Train samples: {len(X_train_resampled)}")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logger.info("Preprocessor saved")

            return (
                X_train_resampled,
                y_train_resampled,
                X_test_transformed,
                y_test,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
