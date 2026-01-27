import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("Starting model training")

            models = {
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                'RandomForest': RandomForestClassifier(random_state=42),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
            }

            params = {
                'LogisticRegression': {
                    'C': [0.1, 1, 10]
                },
                'RandomForest': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None]
                },
                'GradientBoosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                },
                'XGBoost': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_name = max(model_report, key=lambda x: model_report[x]['f1'])
            best_model_score = model_report[best_model_name]['f1']
            best_model = model_report[best_model_name]['model']

            logger.info(f"Best model: {best_model_name} with F1 score: {best_model_score:.4f}")

            if best_model_score < 0.5:
                raise CustomException("No suitable model found with F1 > 0.5", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logger.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

            return {
                'best_model': best_model_name,
                'metrics': model_report[best_model_name]
            }

        except Exception as e:
            raise CustomException(e, sys)
