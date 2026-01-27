import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logger


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")

            param = params.get(model_name, {})

            if param:
                gs = GridSearchCV(model, param, cv=3, scoring='f1', n_jobs=-1)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            report[model_name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }

            logger.info(f"{model_name} - F1: {report[model_name]['f1']:.4f}, ROC-AUC: {report[model_name]['roc_auc']:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
