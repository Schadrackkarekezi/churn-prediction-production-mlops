import sys
from src.exception import CustomException
from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    try:
        logger.info("=" * 50)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 50)

        logger.info("Step 1: Data Ingestion")
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        logger.info("Step 2: Data Transformation")
        data_transformation = DataTransformation()
        X_train, y_train, X_test, y_test, _ = data_transformation.initiate_data_transformation(
            train_path, test_path
        )

        logger.info("Step 3: Model Training")
        model_trainer = ModelTrainer()
        result = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

        logger.info("=" * 50)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best Model: {result['best_model']}")
        logger.info(f"F1 Score: {result['metrics']['f1']:.4f}")
        logger.info(f"ROC-AUC: {result['metrics']['roc_auc']:.4f}")
        logger.info("=" * 50)

        return result

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
