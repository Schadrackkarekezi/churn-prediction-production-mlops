"""
Model Training Pipeline
Production-ready end-to-end ML training pipeline.
"""
import argparse
import joblib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.config import MODEL_PATH, RANDOM_STATE, RAW_DATA_PATH
from src.data_preprocessing import load_data, clean_data, prepare_features, split_data
from src.feature_engineering import create_features, get_feature_columns
from src.data_validation import validate_data
from src.tuning import tune_hyperparameters, get_model_and_params
from src.evaluation import ModelEvaluator
from src.logger import setup_logger

# Setup logger
logger = setup_logger("churn_prediction")


class ChurnModelTrainer:
    """End-to-end churn model training pipeline."""

    def __init__(
        self,
        model_name: str = "GradientBoosting",
        use_smote: bool = True,
        use_feature_engineering: bool = True,
        tune_hyperparams: bool = True,
        n_iter: int = 50
    ):
        """
        Initialize trainer.

        Args:
            model_name: Model to train (GradientBoosting, XGBoost, LightGBM, RandomForest)
            use_smote: Whether to use SMOTE for class imbalance
            use_feature_engineering: Whether to create additional features
            tune_hyperparams: Whether to tune hyperparameters
            n_iter: Number of iterations for hyperparameter tuning
        """
        self.model_name = model_name
        self.use_smote = use_smote
        self.use_feature_engineering = use_feature_engineering
        self.tune_hyperparams = tune_hyperparams
        self.n_iter = n_iter

        self.pipeline = None
        self.metrics = {}
        self.feature_columns = None

    def load_and_validate_data(self):
        """Load and validate raw data."""
        logger.info("=" * 60)
        logger.info("STEP 1: Loading and Validating Data")
        logger.info("=" * 60)

        # Load data
        logger.info(f"Loading data from {RAW_DATA_PATH}")
        df = load_data()
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Validate data
        logger.info("Validating data...")
        is_valid, report = validate_data(df, is_training=True)

        if not is_valid:
            raise ValueError(f"Data validation failed: {report['errors']}")

        logger.info(f"Data validation passed with {len(report['warnings'])} warnings")

        return df

    def preprocess_data(self, df):
        """Clean and preprocess data."""
        logger.info("=" * 60)
        logger.info("STEP 2: Data Preprocessing")
        logger.info("=" * 60)

        # Clean data
        logger.info("Cleaning data...")
        df = clean_data(df)

        # Feature engineering
        if self.use_feature_engineering:
            logger.info("Creating engineered features...")
            df = create_features(df)
            self.feature_columns = get_feature_columns(include_engineered=True)
            logger.info(f"Created {len(self.feature_columns['numerical']) + len(self.feature_columns['categorical'])} total features")
        else:
            self.feature_columns = get_feature_columns(include_engineered=False)

        return df

    def prepare_data(self, df):
        """Prepare features and split data."""
        logger.info("=" * 60)
        logger.info("STEP 3: Data Splitting")
        logger.info("=" * 60)

        # Prepare features
        X = df.drop('Churn', axis=1)
        y = df['Churn'].map({'Yes': 1, 'No': 0})

        # Keep only the columns we need
        all_features = self.feature_columns['numerical'] + self.feature_columns['categorical']
        available_features = [col for col in all_features if col in X.columns]
        X = X[available_features]

        # Update feature columns to only include available ones
        self.feature_columns['numerical'] = [c for c in self.feature_columns['numerical'] if c in available_features]
        self.feature_columns['categorical'] = [c for c in self.feature_columns['categorical'] if c in available_features]

        logger.info(f"Using {len(self.feature_columns['numerical'])} numerical features")
        logger.info(f"Using {len(self.feature_columns['categorical'])} categorical features")

        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

        return X_train, X_test, y_train, y_test

    def create_pipeline(self, model):
        """Create preprocessing and model pipeline."""
        # Preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.feature_columns['numerical']),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'),
                 self.feature_columns['categorical'])
            ]
        )

        # Pipeline with or without SMOTE
        if self.use_smote:
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('classifier', model)
            ])
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

        return pipeline

    def train(self, X_train, y_train):
        """Train the model with optional hyperparameter tuning."""
        logger.info("=" * 60)
        logger.info("STEP 4: Model Training")
        logger.info("=" * 60)

        # Get base model
        model, _ = get_model_and_params(self.model_name)
        logger.info(f"Training {self.model_name} model")
        logger.info(f"SMOTE: {self.use_smote}")

        # Create pipeline
        pipeline = self.create_pipeline(model)

        if self.tune_hyperparams:
            logger.info(f"Hyperparameter tuning with {self.n_iter} iterations...")
            self.pipeline, best_params, best_score = tune_hyperparameters(
                pipeline, X_train, y_train,
                model_name=self.model_name,
                n_iter=self.n_iter,
                cv=5,
                scoring='f1'
            )
            self.metrics['cv_f1_score'] = best_score
            self.metrics['best_params'] = best_params
        else:
            # Cross-validation without tuning
            logger.info("Performing cross-validation...")
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
            self.metrics['cv_f1_score'] = cv_scores.mean()
            self.metrics['cv_f1_std'] = cv_scores.std()
            logger.info(f"CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            # Train final model
            logger.info("Training final model...")
            pipeline.fit(X_train, y_train)
            self.pipeline = pipeline

        return self.pipeline

    def evaluate(self, X_test, y_test):
        """Evaluate the trained model."""
        logger.info("=" * 60)
        logger.info("STEP 5: Model Evaluation")
        logger.info("=" * 60)

        evaluator = ModelEvaluator()

        # Get predictions
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]

        # Get feature names after preprocessing
        feature_names = self._get_feature_names()

        # Get the classifier from pipeline
        classifier = self.pipeline.named_steps['classifier']

        # Generate comprehensive report
        report = evaluator.generate_report(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            model=classifier,
            feature_names=feature_names,
            model_name=self.model_name
        )

        self.metrics.update(report['metrics'])
        self.metrics['optimal_threshold'] = report.get('optimal_threshold', 0.5)

        return report

    def _get_feature_names(self):
        """Get feature names after preprocessing."""
        preprocessor = self.pipeline.named_steps['preprocessor']

        # Numerical feature names
        num_features = self.feature_columns['numerical']

        # Categorical feature names (after one-hot encoding)
        cat_encoder = preprocessor.named_transformers_['cat']
        if hasattr(cat_encoder, 'get_feature_names_out'):
            cat_features = cat_encoder.get_feature_names_out(
                self.feature_columns['categorical']
            ).tolist()
        else:
            cat_features = []

        return num_features + cat_features

    def save_model(self, path=None):
        """Save the trained model and metadata."""
        logger.info("=" * 60)
        logger.info("STEP 6: Saving Model")
        logger.info("=" * 60)

        if path is None:
            path = MODEL_PATH

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.pipeline, path)
        logger.info(f"Model saved to {path}")

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'training_date': datetime.now().isoformat(),
            'use_smote': self.use_smote,
            'use_feature_engineering': self.use_feature_engineering,
            'feature_columns': self.feature_columns,
            'metrics': {k: v for k, v in self.metrics.items()
                       if not isinstance(v, (np.ndarray, dict)) or k == 'best_params'}
        }

        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {metadata_path}")

        return path

    def run(self):
        """Run the full training pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING CHURN PREDICTION MODEL TRAINING")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Step 1: Load and validate
        df = self.load_and_validate_data()

        # Step 2: Preprocess
        df = self.preprocess_data(df)

        # Step 3: Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        # Step 4: Train
        self.train(X_train, y_train)

        # Step 5: Evaluate
        self.evaluate(X_test, y_test)

        # Step 6: Save
        self.save_model()

        # Summary
        elapsed = datetime.now() - start_time
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Test F1 Score: {self.metrics.get('f1', 'N/A'):.4f}")
        logger.info(f"Test ROC-AUC: {self.metrics.get('roc_auc', 'N/A'):.4f}")
        logger.info(f"Total time: {elapsed}")

        return self.pipeline, self.metrics


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description='Train churn prediction model')
    parser.add_argument('--model', type=str, default='GradientBoosting',
                       choices=['GradientBoosting', 'XGBoost', 'LightGBM', 'RandomForest'],
                       help='Model to train')
    parser.add_argument('--no-smote', action='store_true',
                       help='Disable SMOTE')
    parser.add_argument('--no-feature-engineering', action='store_true',
                       help='Disable feature engineering')
    parser.add_argument('--no-tuning', action='store_true',
                       help='Disable hyperparameter tuning')
    parser.add_argument('--n-iter', type=int, default=50,
                       help='Number of tuning iterations')

    args = parser.parse_args()

    trainer = ChurnModelTrainer(
        model_name=args.model,
        use_smote=not args.no_smote,
        use_feature_engineering=not args.no_feature_engineering,
        tune_hyperparams=not args.no_tuning,
        n_iter=args.n_iter
    )

    trainer.run()


if __name__ == "__main__":
    main()
