import logging
from typing import Dict, Any, Tuple
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)

PARAM_GRIDS = {
    'GradientBoosting': {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__subsample': [0.8, 0.9, 1.0]
    },
    'RandomForest': {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [5, 10, 15, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__class_weight': ['balanced', 'balanced_subsample', None]
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__min_child_weight': [1, 3, 5],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        'classifier__reg_alpha': [0, 0.1, 1],
        'classifier__reg_lambda': [0, 0.1, 1],
        'classifier__scale_pos_weight': [1, 2, 3]
    },
    'LightGBM': {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [3, 5, 7, 10, -1],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__num_leaves': [20, 31, 50, 100],
        'classifier__min_child_samples': [10, 20, 30],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        'classifier__reg_alpha': [0, 0.1, 1],
        'classifier__reg_lambda': [0, 0.1, 1],
        'classifier__class_weight': ['balanced', None]
    }
}


def get_model_and_params(model_name: str) -> Tuple[Any, Dict]:
    models = {
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return models[model_name], PARAM_GRIDS[model_name]


def tune_hyperparameters(pipeline, X_train, y_train, model_name='GradientBoosting',
                         n_iter=50, cv=5, scoring='f1', n_jobs=-1) -> Tuple[Any, Dict, float]:
    logger.info(f"Tuning {model_name}: n_iter={n_iter}, cv={cv}, scoring={scoring}")

    _, param_grid = get_model_and_params(model_name)
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=42,
        verbose=1,
        return_train_score=True
    )

    search.fit(X_train, y_train)

    logger.info(f"Best {scoring}: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")

    train_score = search.cv_results_['mean_train_score'][search.best_index_]
    test_score = search.cv_results_['mean_test_score'][search.best_index_]

    if train_score - test_score > 0.1:
        logger.warning(f"Overfitting: train={train_score:.4f}, cv={test_score:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_


def compare_models(pipeline_func, X_train, y_train, models=None, n_iter=30, cv=5, scoring='f1'):
    if models is None:
        models = ['GradientBoosting', 'XGBoost', 'LightGBM', 'RandomForest']

    results = {}

    for model_name in models:
        logger.info(f"Tuning {model_name}...")

        try:
            model, _ = get_model_and_params(model_name)
            pipeline = pipeline_func(model)
            best_pipeline, best_params, best_score = tune_hyperparameters(
                pipeline, X_train, y_train, model_name=model_name, n_iter=n_iter, cv=cv, scoring=scoring
            )
            results[model_name] = {'pipeline': best_pipeline, 'params': best_params, 'score': best_score}
        except Exception as e:
            logger.error(f"Error tuning {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    valid_results = {k: v for k, v in results.items() if 'score' in v}
    if valid_results:
        best = max(valid_results.keys(), key=lambda k: valid_results[k]['score'])
        logger.info(f"Best model: {best} ({scoring}={valid_results[best]['score']:.4f})")

    return results
