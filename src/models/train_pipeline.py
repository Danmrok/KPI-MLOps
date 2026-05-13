import os
import sys
import pickle
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.pipeline import (
    create_decision_tree_pipeline,
    create_random_forest_pipeline,
    create_gradient_boosting_pipeline,
    FEATURE_COLS, TARGET_COL
)

warnings.filterwarnings('ignore')

EXPERIMENT_NAME = "air_quality_co_classification"
DATA_PATH = "data/processed/air_quality_processed.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def load_data():
    """Завантаження оброблених даних."""
    df = pd.read_csv(DATA_PATH)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df[TARGET_COL].values
    print(f"📊 Дані завантажено: X={X.shape}, y={y.shape}")
    print(f"   Класи: {np.unique(y, return_counts=True)}")
    return X, y, available


def compute_metrics(y_true, y_pred):
    """Обчислення набору класифікаційних метрик."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }


def run_experiment(pipeline, params: dict, model_name: str, X_train, X_test, y_train, y_test):
    """Один MLflow run: тренування + логування + збереження."""
    with mlflow.start_run(run_name=f"{model_name}"):

        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", "Air Quality UCI")
        mlflow.set_tag("variant", "18")

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_params(params)

        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=CV_FOLDS, scoring='accuracy', n_jobs=-1
        )
        mlflow.log_metric("cv_accuracy_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_accuracy_std", float(cv_scores.std()))

        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        train_metrics = compute_metrics(y_train, train_pred)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)

        test_pred = pipeline.predict(X_test)
        test_metrics = compute_metrics(y_test, test_pred)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        mlflow.sklearn.log_model(pipeline, "pipeline")

        run_id = mlflow.active_run().info.run_id

        print(f"\n{'='*55}")
        print(f"  {model_name}")
        print(f"  Params: {params}")
        print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test F1: {test_metrics['f1_weighted']:.4f}")
        print(f"  Run ID: {run_id}")

        return run_id, test_metrics, pipeline


def train_all():
    """Запуск всіх 7 експериментів (3 DT + 2 RF + 2 GB)."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    # ── Decision Tree: 3 експерименти ─────────────────────────────
    dt_configs = [
        {'max_depth': 5,  'min_samples_split': 2,  'criterion': 'gini'},
        {'max_depth': 10, 'min_samples_split': 5,  'criterion': 'gini'},
        {'max_depth': 15, 'min_samples_split': 10, 'criterion': 'entropy'},
    ]
    for cfg in dt_configs:
        pipe = create_decision_tree_pipeline(**cfg, random_state=RANDOM_STATE)
        run_id, metrics, fitted = run_experiment(
            pipe, cfg, "DecisionTree", X_train, X_test, y_train, y_test
        )
        results.append({'model': 'DecisionTree', 'run_id': run_id,
                        'params': cfg, **metrics})

    # ── Random Forest: 2 експерименти ─────────────────────────────
    rf_configs = [
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 200, 'max_depth': 15},
    ]
    best_rf_pipe = None
    best_rf_acc = 0
    for cfg in rf_configs:
        pipe = create_random_forest_pipeline(**cfg, random_state=RANDOM_STATE)
        run_id, metrics, fitted = run_experiment(
            pipe, cfg, "RandomForest", X_train, X_test, y_train, y_test
        )
        results.append({'model': 'RandomForest', 'run_id': run_id,
                        'params': cfg, **metrics})
        if metrics['accuracy'] > best_rf_acc:
            best_rf_acc = metrics['accuracy']
            best_rf_pipe = fitted

    # ── Gradient Boosting: 2 експерименти ─────────────────────────
    gb_configs = [
        {'n_estimators': 100, 'learning_rate': 0.1,  'max_depth': 3},
        {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 5},
    ]
    for cfg in gb_configs:
        pipe = create_gradient_boosting_pipeline(**cfg, random_state=RANDOM_STATE)
        run_id, metrics, fitted = run_experiment(
            pipe, cfg, "GradientBoosting", X_train, X_test, y_train, y_test
        )
        results.append({'model': 'GradientBoosting', 'run_id': run_id,
                        'params': cfg, **metrics})

    print(f"\n{'='*55}")
    print("  ПІДСУМОК ВСІХ ЕКСПЕРИМЕНТІВ")
    print(f"{'='*55}")
    best = max(results, key=lambda r: r['accuracy'])
    for r in sorted(results, key=lambda r: r['accuracy'], reverse=True):
        marker = " ⭐" if r['run_id'] == best['run_id'] else ""
        print(f"  {r['model']:20s} acc={r['accuracy']:.4f}  f1={r['f1_weighted']:.4f}{marker}")

    # Зберігаємо найкращу модель
    print(f"\n🏆 Найкраща модель: {best['model']}  accuracy={best['accuracy']:.4f}")

    # Перетренуємо найкращий pipeline для збереження
    if best['model'] == 'DecisionTree':
        best_params = best['params']
        best_pipe = create_decision_tree_pipeline(**best_params, random_state=RANDOM_STATE)
    elif best['model'] == 'RandomForest':
        best_params = best['params']
        best_pipe = create_random_forest_pipeline(**best_params, random_state=RANDOM_STATE)
    else:
        best_params = best['params']
        best_pipe = create_gradient_boosting_pipeline(**best_params, random_state=RANDOM_STATE)

    best_pipe.fit(X_train, y_train)
    model_path = os.path.join(MODELS_DIR, 'pipeline.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipe, f)
    print(f"💾 Найкращу модель збережено: {model_path}")

    # Повний звіт по найкращій
    y_pred = best_pipe.predict(X_test)
    print("\nClassification Report (найкраща модель):")
    print(classification_report(y_test, y_pred,
          target_names=['Low CO', 'Medium CO', 'High CO']))

    return results, best


if __name__ == "__main__":
    train_all()
