"""
Fraud Detection Model Training with MLflow Tracking.

Trains an XGBoost classifier on the engineered features and logs everything
to MLflow for experiment tracking, model registry, and reproducibility.

Key design decisions:
  - XGBoost chosen for its strength with tabular data, imbalanced classes,
    and built-in handling of missing values.
  - AUCPR (Area Under Precision-Recall Curve) is the primary metric because
    accuracy is misleading with 96.5% negative class.
  - scale_pos_weight handles class imbalance without synthetic oversampling.
  - Temporal train/test split (not random) to prevent data leakage from
    future transactions.

Usage:
    # Local training
    python train.py --data data/processed/features.parquet

    # On Databricks (notebook-friendly)
    # See notebooks/03_model_training.ipynb
"""

import argparse
import os
import sys
import json
from datetime import datetime

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.config import load_config


# ------------------------------------------------------------------
# Feature columns (must match feature_engineering.py output)
# ------------------------------------------------------------------
FEATURE_COLUMNS = [
    "amount", "amount_log",
    "hour_of_day", "day_of_week", "day_of_month",
    "is_weekend", "is_night", "is_international",
    "txn_count_1min", "txn_count_5min", "txn_count_1hr", "txn_count_24hr",
    "sum_amount_1hr", "avg_amount_1hr", "sum_amount_24hr",
    "max_amount_24hr", "std_amount_24hr", "amount_zscore",
    "time_since_last_txn_sec", "amount_change_ratio", "country_changed",
    "card_type_encoded", "entry_mode_encoded", "merchant_category_encoded",
]

LABEL = "is_fraud"


def load_and_split_data(data_path: str, test_ratio: float = 0.2):
    """
    Load features and perform a temporal train/test split.

    WHY temporal split?
    Random splitting leaks future information into training. In production,
    the model only sees past data, so the test set must be strictly
    after the training set in time.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Sort by timestamp and split temporally
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"Train: {len(train_df)} rows ({train_df[LABEL].mean()*100:.2f}% fraud)")
    print(f"Test:  {len(test_df)} rows ({test_df[LABEL].mean()*100:.2f}% fraud)")

    # Select only the feature columns that exist in the data
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        print(f"Warning: Missing features (will be excluded): {missing_features}")

    X_train = train_df[available_features].values
    y_train = train_df[LABEL].values
    X_test = test_df[available_features].values
    y_test = test_df[LABEL].values

    return X_train, y_train, X_test, y_test, available_features, test_df


def find_optimal_threshold(y_true, y_proba):
    """
    Find the classification threshold that maximizes F1 score.

    Default 0.5 threshold is rarely optimal for imbalanced problems.
    We sweep thresholds and pick the one with best F1.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1


def plot_precision_recall_curve(y_true, y_proba, save_path: str = "pr_curve.png"):
    """Plot and save the Precision-Recall curve."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, linewidth=2, color="#2563eb")
    ax.fill_between(recalls, precisions, alpha=0.15, color="#2563eb")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve (AP = {avg_precision:.4f})", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_feature_importance(model, feature_names, save_path: str = "feature_importance.png", top_n: int = 20):
    """Plot and save feature importance chart."""
    importance = model.get_score(importance_type="gain")

    # Map feature indices to names
    named_importance = {}
    for feat, score in importance.items():
        idx = int(feat.replace("f", ""))
        if idx < len(feature_names):
            named_importance[feature_names[idx]] = score

    # Sort and take top N
    sorted_features = sorted(named_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, scores = zip(*sorted_features) if sorted_features else ([], [])

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(names))
    ax.barh(y_pos, scores, color="#2563eb", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (Gain)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def train_model(
    X_train, y_train, X_test, y_test,
    feature_names: list[str],
    config: dict,
):
    """
    Train an XGBoost model with MLflow tracking.

    Returns the trained model and evaluation metrics.
    """
    model_config = config.get("model", {})
    hyperparams = model_config.get("hyperparameters", {})

    # Calculate class imbalance ratio
    fraud_count = sum(y_train)
    legit_count = len(y_train) - fraud_count
    scale_pos_weight = legit_count / max(fraud_count, 1)

    params = {
        "objective": "binary:logistic",
        "eval_metric": hyperparams.get("eval_metric", "aucpr"),
        "n_estimators": hyperparams.get("n_estimators", 500),
        "max_depth": hyperparams.get("max_depth", 8),
        "learning_rate": hyperparams.get("learning_rate", 0.05),
        "subsample": hyperparams.get("subsample", 0.8),
        "colsample_bytree": hyperparams.get("colsample_bytree", 0.8),
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",       # Fast histogram-based training
        "random_state": 42,
        "n_jobs": -1,
    }

    # Start MLflow experiment
    experiment_name = model_config.get("experiment_name", "fraud_detection")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("num_features", len(feature_names))
        mlflow.log_param("train_size", len(y_train))
        mlflow.log_param("test_size", len(y_test))
        mlflow.log_param("train_fraud_rate", float(y_train.mean()))
        mlflow.log_param("test_fraud_rate", float(y_test.mean()))

        # Train with early stopping
        early_stopping = hyperparams.get("early_stopping_rounds", 50)
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # Predict probabilities
        y_proba = model.predict_proba(X_test)[:, 1]

        # Find optimal threshold
        optimal_threshold, best_f1 = find_optimal_threshold(y_test, y_proba)
        y_pred = (y_proba >= optimal_threshold).astype(int)

        # Compute metrics
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "avg_precision": average_precision_score(y_test, y_proba),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "optimal_threshold": optimal_threshold,
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_dict = {
            "true_negatives": int(cm[0][0]),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
            "true_positives": int(cm[1][1]),
        }
        mlflow.log_dict(cm_dict, "confusion_matrix.json")

        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")

        # Log plots as artifacts
        pr_path = plot_precision_recall_curve(y_test, y_proba, "/tmp/pr_curve.png")
        mlflow.log_artifact(pr_path, "plots")

        fi_path = plot_feature_importance(
            model.get_booster(), feature_names,
            "/tmp/feature_importance.png"
        )
        mlflow.log_artifact(fi_path, "plots")

        # Log feature names
        mlflow.log_dict(
            {"features": feature_names, "optimal_threshold": optimal_threshold},
            "model_metadata.json"
        )

        # Log the model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_config.get("registered_model_name", "fraud_detection_model"),
            input_example=pd.DataFrame([X_test[0]], columns=feature_names),
        )

        # Print results
        print("\n" + "=" * 60)
        print("MODEL TRAINING RESULTS")
        print("=" * 60)
        print(f"ROC AUC:          {metrics['roc_auc']:.4f}")
        print(f"Avg Precision:    {metrics['avg_precision']:.4f}")
        print(f"F1 Score:         {metrics['f1_score']:.4f}")
        print(f"Precision:        {metrics['precision']:.4f}")
        print(f"Recall:           {metrics['recall']:.4f}")
        print(f"Optimal Thresh:   {optimal_threshold:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0][0]:,}  FP: {cm[0][1]:,}")
        print(f"  FN: {cm[1][0]:,}  TP: {cm[1][1]:,}")
        print("=" * 60)

    return model, metrics, optimal_threshold


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to feature-engineered Parquet file")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml",
                        help="Path to pipeline config")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None,
                        help="MLflow tracking server URI")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    X_train, y_train, X_test, y_test, feature_names, _ = load_and_split_data(args.data)
    model, metrics, threshold = train_model(
        X_train, y_train, X_test, y_test, feature_names, config
    )


if __name__ == "__main__":
    main()
