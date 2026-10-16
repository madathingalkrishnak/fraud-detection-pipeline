"""
Local Development Pipeline — run the full flow without Azure.

This script demonstrates the complete pipeline locally:
  1. Generate synthetic transactions
  2. Engineer features (batch mode)
  3. Train a model with MLflow
  4. Score predictions
  5. Save results to local Parquet / Delta

Use this to validate logic before deploying to Databricks + Azure.

Usage:
    python src/processing/local_pipeline.py
"""

import os
import sys
import json
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def run_local_pipeline():
    print("=" * 60)
    print("FRAUD DETECTION PIPELINE — LOCAL MODE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # ------------------------------------------------------------------
    print("\n[1/5] Generating synthetic transaction data...")
    from src.ingestion.synthetic_generator import TransactionGenerator

    generator = TransactionGenerator(num_users=500, fraud_rate=0.035, seed=42)
    transactions = generator.generate_batch(num_transactions=50_000)
    df = pd.DataFrame(transactions)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/transactions.csv", index=False)
    df.to_parquet("data/raw/transactions.parquet", index=False)

    fraud_count = df["is_fraud"].sum()
    print(f"  Generated {len(df):,} transactions")
    print(f"  Fraud: {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Saved to data/raw/transactions.parquet")

    # ------------------------------------------------------------------
    # Step 2: Feature Engineering (using PySpark batch mode)
    # ------------------------------------------------------------------
    print("\n[2/5] Engineering features with PySpark...")

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("FraudDetection-Local")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Read into Spark
    sdf = spark.read.parquet("data/raw/transactions.parquet")

    # Parse timestamps
    sdf = sdf.withColumn(
        "timestamp",
        __import__("pyspark.sql.functions", fromlist=["to_timestamp"])
        .to_timestamp("timestamp")
    )

    from pyspark.sql import functions as F
    sdf = sdf.withColumn("timestamp", F.to_timestamp("timestamp"))

    # Apply feature engineering
    from src.features.feature_engineering import FraudFeatureEngineer
    engineer = FraudFeatureEngineer()
    featured_sdf = engineer.transform_batch(sdf)

    # Save features
    os.makedirs("data/processed", exist_ok=True)
    featured_sdf.write.mode("overwrite").parquet("data/processed/features.parquet")

    feature_count = len(featured_sdf.columns)
    print(f"  Engineered {feature_count} columns")
    print(f"  Saved to data/processed/features.parquet")

    spark.stop()

    # ------------------------------------------------------------------
    # Step 3: Train model
    # ------------------------------------------------------------------
    print("\n[3/5] Training XGBoost model with MLflow tracking...")

    from src.model.train import load_and_split_data, train_model
    from src.utils.config import load_config

    try:
        config = load_config("configs/pipeline_config.yaml")
    except FileNotFoundError:
        config = {"model": {"hyperparameters": {}, "experiment_name": "fraud_detection_local"}}

    X_train, y_train, X_test, y_test, feature_names, test_df = \
        load_and_split_data("data/processed/features.parquet")

    model, metrics, threshold = train_model(
        X_train, y_train, X_test, y_test, feature_names, config
    )

    # ------------------------------------------------------------------
    # Step 4: Score full dataset
    # ------------------------------------------------------------------
    print("\n[4/5] Scoring all transactions...")

    features_df = pd.read_parquet("data/processed/features.parquet")
    available_features = [c for c in feature_names if c in features_df.columns]
    X_all = features_df[available_features].values
    probabilities = model.predict_proba(X_all)[:, 1]

    features_df["fraud_probability"] = probabilities
    features_df["is_fraud_predicted"] = (probabilities >= threshold).astype(int)

    # Save predictions
    features_df.to_parquet("data/processed/predictions.parquet", index=False)

    predicted_fraud = features_df["is_fraud_predicted"].sum()
    actual_fraud = features_df["is_fraud"].sum()
    print(f"  Predicted fraud: {predicted_fraud:,}")
    print(f"  Actual fraud:    {actual_fraud:,}")

    # ------------------------------------------------------------------
    # Step 5: Generate summary stats for dashboard
    # ------------------------------------------------------------------
    print("\n[5/5] Computing dashboard metrics...")

    # Hourly fraud rate
    features_df["hour"] = pd.to_datetime(features_df["timestamp"]).dt.hour
    hourly = features_df.groupby("hour").agg(
        total=("is_fraud", "count"),
        fraud=("is_fraud", "sum"),
        predicted=("is_fraud_predicted", "sum"),
        avg_prob=("fraud_probability", "mean"),
    ).reset_index()
    hourly["fraud_rate"] = hourly["fraud"] / hourly["total"]

    os.makedirs("data/dashboard", exist_ok=True)
    hourly.to_csv("data/dashboard/hourly_stats.csv", index=False)

    # Category-level stats
    cat_stats = features_df.groupby("merchant_category").agg(
        total=("is_fraud", "count"),
        fraud=("is_fraud", "sum"),
        avg_amount=("amount", "mean"),
    ).reset_index()
    cat_stats["fraud_rate"] = cat_stats["fraud"] / cat_stats["total"]
    cat_stats.to_csv("data/dashboard/category_stats.csv", index=False)

    # Summary
    summary = {
        "total_transactions": int(len(features_df)),
        "total_fraud": int(actual_fraud),
        "fraud_rate": float(round(actual_fraud / len(features_df), 4)),
        "model_precision": float(round(metrics["precision"], 4)),
        "model_recall": float(round(metrics["recall"], 4)),
        "model_f1": float(round(metrics["f1_score"], 4)),
        "model_roc_auc": float(round(metrics["roc_auc"], 4)),
        "optimal_threshold": float(round(threshold, 4)),
        "generated_at": datetime.now().isoformat(),
    }
    with open("data/dashboard/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Dashboard data saved to data/dashboard/")

    print("\n" + "=" * 60)
    print("LOCAL PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Review MLflow experiments: mlflow ui (then open http://localhost:5000)")
    print(f"  2. Explore predictions: data/processed/predictions.parquet")
    print(f"  3. Deploy to Azure: see docs/deployment_guide.md")


if __name__ == "__main__":
    run_local_pipeline()
