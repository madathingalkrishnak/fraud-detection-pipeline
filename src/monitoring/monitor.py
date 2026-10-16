"""
Monitoring and Alerting for the Fraud Detection Pipeline.

Three monitoring layers:
  1. Pipeline Health — throughput, latency, error rates
  2. Data Quality  — schema validation, null rates, value distributions
  3. Model Drift   — PSI (Population Stability Index), feature drift

In production, metrics are pushed to Azure Monitor and visualized
in Grafana or Azure Dashboards with alerts via PagerDuty/OpsGenie.
"""

import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data Quality Checks
# ------------------------------------------------------------------
@dataclass
class DataQualityReport:
    """Results of data quality checks on a batch of transactions."""
    batch_id: str
    timestamp: str
    total_rows: int
    null_rates: dict          # column -> fraction of nulls
    schema_valid: bool
    anomalies: list[str]      # human-readable issues found
    passed: bool


class DataQualityMonitor:
    """
    Validates incoming transaction data for quality issues.

    Catches problems like:
      - Missing required fields
      - Negative amounts
      - Impossible timestamps (future dates, too old)
      - Unexpected categorical values
    """

    REQUIRED_COLUMNS = [
        "transaction_id", "timestamp", "card_id", "user_id",
        "merchant_id", "amount",
    ]

    VALID_ENTRY_MODES = {"chip", "swipe", "online", "contactless"}
    VALID_CARD_TYPES = {"credit", "debit"}

    def __init__(self, max_null_rate: float = 0.05):
        self.max_null_rate = max_null_rate

    def check(self, df: pd.DataFrame, batch_id: str = "unknown") -> DataQualityReport:
        anomalies = []

        # Schema check
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        schema_valid = len(missing_cols) == 0
        if not schema_valid:
            anomalies.append(f"Missing required columns: {missing_cols}")

        # Null rates
        null_rates = {}
        for col in df.columns:
            rate = df[col].isnull().mean()
            null_rates[col] = round(rate, 4)
            if col in self.REQUIRED_COLUMNS and rate > 0:
                anomalies.append(f"Required column '{col}' has {rate*100:.1f}% nulls")
            elif rate > self.max_null_rate:
                anomalies.append(f"Column '{col}' has high null rate: {rate*100:.1f}%")

        # Value range checks
        if "amount" in df.columns:
            neg_count = (df["amount"] < 0).sum()
            if neg_count > 0:
                anomalies.append(f"{neg_count} transactions have negative amounts")

            zero_count = (df["amount"] == 0).sum()
            if zero_count > len(df) * 0.01:
                anomalies.append(f"{zero_count} transactions have zero amount ({zero_count/len(df)*100:.1f}%)")

        # Categorical validation
        if "entry_mode" in df.columns:
            unknown = set(df["entry_mode"].dropna().unique()) - self.VALID_ENTRY_MODES
            if unknown:
                anomalies.append(f"Unknown entry_mode values: {unknown}")

        # Duplicate check
        if "transaction_id" in df.columns:
            dup_count = df["transaction_id"].duplicated().sum()
            if dup_count > 0:
                anomalies.append(f"{dup_count} duplicate transaction IDs")

        return DataQualityReport(
            batch_id=batch_id,
            timestamp=datetime.now().isoformat(),
            total_rows=len(df),
            null_rates=null_rates,
            schema_valid=schema_valid,
            anomalies=anomalies,
            passed=len(anomalies) == 0,
        )


# ------------------------------------------------------------------
# Model Drift Detection
# ------------------------------------------------------------------
class ModelDriftMonitor:
    """
    Detect model drift using Population Stability Index (PSI).

    PSI measures how much the distribution of predictions (or features)
    has shifted compared to a reference distribution. A PSI > 0.2
    indicates significant drift requiring model retraining.

    PSI interpretation:
      < 0.10 — No significant shift
      0.10 - 0.20 — Moderate shift, monitor closely
      > 0.20 — Significant shift, retrain model
    """

    def __init__(self, psi_threshold: float = 0.2, n_bins: int = 10):
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins
        self.reference_distributions: dict[str, np.ndarray] = {}

    def set_reference(self, feature_name: str, values: np.ndarray):
        """Set the reference distribution for a feature (from training data)."""
        hist, bin_edges = np.histogram(values, bins=self.n_bins)
        self.reference_distributions[feature_name] = {
            "hist": hist / hist.sum(),  # Normalize to proportions
            "bin_edges": bin_edges,
        }

    def compute_psi(self, feature_name: str, current_values: np.ndarray) -> float:
        """
        Compute Population Stability Index for a single feature.

        PSI = Σ (actual_% - expected_%) * ln(actual_% / expected_%)
        """
        if feature_name not in self.reference_distributions:
            raise ValueError(f"No reference set for feature: {feature_name}")

        ref = self.reference_distributions[feature_name]
        expected = ref["hist"]
        bin_edges = ref["bin_edges"]

        actual_hist, _ = np.histogram(current_values, bins=bin_edges)
        actual = actual_hist / max(actual_hist.sum(), 1)

        # Avoid division by zero and log(0)
        epsilon = 1e-6
        expected = np.clip(expected, epsilon, None)
        actual = np.clip(actual, epsilon, None)

        psi = np.sum((actual - expected) * np.log(actual / expected))
        return float(psi)

    def check_drift(self, feature_name: str, current_values: np.ndarray) -> dict:
        """Check if a feature has drifted beyond the threshold."""
        psi = self.compute_psi(feature_name, current_values)
        status = (
            "OK" if psi < 0.1
            else "WARNING" if psi < self.psi_threshold
            else "DRIFT_DETECTED"
        )
        return {
            "feature": feature_name,
            "psi": round(psi, 4),
            "threshold": self.psi_threshold,
            "status": status,
            "checked_at": datetime.now().isoformat(),
        }

    def check_all_features(self, current_df: pd.DataFrame) -> list[dict]:
        """Check drift for all registered features."""
        results = []
        for feature_name in self.reference_distributions:
            if feature_name in current_df.columns:
                values = current_df[feature_name].dropna().values
                if len(values) > 0:
                    results.append(self.check_drift(feature_name, values))
        return results


# ------------------------------------------------------------------
# Pipeline Metrics Collector
# ------------------------------------------------------------------
class PipelineMetrics:
    """
    Collects and reports pipeline performance metrics.

    Tracks:
      - Throughput (transactions processed per second)
      - Latency (processing time per batch)
      - Error rate
      - Fraud detection rate
    """

    def __init__(self):
        self.metrics_log: list[dict] = []

    def record_batch(
        self,
        batch_id: str,
        num_records: int,
        processing_time_ms: float,
        num_fraud_detected: int,
        num_errors: int = 0,
    ):
        metric = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "num_records": num_records,
            "processing_time_ms": round(processing_time_ms, 2),
            "throughput_per_sec": round(num_records / max(processing_time_ms / 1000, 0.001), 2),
            "fraud_detected": num_fraud_detected,
            "fraud_rate": round(num_fraud_detected / max(num_records, 1), 4),
            "num_errors": num_errors,
            "error_rate": round(num_errors / max(num_records, 1), 4),
        }
        self.metrics_log.append(metric)
        return metric

    def get_summary(self, last_n: int = 100) -> dict:
        """Get summary statistics over recent batches."""
        recent = self.metrics_log[-last_n:]
        if not recent:
            return {"status": "no_data"}

        return {
            "total_batches": len(recent),
            "total_records": sum(m["num_records"] for m in recent),
            "avg_throughput": round(np.mean([m["throughput_per_sec"] for m in recent]), 2),
            "avg_latency_ms": round(np.mean([m["processing_time_ms"] for m in recent]), 2),
            "p99_latency_ms": round(np.percentile([m["processing_time_ms"] for m in recent], 99), 2),
            "total_fraud_detected": sum(m["fraud_detected"] for m in recent),
            "avg_fraud_rate": round(np.mean([m["fraud_rate"] for m in recent]), 4),
            "total_errors": sum(m["num_errors"] for m in recent),
            "period_start": recent[0]["timestamp"],
            "period_end": recent[-1]["timestamp"],
        }

    def export_metrics(self, output_path: str = "data/metrics/pipeline_metrics.json"):
        """Export metrics log to JSON."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.metrics_log, f, indent=2)
