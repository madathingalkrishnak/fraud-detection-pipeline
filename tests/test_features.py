"""
Unit tests for the feature engineering module.

Run: pytest tests/test_features.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestSyntheticGenerator:
    """Tests for the synthetic data generator."""

    def test_generates_correct_count(self):
        from src.ingestion.synthetic_generator import TransactionGenerator
        gen = TransactionGenerator(num_users=10, seed=42)
        txns = gen.generate_batch(100)
        # May have slightly more due to card_testing generating multiple txns
        assert len(txns) >= 100

    def test_fraud_rate_approximate(self):
        from src.ingestion.synthetic_generator import TransactionGenerator
        gen = TransactionGenerator(num_users=50, fraud_rate=0.05, seed=42)
        txns = gen.generate_batch(10_000)
        fraud_count = sum(1 for t in txns if t["is_fraud"] == 1)
        fraud_rate = fraud_count / len(txns)
        # Allow ±3% tolerance due to card_testing generating bursts
        assert 0.02 < fraud_rate < 0.12

    def test_required_fields_present(self):
        from src.ingestion.synthetic_generator import TransactionGenerator
        gen = TransactionGenerator(num_users=5, seed=42)
        txns = gen.generate_batch(10)

        required = [
            "transaction_id", "timestamp", "card_id", "user_id",
            "merchant_id", "amount", "is_fraud"
        ]
        for txn in txns:
            for field in required:
                assert field in txn, f"Missing field: {field}"

    def test_amounts_are_positive(self):
        from src.ingestion.synthetic_generator import TransactionGenerator
        gen = TransactionGenerator(num_users=10, seed=42)
        txns = gen.generate_batch(1000)
        for txn in txns:
            assert txn["amount"] > 0

    def test_transactions_sorted_by_time(self):
        from src.ingestion.synthetic_generator import TransactionGenerator
        gen = TransactionGenerator(num_users=10, seed=42)
        txns = gen.generate_batch(100)
        timestamps = [t["timestamp"] for t in txns]
        assert timestamps == sorted(timestamps)


class TestDataQualityMonitor:
    """Tests for the data quality monitoring module."""

    def test_detects_missing_columns(self):
        from src.monitoring.monitor import DataQualityMonitor
        monitor = DataQualityMonitor()
        df = pd.DataFrame({"amount": [100]})  # Missing required fields
        report = monitor.check(df)
        assert not report.schema_valid
        assert not report.passed

    def test_detects_negative_amounts(self):
        from src.monitoring.monitor import DataQualityMonitor
        monitor = DataQualityMonitor()
        df = pd.DataFrame({
            "transaction_id": ["t1", "t2"],
            "timestamp": [datetime.now()] * 2,
            "card_id": ["c1", "c2"],
            "user_id": ["u1", "u2"],
            "merchant_id": ["m1", "m2"],
            "amount": [100, -50],
        })
        report = monitor.check(df)
        assert any("negative" in a.lower() for a in report.anomalies)

    def test_passes_clean_data(self):
        from src.monitoring.monitor import DataQualityMonitor
        monitor = DataQualityMonitor()
        df = pd.DataFrame({
            "transaction_id": [f"t{i}" for i in range(10)],
            "timestamp": [datetime.now()] * 10,
            "card_id": [f"c{i}" for i in range(10)],
            "user_id": [f"u{i}" for i in range(10)],
            "merchant_id": [f"m{i}" for i in range(10)],
            "amount": [float(i * 10 + 1) for i in range(10)],
            "entry_mode": ["chip"] * 10,
        })
        report = monitor.check(df)
        assert report.passed


class TestModelDriftMonitor:
    """Tests for the model drift detection."""

    def test_no_drift_same_distribution(self):
        from src.monitoring.monitor import ModelDriftMonitor
        monitor = ModelDriftMonitor()

        np.random.seed(42)
        ref = np.random.normal(100, 20, 10000)
        current = np.random.normal(100, 20, 10000)

        monitor.set_reference("amount", ref)
        result = monitor.check_drift("amount", current)
        assert result["status"] == "OK"

    def test_detects_drift(self):
        from src.monitoring.monitor import ModelDriftMonitor
        monitor = ModelDriftMonitor(psi_threshold=0.2)

        np.random.seed(42)
        ref = np.random.normal(100, 20, 10000)
        # Shifted distribution
        current = np.random.normal(200, 50, 10000)

        monitor.set_reference("amount", ref)
        result = monitor.check_drift("amount", current)
        assert result["status"] == "DRIFT_DETECTED"
