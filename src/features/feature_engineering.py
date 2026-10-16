"""
Feature Engineering for Fraud Detection.

Two modes:
  1. Batch mode   — for model training on historical data (Spark DataFrame)
  2. Stream mode  — for real-time inference (Spark Structured Streaming)

Feature categories:
  - Transaction-level: raw + derived from the single transaction
  - Time-based: hour, day of week, weekend/night flags
  - Aggregation: rolling window stats per card (count, sum, avg, max)
  - Velocity: time since last txn, amount change ratio
  - Categorical encoding: merchant category, entry mode, card type
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, BooleanType


class FraudFeatureEngineer:
    """
    Builds features for fraud detection from raw transaction data.

    Design decisions:
      - All features are deterministic and reproducible.
      - Window aggregations use card_id as the partition key.
      - No data leakage: features use only past transactions relative to each row.
      - Features work in both batch (Window functions) and streaming
        (stateful aggregations) modes.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # Transaction-Level Features
    # ------------------------------------------------------------------
    def add_transaction_features(self, df: DataFrame) -> DataFrame:
        """Derive features from individual transaction fields."""
        return (
            df
            # Log-transform amount (reduces skew from high-value transactions)
            .withColumn("amount_log", F.log1p(F.col("amount")))

            # Boolean flags
            .withColumn("is_international",
                        F.when(F.col("is_international").isNull(), False)
                         .otherwise(F.col("is_international")))

            # Amount binning (useful for tree-based models)
            .withColumn("amount_bin",
                        F.when(F.col("amount") < 10, "micro")
                         .when(F.col("amount") < 50, "small")
                         .when(F.col("amount") < 200, "medium")
                         .when(F.col("amount") < 1000, "large")
                         .otherwise("very_large"))
        )

    # ------------------------------------------------------------------
    # Time-Based Features
    # ------------------------------------------------------------------
    def add_time_features(self, df: DataFrame) -> DataFrame:
        """Extract temporal patterns from the transaction timestamp."""
        return (
            df
            .withColumn("hour_of_day", F.hour(F.col("timestamp")))
            .withColumn("day_of_week", F.dayofweek(F.col("timestamp")))
            .withColumn("day_of_month", F.dayofmonth(F.col("timestamp")))
            .withColumn("is_weekend",
                        F.when(F.col("day_of_week").isin(1, 7), True)
                         .otherwise(False))
            # Night = 11 PM to 5 AM — higher fraud risk window
            .withColumn("is_night",
                        F.when((F.col("hour_of_day") >= 23) | (F.col("hour_of_day") < 5), True)
                         .otherwise(False))
        )

    # ------------------------------------------------------------------
    # Aggregation Features (Batch Mode — uses Window functions)
    # ------------------------------------------------------------------
    def add_aggregation_features_batch(self, df: DataFrame) -> DataFrame:
        """
        Compute rolling window aggregation features using Spark Window functions.

        For batch training only. In streaming mode, these are computed via
        stateful structured streaming (see streaming_features.py).

        Windows: 1 min, 5 min, 1 hour, 24 hours
        Partitioned by: card_id
        Ordered by: timestamp
        """
        # Convert timestamp to epoch seconds for range-based windows
        df = df.withColumn("ts_epoch", F.col("timestamp").cast("long"))

        # Define windows: (name_suffix, range_in_seconds)
        windows = [
            ("1min", 60),
            ("5min", 300),
            ("1hr", 3600),
            ("24hr", 86400),
        ]

        for suffix, range_sec in windows:
            w = (
                Window
                .partitionBy("card_id")
                .orderBy("ts_epoch")
                .rangeBetween(-range_sec, 0)
            )

            # Transaction count in window
            df = df.withColumn(
                f"txn_count_{suffix}",
                F.count("transaction_id").over(w)
            )

            # Sum of amounts in window
            df = df.withColumn(
                f"sum_amount_{suffix}",
                F.sum("amount").over(w)
            )

            # Average amount in window
            df = df.withColumn(
                f"avg_amount_{suffix}",
                F.avg("amount").over(w)
            )

        # Max amount in 24-hour window
        w_24hr = (
            Window
            .partitionBy("card_id")
            .orderBy("ts_epoch")
            .rangeBetween(-86400, 0)
        )
        df = df.withColumn("max_amount_24hr", F.max("amount").over(w_24hr))
        df = df.withColumn("std_amount_24hr", F.stddev("amount").over(w_24hr))

        # Z-score of current amount relative to 24hr window
        df = df.withColumn(
            "amount_zscore",
            F.when(F.col("std_amount_24hr") > 0,
                   (F.col("amount") - F.col("avg_amount_24hr")) / F.col("std_amount_24hr"))
             .otherwise(0.0)
        )

        return df.drop("ts_epoch")

    # ------------------------------------------------------------------
    # Velocity Features
    # ------------------------------------------------------------------
    def add_velocity_features(self, df: DataFrame) -> DataFrame:
        """
        Compute velocity-based features that capture spending acceleration.

        These are strong fraud signals: fraudsters tend to act quickly
        once they have card details, producing transactions faster
        than the cardholder's normal pace.
        """
        # Window: previous transaction for same card
        w_prev = (
            Window
            .partitionBy("card_id")
            .orderBy("timestamp")
        )

        df = df.withColumn("prev_timestamp", F.lag("timestamp", 1).over(w_prev))
        df = df.withColumn("prev_amount", F.lag("amount", 1).over(w_prev))
        df = df.withColumn("prev_country", F.lag("country", 1).over(w_prev))

        # Time since last transaction (in seconds)
        df = df.withColumn(
            "time_since_last_txn_sec",
            F.when(
                F.col("prev_timestamp").isNotNull(),
                F.col("timestamp").cast("long") - F.col("prev_timestamp").cast("long")
            ).otherwise(None)
        )

        # Amount change ratio
        df = df.withColumn(
            "amount_change_ratio",
            F.when(
                (F.col("prev_amount").isNotNull()) & (F.col("prev_amount") > 0),
                F.col("amount") / F.col("prev_amount")
            ).otherwise(1.0)
        )

        # Country changed since last transaction (geographic anomaly)
        df = df.withColumn(
            "country_changed",
            F.when(
                F.col("prev_country").isNotNull(),
                (F.col("country") != F.col("prev_country")).cast("int")
            ).otherwise(0)
        )

        return df.drop("prev_timestamp", "prev_amount", "prev_country")

    # ------------------------------------------------------------------
    # Categorical Encoding
    # ------------------------------------------------------------------
    def add_categorical_encodings(self, df: DataFrame) -> DataFrame:
        """
        Encode categorical features for ML models.

        Uses frequency encoding (more robust than one-hot for high cardinality)
        and target encoding for merchant_category.
        """
        from pyspark.ml.feature import StringIndexer

        categorical_cols = ["card_type", "entry_mode", "merchant_category"]

        for col_name in categorical_cols:
            indexer = StringIndexer(
                inputCol=col_name,
                outputCol=f"{col_name}_encoded",
                handleInvalid="keep"  # Unseen categories get a default index
            )
            df = indexer.fit(df).transform(df)

        return df

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------
    def transform_batch(self, df: DataFrame) -> DataFrame:
        """
        Apply all feature engineering steps for batch/training mode.

        Args:
            df: Raw transaction DataFrame with schema matching RAW_TRANSACTION_SCHEMA.

        Returns:
            Enriched DataFrame ready for model training.
        """
        df = self.add_transaction_features(df)
        df = self.add_time_features(df)
        df = self.add_aggregation_features_batch(df)
        df = self.add_velocity_features(df)
        df = self.add_categorical_encodings(df)

        # Fill nulls for numeric features (first transaction per card has no history)
        numeric_fill = {
            "txn_count_1min": 1,
            "txn_count_5min": 1,
            "txn_count_1hr": 1,
            "txn_count_24hr": 1,
            "sum_amount_1hr": 0.0,
            "avg_amount_1hr": 0.0,
            "max_amount_24hr": 0.0,
            "std_amount_24hr": 0.0,
            "amount_zscore": 0.0,
            "time_since_last_txn_sec": 86400.0,  # Default to 24 hours
            "amount_change_ratio": 1.0,
            "country_changed": 0,
        }
        df = df.fillna(numeric_fill)

        return df
