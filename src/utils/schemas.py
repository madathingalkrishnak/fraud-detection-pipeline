"""
Shared schema definitions for the fraud detection pipeline.

Centralizing schemas ensures consistency across ingestion, processing,
feature engineering, and inference stages.
"""

from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType, BooleanType, LongType
)


# ------------------------------------------------------------------
# Raw transaction schema — what arrives from Event Hubs / the generator
# ------------------------------------------------------------------
RAW_TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id", StringType(), nullable=False),
    StructField("timestamp", TimestampType(), nullable=False),
    StructField("card_id", StringType(), nullable=False),
    StructField("user_id", StringType(), nullable=False),
    StructField("merchant_id", StringType(), nullable=False),
    StructField("merchant_category", StringType(), nullable=True),
    StructField("amount", DoubleType(), nullable=False),
    StructField("currency", StringType(), nullable=True),
    StructField("card_type", StringType(), nullable=True),       # credit | debit
    StructField("entry_mode", StringType(), nullable=True),      # chip | swipe | online | contactless
    StructField("country", StringType(), nullable=True),
    StructField("city", StringType(), nullable=True),
    StructField("latitude", DoubleType(), nullable=True),
    StructField("longitude", DoubleType(), nullable=True),
    StructField("is_international", BooleanType(), nullable=True),
    StructField("device_id", StringType(), nullable=True),
    StructField("ip_address", StringType(), nullable=True),
    StructField("is_fraud", IntegerType(), nullable=True),       # 1=fraud, 0=legit (label)
])

# JSON schema string for reading from Event Hubs / Kafka
RAW_TRANSACTION_JSON_SCHEMA = """
{
    "type": "struct",
    "fields": [
        {"name": "transaction_id", "type": "string", "nullable": false},
        {"name": "timestamp", "type": "string", "nullable": false},
        {"name": "card_id", "type": "string", "nullable": false},
        {"name": "user_id", "type": "string", "nullable": false},
        {"name": "merchant_id", "type": "string", "nullable": false},
        {"name": "merchant_category", "type": "string", "nullable": true},
        {"name": "amount", "type": "double", "nullable": false},
        {"name": "currency", "type": "string", "nullable": true},
        {"name": "card_type", "type": "string", "nullable": true},
        {"name": "entry_mode", "type": "string", "nullable": true},
        {"name": "country", "type": "string", "nullable": true},
        {"name": "city", "type": "string", "nullable": true},
        {"name": "latitude", "type": "double", "nullable": true},
        {"name": "longitude", "type": "double", "nullable": true},
        {"name": "is_international", "type": "boolean", "nullable": true},
        {"name": "device_id", "type": "string", "nullable": true},
        {"name": "ip_address", "type": "string", "nullable": true},
        {"name": "is_fraud", "type": "integer", "nullable": true}
    ]
}
"""


# ------------------------------------------------------------------
# Enriched transaction schema — after feature engineering
# ------------------------------------------------------------------
ENRICHED_TRANSACTION_SCHEMA = StructType([
    # Original fields
    StructField("transaction_id", StringType(), nullable=False),
    StructField("timestamp", TimestampType(), nullable=False),
    StructField("card_id", StringType(), nullable=False),
    StructField("user_id", StringType(), nullable=False),
    StructField("merchant_id", StringType(), nullable=False),
    StructField("merchant_category", StringType(), nullable=True),
    StructField("amount", DoubleType(), nullable=False),
    StructField("entry_mode", StringType(), nullable=True),
    StructField("country", StringType(), nullable=True),

    # Time-based features
    StructField("hour_of_day", IntegerType(), nullable=True),
    StructField("day_of_week", IntegerType(), nullable=True),
    StructField("is_weekend", BooleanType(), nullable=True),
    StructField("is_night", BooleanType(), nullable=True),

    # Amount features
    StructField("amount_log", DoubleType(), nullable=True),
    StructField("amount_zscore", DoubleType(), nullable=True),

    # Aggregation features (rolling windows)
    StructField("txn_count_1min", LongType(), nullable=True),
    StructField("txn_count_5min", LongType(), nullable=True),
    StructField("txn_count_1hr", LongType(), nullable=True),
    StructField("txn_count_24hr", LongType(), nullable=True),
    StructField("sum_amount_1hr", DoubleType(), nullable=True),
    StructField("avg_amount_1hr", DoubleType(), nullable=True),
    StructField("max_amount_24hr", DoubleType(), nullable=True),

    # Velocity features
    StructField("time_since_last_txn_sec", DoubleType(), nullable=True),
    StructField("amount_change_ratio", DoubleType(), nullable=True),

    # Prediction
    StructField("fraud_probability", DoubleType(), nullable=True),
    StructField("is_fraud_predicted", IntegerType(), nullable=True),
    StructField("is_fraud", IntegerType(), nullable=True),
])


# ------------------------------------------------------------------
# Feature list for model training
# ------------------------------------------------------------------
FEATURE_COLUMNS = [
    "amount",
    "amount_log",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "is_international",
    "txn_count_1min",
    "txn_count_5min",
    "txn_count_1hr",
    "txn_count_24hr",
    "sum_amount_1hr",
    "avg_amount_1hr",
    "max_amount_24hr",
    "time_since_last_txn_sec",
    "amount_change_ratio",
    # Encoded categoricals
    "card_type_encoded",
    "entry_mode_encoded",
    "merchant_category_encoded",
]

LABEL_COLUMN = "is_fraud"
