"""
Spark Structured Streaming Pipeline for Real-Time Fraud Detection.

This is the core of the system — it reads from Azure Event Hubs, engineers
features in real-time, runs ML inference, and writes results to Delta Lake.

Architecture:
    Event Hubs → Spark Streaming → Feature Engineering → Model Inference → Delta Lake
                                                                       → Alert Sink

For Databricks deployment, use notebook 04_streaming_pipeline.ipynb which
wraps this logic with Databricks-specific configurations.

Usage (local development):
    python streaming_pipeline.py --config configs/pipeline_config.yaml --local
"""

import argparse
import json
import os
import sys
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType, DoubleType, TimestampType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.config import load_config, get_spark_config
from src.utils.schemas import RAW_TRANSACTION_SCHEMA


def create_spark_session(config: dict, local: bool = False) -> SparkSession:
    """
    Create a Spark session configured for the pipeline.

    In Databricks, spark is pre-configured — use `spark` directly in notebooks.
    Locally, we configure Delta Lake and Event Hubs connectors.
    """
    spark_conf = get_spark_config(config)

    builder = (
        SparkSession.builder
        .appName(config["pipeline"]["name"])
    )

    if local:
        builder = (
            builder
            .master("local[*]")
            .config("spark.jars.packages",
                    "io.delta:delta-spark_2.12:3.1.0,"
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
        )

    for key, value in spark_conf.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


# ------------------------------------------------------------------
# Source: Azure Event Hubs
# ------------------------------------------------------------------
def read_from_event_hubs(spark: SparkSession, config: dict) -> DataFrame:
    """
    Read streaming data from Azure Event Hubs.

    Event Hubs is compatible with the Kafka protocol, so we use
    Spark's Kafka connector with the Event Hubs endpoint.
    """
    eh_config = config["event_hubs"]
    conn_str = eh_config["connection_string"]
    namespace = eh_config["namespace"]
    name = eh_config["name"]

    # Event Hubs uses Kafka protocol on port 9093
    bootstrap_servers = f"{namespace}.servicebus.windows.net:9093"

    # SASL configuration for Event Hubs authentication
    sasl_jaas = (
        f'org.apache.kafka.common.security.plain.PlainLoginModule required '
        f'username="$ConnectionString" '
        f'password="{conn_str}";'
    )

    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", name)
        .option("kafka.security.protocol", "SASL_SSL")
        .option("kafka.sasl.mechanism", "PLAIN")
        .option("kafka.sasl.jaas.config", sasl_jaas)
        .option("kafka.request.timeout.ms", "60000")
        .option("kafka.session.timeout.ms", "30000")
        .option("maxOffsetsPerTrigger",
                str(eh_config.get("max_events_per_trigger", 1000)))
        .option("startingOffsets",
                eh_config.get("starting_position", "latest"))
        .load()
    )


# ------------------------------------------------------------------
# Source: Local file stream (for development)
# ------------------------------------------------------------------
def read_from_local(spark: SparkSession, config: dict) -> DataFrame:
    """
    Read streaming data from a local directory (for development).

    Drop JSON files into data/stream_input/ to simulate incoming events.
    """
    input_path = config.get("data", {}).get("stream_input_path", "data/stream_input/")

    return (
        spark.readStream
        .format("json")
        .schema(RAW_TRANSACTION_SCHEMA)
        .option("maxFilesPerTrigger", 1)
        .load(input_path)
    )


# ------------------------------------------------------------------
# Parse raw Event Hubs / Kafka messages
# ------------------------------------------------------------------
def parse_events(raw_df: DataFrame) -> DataFrame:
    """
    Parse the raw Kafka/Event Hubs messages into structured transaction data.

    Event Hubs wraps the payload in a Kafka message with key, value, topic, etc.
    The actual transaction JSON is in the 'value' field as bytes.
    """
    # Parse JSON from the Kafka value field
    parsed = (
        raw_df
        .select(
            F.from_json(
                F.col("value").cast("string"),
                RAW_TRANSACTION_SCHEMA
            ).alias("data"),
            F.col("timestamp").alias("event_time"),  # Kafka ingestion timestamp
            F.col("partition").alias("event_partition"),
            F.col("offset").alias("event_offset"),
        )
        .select("data.*", "event_time", "event_partition", "event_offset")
    )

    # Parse the timestamp string to actual timestamp type
    parsed = parsed.withColumn(
        "timestamp",
        F.coalesce(
            F.to_timestamp("timestamp", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"),
            F.to_timestamp("timestamp", "yyyy-MM-dd'T'HH:mm:ss"),
            F.to_timestamp("timestamp"),
        )
    )

    return parsed


# ------------------------------------------------------------------
# Real-time Feature Engineering (Streaming-compatible)
# ------------------------------------------------------------------
def add_streaming_features(df: DataFrame) -> DataFrame:
    """
    Add features that can be computed per-record in streaming mode.

    Note: Aggregation features (rolling windows) require stateful processing.
    In Databricks, use Delta Live Tables or the Databricks Feature Store
    for proper windowed aggregations. For this local version, we compute
    the per-record features and use pre-computed aggregations from the
    feature store.
    """
    return (
        df
        # Transaction features
        .withColumn("amount_log", F.log1p(F.col("amount")))

        # Time features
        .withColumn("hour_of_day", F.hour(F.col("timestamp")))
        .withColumn("day_of_week", F.dayofweek(F.col("timestamp")))
        .withColumn("day_of_month", F.dayofmonth(F.col("timestamp")))
        .withColumn("is_weekend",
                    F.when(F.col("day_of_week").isin(1, 7), True)
                     .otherwise(False))
        .withColumn("is_night",
                    F.when((F.col("hour_of_day") >= 23) | (F.col("hour_of_day") < 5), True)
                     .otherwise(False))

        # Amount binning
        .withColumn("amount_bin",
                    F.when(F.col("amount") < 10, "micro")
                     .when(F.col("amount") < 50, "small")
                     .when(F.col("amount") < 200, "medium")
                     .when(F.col("amount") < 1000, "large")
                     .otherwise("very_large"))

        # Processing metadata
        .withColumn("processed_at", F.current_timestamp())
        .withColumn("pipeline_version", F.lit("1.0.0"))
    )


# ------------------------------------------------------------------
# Stateful Aggregations via Watermarking
# ------------------------------------------------------------------
def add_windowed_aggregations(df: DataFrame, config: dict) -> DataFrame:
    """
    Compute windowed aggregation features using Structured Streaming watermarks.

    This handles late-arriving data and maintains state per card_id.
    The watermark tells Spark how long to keep state for late events.
    """
    watermark_delay = config.get("streaming", {}).get("watermark_delay", "30 seconds")

    # Apply watermark for handling late data
    df_watermarked = df.withWatermark("timestamp", watermark_delay)

    # 5-minute tumbling window aggregations per card
    windowed_stats = (
        df_watermarked
        .groupBy(
            F.col("card_id"),
            F.window(F.col("timestamp"), "5 minutes")
        )
        .agg(
            F.count("*").alias("txn_count_5min"),
            F.sum("amount").alias("sum_amount_5min"),
            F.avg("amount").alias("avg_amount_5min"),
            F.max("amount").alias("max_amount_5min"),
        )
    )

    return windowed_stats


# ------------------------------------------------------------------
# Model Inference (Placeholder — uses UDF wrapping MLflow model)
# ------------------------------------------------------------------
def apply_model_inference(df: DataFrame, model_uri: str = None) -> DataFrame:
    """
    Apply the trained fraud detection model to incoming transactions.

    In Databricks, use mlflow.pyfunc.spark_udf for efficient distributed inference.
    Locally, we use a simple UDF wrapper.
    """
    if model_uri:
        import mlflow
        # Load model as a Spark UDF for distributed inference
        predict_udf = mlflow.pyfunc.spark_udf(
            spark=df.sparkSession,
            model_uri=model_uri,
            result_type=DoubleType(),
        )

        feature_cols = [
            "amount", "amount_log", "hour_of_day", "day_of_week",
            "is_weekend", "is_night", "is_international",
        ]

        df = (
            df
            .withColumn("fraud_probability",
                        predict_udf(F.struct(*[F.col(c) for c in feature_cols])))
            .withColumn("is_fraud_predicted",
                        F.when(F.col("fraud_probability") > 0.5, 1).otherwise(0))
        )
    else:
        # Rule-based fallback for local development (no model loaded)
        df = (
            df
            .withColumn("fraud_probability",
                        F.when(
                            (F.col("amount") > 2000) &
                            (F.col("is_night") == True) &
                            (F.col("is_international") == True),
                            F.lit(0.85)
                        ).when(
                            (F.col("amount") > 5000),
                            F.lit(0.7)
                        ).when(
                            (F.col("is_night") == True) & (F.col("amount") > 500),
                            F.lit(0.4)
                        ).otherwise(F.lit(0.05)))
            .withColumn("is_fraud_predicted",
                        F.when(F.col("fraud_probability") > 0.5, 1).otherwise(0))
        )

    return df


# ------------------------------------------------------------------
# Sinks: Delta Lake + Alert stream
# ------------------------------------------------------------------
def write_to_delta(df: DataFrame, config: dict, table_name: str = "transactions_enriched"):
    """
    Write the enriched stream to a Delta Lake table.

    Delta Lake gives us:
      - ACID transactions on streaming writes
      - Time travel for auditing
      - Schema enforcement and evolution
      - Efficient reads for the dashboard layer
    """
    checkpoint = config["streaming"]["checkpoint_location"]
    output_path = config["data"]["processed_path"] + table_name

    return (
        df.writeStream
        .format("delta")
        .outputMode(config["streaming"].get("output_mode", "append"))
        .option("checkpointLocation", f"{checkpoint}/{table_name}")
        .option("mergeSchema", "true")  # Allow schema evolution
        .trigger(processingTime=config["streaming"]["trigger_interval"])
        .start(output_path)
    )


def write_alerts(df: DataFrame, config: dict):
    """
    Write high-confidence fraud predictions to a separate alert stream.

    This feeds into Azure Monitor / PagerDuty / Slack for real-time alerts.
    """
    alert_threshold = config.get("monitoring", {}).get(
        "alerts", {}
    ).get("fraud_rate_spike_threshold", 0.8)

    alerts_df = df.filter(F.col("fraud_probability") >= alert_threshold)
    checkpoint = config["streaming"]["checkpoint_location"]

    return (
        alerts_df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{checkpoint}/fraud_alerts")
        .trigger(processingTime=config["streaming"]["trigger_interval"])
        .start(config["data"]["processed_path"] + "fraud_alerts")
    )


# ------------------------------------------------------------------
# Console sink for local debugging
# ------------------------------------------------------------------
def write_to_console(df: DataFrame, config: dict):
    """Write to console for local development/debugging."""
    return (
        df.writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", False)
        .option("numRows", 20)
        .trigger(processingTime=config["streaming"]["trigger_interval"])
        .start()
    )


# ------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------
def run_pipeline(config: dict, local: bool = False, model_uri: str = None):
    """
    Execute the end-to-end streaming fraud detection pipeline.

    Args:
        config: Pipeline configuration dictionary.
        local: If True, read from local files instead of Event Hubs.
        model_uri: MLflow model URI (e.g., "models:/fraud_detection_model/Production").
    """
    spark = create_spark_session(config, local=local)
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("FRAUD DETECTION PIPELINE")
    print(f"Mode: {'LOCAL' if local else 'AZURE'}")
    print(f"Model: {model_uri or 'Rule-based fallback'}")
    print("=" * 60)

    # 1. Read from source
    if local:
        raw_stream = read_from_local(spark, config)
        enriched = add_streaming_features(raw_stream)
    else:
        raw_stream = read_from_event_hubs(spark, config)
        parsed = parse_events(raw_stream)
        enriched = add_streaming_features(parsed)

    # 2. Apply model inference
    predictions = apply_model_inference(enriched, model_uri)

    # 3. Write outputs
    if local:
        # Local: write to console for debugging
        query = write_to_console(predictions, config)
    else:
        # Azure: write to Delta Lake
        main_query = write_to_delta(predictions, config)
        alert_query = write_alerts(predictions, config)

        print("Pipeline running. Press Ctrl+C to stop.")
        main_query.awaitTermination()
        return

    print("Pipeline running. Press Ctrl+C to stop.")
    query.awaitTermination()


def main():
    parser = argparse.ArgumentParser(description="Run the fraud detection streaming pipeline")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--local", action="store_true", help="Run in local development mode")
    parser.add_argument("--model-uri", type=str, default=None,
                        help="MLflow model URI for inference")
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, local=args.local, model_uri=args.model_uri)


if __name__ == "__main__":
    main()
