"""
Configuration loader with environment variable substitution.
"""

import os
import re
import yaml
from pathlib import Path


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    pattern = re.compile(r'\$\{([^}]+)\}')
    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))
    return pattern.sub(replacer, value)


def _walk_and_resolve(obj):
    """Recursively resolve env vars in a nested dict/list."""
    if isinstance(obj, dict):
        return {k: _walk_and_resolve(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_walk_and_resolve(item) for item in obj]
    elif isinstance(obj, str):
        return _resolve_env_vars(obj)
    return obj


def load_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    """
    Load YAML config and resolve environment variable placeholders.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with resolved configuration values.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    return _walk_and_resolve(raw_config)


def get_spark_config(config: dict) -> dict:
    """Extract Spark-specific configuration for session builder."""
    env = config["pipeline"]["environment"]

    spark_conf = {
        "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
        "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        "spark.sql.streaming.schemaInference": "true",
        "spark.sql.adaptive.enabled": "true",
    }

    if env == "production":
        spark_conf.update({
            "spark.sql.shuffle.partitions": "200",
            "spark.streaming.backpressure.enabled": "true",
            "spark.sql.streaming.metricsEnabled": "true",
        })
    else:
        spark_conf.update({
            "spark.sql.shuffle.partitions": "8",
            "spark.driver.memory": "4g",
        })

    return spark_conf
