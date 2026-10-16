# Real-Time Fraud Detection Pipeline

A production-grade real-time fraud detection system built with **Databricks**, **Azure Event Hubs**, **Delta Lake**, and **MLflow**. This pipeline ingests streaming financial transactions, engineers features on the fly, runs ML inference, and surfaces fraud alerts through a live dashboard.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│  Transaction     │────▶│  Azure Event     │────▶│  Databricks Spark       │
│  Sources /       │     │  Hubs            │     │  Structured Streaming   │
│  Synthetic Gen   │     │  (Ingestion)     │     │  (Processing + Features)│
└─────────────────┘     └──────────────────┘     └────────┬────────────────┘
                                                          │
                         ┌────────────────────────────────┤
                         │                                │
                         ▼                                ▼
              ┌─────────────────┐              ┌─────────────────────┐
              │  MLflow Model   │              │  Azure Data Lake    │
              │  (Real-time     │              │  Storage Gen2       │
              │   Inference)    │              │  (Delta Lake)       │
              └────────┬────────┘              └─────────┬───────────┘
                       │                                 │
                       ▼                                 │
              ┌─────────────────┐                        │
              │  Azure Monitor  │                        │
              │  (Alerts)       │                        │
              └─────────────────┘                        │
                                                         ▼
                                              ┌─────────────────────┐
                                              │  Databricks SQL /   │
                                              │  Power BI Dashboard │
                                              └─────────────────────┘
```

## Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Ingestion | Azure Event Hubs | High-throughput streaming ingestion |
| Stream Processing | Spark Structured Streaming | Real-time data transformation |
| Feature Store | Databricks Feature Store | Centralized feature management |
| ML Training | XGBoost + MLflow | Model training with experiment tracking |
| Model Serving | Azure ML / Databricks Model Serving | Low-latency real-time inference |
| Data Storage | Delta Lake on ADLS Gen2 | ACID-compliant data lake |
| Monitoring | Azure Monitor + custom metrics | Pipeline health + model drift |
| Dashboard | Databricks SQL Analytics | Real-time fraud metrics |

## Dataset

This project uses the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset, augmented with a synthetic data generator that simulates realistic transaction streams.

## Quick Start

### Prerequisites
- Python 3.10+
- Azure subscription (Free tier works)
- Databricks workspace (Community Edition for development)
- Kaggle account (for dataset download)

### Local Development
```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/fraud-detection-pipeline.git
cd fraud-detection-pipeline
pip install -r requirements.txt

# Download dataset
python scripts/download_data.py

# Run data exploration
jupyter notebook notebooks/01_data_exploration.ipynb

# Test the synthetic data generator
python src/ingestion/synthetic_generator.py --num-transactions 1000

# Run the local pipeline (no Azure required)
python src/processing/local_pipeline.py
```

### Azure Deployment
See [docs/deployment_guide.md](docs/deployment_guide.md) for full Azure setup instructions.

## Project Structure
```
fraud-detection-pipeline/
├── configs/                    # Configuration files
│   ├── pipeline_config.yaml    # Pipeline parameters
│   ├── event_hubs_config.yaml  # Event Hubs connection settings
│   └── model_config.yaml       # ML model hyperparameters
├── data/                       # Local data (gitignored)
├── docs/                       # Documentation
│   ├── architecture.md         # Detailed architecture doc
│   └── deployment_guide.md     # Azure deployment instructions
├── notebooks/                  # Databricks / Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_streaming_pipeline.ipynb
│   └── 05_dashboard_queries.sql
├── scripts/                    # Utility scripts
│   ├── download_data.py
│   ├── setup_azure.sh
│   └── deploy_model.py
├── src/                        # Source code
│   ├── ingestion/              # Data ingestion
│   ├── processing/             # Stream processing
│   ├── features/               # Feature engineering
│   ├── model/                  # ML model training
│   ├── inference/              # Real-time inference
│   ├── monitoring/             # Alerting & monitoring
│   ├── dashboard/              # Dashboard definitions
│   └── utils/                  # Shared utilities
├── tests/                      # Unit and integration tests
├── requirements.txt
├── setup.py
└── README.md
```

## License
MIT