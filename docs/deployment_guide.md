# Azure Deployment Guide

Step-by-step instructions for deploying the fraud detection pipeline on Azure + Databricks.

## Prerequisites

- Azure subscription (Free tier with $200 credit works for initial setup)
- Azure CLI installed (`az login` authenticated)
- Databricks workspace (or create one below)

## Step 1: Create Azure Resources

```bash
# Set variables
RESOURCE_GROUP="fraud-detection-rg"
LOCATION="eastus"
STORAGE_ACCOUNT="frauddetectionadls"   # Must be globally unique
EVENTHUB_NAMESPACE="fraud-eventhubs"    # Must be globally unique
DATABRICKS_WORKSPACE="fraud-databricks"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Data Lake Storage Gen2
az storage account create \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard_LRS \
    --kind StorageV2 \
    --hns true   # Hierarchical namespace = ADLS Gen2

# Create storage containers
az storage container create --name raw --account-name $STORAGE_ACCOUNT
az storage container create --name processed --account-name $STORAGE_ACCOUNT
az storage container create --name checkpoints --account-name $STORAGE_ACCOUNT

# Create Event Hubs namespace
az eventhubs namespace create \
    --name $EVENTHUB_NAMESPACE \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard \
    --capacity 1

# Create the transactions Event Hub
az eventhubs eventhub create \
    --name transactions \
    --namespace-name $EVENTHUB_NAMESPACE \
    --resource-group $RESOURCE_GROUP \
    --message-retention 7 \
    --partition-count 4

# Get the connection string
az eventhubs namespace authorization-rule keys list \
    --resource-group $RESOURCE_GROUP \
    --namespace-name $EVENTHUB_NAMESPACE \
    --name RootManageSharedAccessKey \
    --query primaryConnectionString \
    --output tsv
```

## Step 2: Create Databricks Workspace

```bash
# Create Databricks workspace
az databricks workspace create \
    --name $DATABRICKS_WORKSPACE \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku standard

# Get the workspace URL (open in browser to complete setup)
az databricks workspace show \
    --name $DATABRICKS_WORKSPACE \
    --resource-group $RESOURCE_GROUP \
    --query workspaceUrl \
    --output tsv
```

## Step 3: Configure Databricks

1. **Open the Databricks workspace URL** in your browser.

2. **Create a cluster** with these settings:
   - Runtime: Databricks Runtime 14.3 LTS ML
   - Node type: Standard_DS3_v2 (for cost efficiency)
   - Workers: 1-2 (autoscaling)
   - Install these libraries:
     - PyPI: `xgboost`, `azure-eventhub`, `great-expectations`
     - Maven: `com.microsoft.azure:azure-eventhubs-spark_2.12:2.3.22`

3. **Configure storage access** (in cluster Spark config):
   ```
   fs.azure.account.key.<STORAGE_ACCOUNT>.dfs.core.windows.net <YOUR_ACCESS_KEY>
   ```

4. **Store secrets** using Databricks Secrets:
   ```bash
   # Install Databricks CLI
   pip install databricks-cli
   databricks configure --token

   # Create a secret scope
   databricks secrets create-scope --scope fraud-detection

   # Store the Event Hubs connection string
   databricks secrets put --scope fraud-detection --key eventhub-connection-string

   # Store the storage account key
   databricks secrets put --scope fraud-detection --key storage-account-key
   ```

## Step 4: Upload Project to Databricks

```bash
# Upload notebooks
databricks workspace import_dir notebooks/ /fraud-detection/notebooks

# Upload source code as a wheel or sync via Git
# Option A: Git integration (recommended)
# In Databricks UI: Repos > Add Repo > paste your GitHub URL

# Option B: Upload as files
databricks workspace import_dir src/ /fraud-detection/src
```

## Step 5: Run the Pipeline

### A. Generate and ingest data

```bash
# From your local machine, send synthetic data to Event Hubs
export EVENTHUB_CONNECTION_STRING="<your-connection-string>"
python src/ingestion/event_hubs_producer.py --mode synthetic --rate 50 --duration 600
```

### B. Start the streaming pipeline

Open `notebooks/04_streaming_pipeline.ipynb` in Databricks and run all cells. The key Databricks-specific configuration:

```python
# Read secrets
eh_conn_str = dbutils.secrets.get("fraud-detection", "eventhub-connection-string")

# Read from Event Hubs
raw_stream = (
    spark.readStream
    .format("eventhubs")
    .options(**{
        "eventhubs.connectionString":
            sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(eh_conn_str),
        "eventhubs.consumerGroup": "$Default",
        "maxEventsPerTrigger": "1000",
    })
    .load()
)
```

### C. Create the dashboard

1. In Databricks, go to **SQL > SQL Editor**.
2. Create queries from `notebooks/05_dashboard_queries.sql`.
3. Create a dashboard and add each query as a visualization.
4. Set the dashboard to **auto-refresh every 30 seconds**.

## Step 6: Set Up Monitoring

### Azure Monitor alerts

```bash
# Create an action group for email alerts
az monitor action-group create \
    --name fraud-alerts \
    --resource-group $RESOURCE_GROUP \
    --short-name FraudAlert \
    --action email admin your-email@example.com

# Create an alert rule for high fraud rate
# (This monitors a custom metric you push from the pipeline)
az monitor metrics alert create \
    --name high-fraud-rate \
    --resource-group $RESOURCE_GROUP \
    --scopes "/subscriptions/<sub-id>/resourceGroups/$RESOURCE_GROUP" \
    --condition "avg FraudRate > 0.10" \
    --window-size 5m \
    --evaluation-frequency 1m \
    --action fraud-alerts \
    --description "Fraud rate exceeded 10% threshold"
```

## Cost Estimates (Development)

| Resource | SKU | Monthly Cost |
|----------|-----|-------------|
| ADLS Gen2 | Pay-per-use | ~$2 |
| Event Hubs | Standard, 1 TU | ~$22 |
| Databricks | Standard, DS3_v2 | ~$50 (if running 4hr/day) |
| **Total** | | **~$74/month** |

**Cost-saving tips:**
- Stop the Databricks cluster when not in use (set auto-termination to 30 min).
- Use Event Hubs Basic tier for development ($11/mo).
- Run the pipeline in bursts rather than 24/7 during development.
- Record a video demo so you can tear down resources afterward.

## Teardown

```bash
# Delete everything when done
az group delete --name $RESOURCE_GROUP --yes --no-wait
```
