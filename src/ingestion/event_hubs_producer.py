"""
Azure Event Hubs Producer — sends transaction data to Event Hubs for downstream consumption.

In production, this would be replaced by actual payment gateways, POS systems,
and banking APIs. For this project, we replay synthetic or historical data.

Setup:
    1. Create an Event Hubs namespace in Azure Portal.
    2. Create an Event Hub named "transactions".
    3. Create a Shared Access Policy with "Send" permission.
    4. Set EVENTHUB_CONNECTION_STRING in your environment.

Usage:
    # Send synthetic data
    python event_hubs_producer.py --mode synthetic --rate 50 --duration 600

    # Replay a CSV file
    python event_hubs_producer.py --mode replay --input data/raw/transactions.csv --rate 100
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

from azure.eventhub.aio import EventHubProducerClient
from azure.eventhub import EventData
from azure.eventhub.exceptions import EventHubError

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.ingestion.synthetic_generator import TransactionGenerator


class EventHubsProducer:
    """
    Sends transaction events to Azure Event Hubs.

    Supports batching for throughput and uses partition keys for ordered
    delivery per card_id (important for fraud detection — we need
    transactions from the same card processed in order).
    """

    def __init__(self, connection_string: str, eventhub_name: str = "transactions"):
        self.producer = EventHubProducerClient.from_connection_string(
            conn_str=connection_string,
            eventhub_name=eventhub_name,
        )
        self.total_sent = 0
        self.total_errors = 0

    async def send_batch(self, transactions: list[dict], partition_key_field: str = "card_id"):
        """
        Send a batch of transactions to Event Hubs.

        Uses card_id as partition key so all transactions for the same card
        go to the same partition, preserving ordering.
        """
        # Group by partition key for optimal batching
        partitions: dict[str, list[dict]] = {}
        for txn in transactions:
            key = txn.get(partition_key_field, "default")
            partitions.setdefault(key, []).append(txn)

        for partition_key, txns in partitions.items():
            try:
                event_batch = await self.producer.create_batch(partition_key=partition_key)
                for txn in txns:
                    event_data = EventData(json.dumps(txn))
                    # Add metadata as application properties
                    event_data.properties = {
                        "source": "synthetic_generator",
                        "schema_version": "1.0",
                    }
                    try:
                        event_batch.add(event_data)
                    except ValueError:
                        # Batch is full, send it and start a new one
                        await self.producer.send_batch(event_batch)
                        self.total_sent += event_batch.size_in_bytes
                        event_batch = await self.producer.create_batch(partition_key=partition_key)
                        event_batch.add(event_data)

                await self.producer.send_batch(event_batch)
                self.total_sent += len(txns)

            except EventHubError as e:
                self.total_errors += len(txns)
                print(f"Error sending batch: {e}", file=sys.stderr)

    async def stream_synthetic(self, rate: int = 50, duration: int = 300):
        """
        Generate and send synthetic transactions at the specified rate.

        Args:
            rate: Target transactions per second.
            duration: How long to stream in seconds.
        """
        generator = TransactionGenerator(num_users=500, fraud_rate=0.035)
        batch_size = max(1, rate // 10)  # Send in small batches for efficiency
        interval = batch_size / rate

        start_time = time.time()
        batch = []

        print(f"Streaming synthetic data at ~{rate} txn/sec for {duration}s...")

        for txn in generator.stream(rate=rate * 2):  # Over-generate to fill batches
            batch.append(txn)

            if len(batch) >= batch_size:
                await self.send_batch(batch)
                batch = []

                elapsed = time.time() - start_time
                if elapsed > duration:
                    break

                # Status update every 10 seconds
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    throughput = self.total_sent / elapsed
                    print(f"  [{elapsed:.0f}s] Sent: {self.total_sent}, "
                          f"Errors: {self.total_errors}, "
                          f"Throughput: {throughput:.0f} txn/sec")

                await asyncio.sleep(interval)

        # Send remaining
        if batch:
            await self.send_batch(batch)

        elapsed = time.time() - start_time
        print(f"\nDone. Sent {self.total_sent} transactions in {elapsed:.1f}s "
              f"({self.total_sent/elapsed:.0f} txn/sec)")

    async def replay_file(self, file_path: str, rate: int = 100):
        """
        Replay transactions from a CSV or JSON file.

        Args:
            file_path: Path to the transaction data file.
            rate: Target transactions per second.
        """
        import pandas as pd

        print(f"Loading transactions from {file_path}...")
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path, lines=True)
        else:
            df = pd.read_parquet(file_path)

        # Update timestamps to current time
        df = df.sort_values("timestamp")
        transactions = df.to_dict(orient="records")

        print(f"Replaying {len(transactions)} transactions at ~{rate} txn/sec...")

        batch_size = max(1, rate // 5)
        interval = batch_size / rate

        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            # Update timestamps to now
            for txn in batch:
                txn["timestamp"] = datetime.now().isoformat()
            await self.send_batch(batch)
            await asyncio.sleep(interval)

            if (i // batch_size) % 20 == 0:
                print(f"  Progress: {i}/{len(transactions)} "
                      f"({i/len(transactions)*100:.1f}%)")

        print(f"Replay complete. Sent {len(transactions)} transactions.")

    async def close(self):
        await self.producer.close()


async def main():
    parser = argparse.ArgumentParser(description="Send transactions to Event Hubs")
    parser.add_argument("--mode", choices=["synthetic", "replay"], default="synthetic")
    parser.add_argument("--rate", type=int, default=50, help="Transactions per second")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds (synthetic mode)")
    parser.add_argument("--input", type=str, help="Input file path (replay mode)")
    args = parser.parse_args()

    connection_string = os.environ.get("EVENTHUB_CONNECTION_STRING")
    if not connection_string:
        print("Error: Set EVENTHUB_CONNECTION_STRING environment variable.", file=sys.stderr)
        print("Get it from Azure Portal > Event Hubs > Shared Access Policies.", file=sys.stderr)
        sys.exit(1)

    producer = EventHubsProducer(connection_string)

    try:
        if args.mode == "synthetic":
            await producer.stream_synthetic(rate=args.rate, duration=args.duration)
        elif args.mode == "replay":
            if not args.input:
                print("Error: --input required for replay mode", file=sys.stderr)
                sys.exit(1)
            await producer.replay_file(args.input, rate=args.rate)
    finally:
        await producer.close()


if __name__ == "__main__":
    asyncio.run(main())
