"""
Synthetic Transaction Generator for Fraud Detection Pipeline.

Generates realistic financial transaction data with configurable fraud patterns.
This serves two purposes:
  1. Local development — test the pipeline without Azure Event Hubs.
  2. Demo / replay — simulate a live stream by writing to Event Hubs or a local socket.

Fraud patterns modeled:
  - Card-not-present (CNP) fraud: online transactions with unusual amounts
  - Card testing: rapid low-value transactions to verify stolen card details
  - Account takeover: sudden change in spending geography or merchant category
  - Bust-out fraud: gradual increase in spending before maxing out

Usage:
    # Generate a batch CSV
    python synthetic_generator.py --num-transactions 50000 --output data/raw/transactions.csv

    # Stream to stdout (pipe into Event Hubs producer or local pipeline)
    python synthetic_generator.py --stream --rate 100 --duration 300
"""

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Generator

import numpy as np


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail", "electronics",
    "travel", "entertainment", "healthcare", "education", "utilities",
    "clothing", "jewelry", "atm_withdrawal", "money_transfer", "gambling",
]

COUNTRIES = ["US", "US", "US", "US", "CA", "GB", "DE", "FR", "BR", "IN", "NG", "RU", "CN"]
CITIES_BY_COUNTRY = {
    "US": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"],
    "CA": ["Toronto", "Vancouver", "Montreal"],
    "GB": ["London", "Manchester", "Birmingham"],
    "DE": ["Berlin", "Munich", "Hamburg"],
    "FR": ["Paris", "Lyon", "Marseille"],
    "BR": ["São Paulo", "Rio de Janeiro"],
    "IN": ["Mumbai", "Delhi", "Bangalore"],
    "NG": ["Lagos", "Abuja"],
    "RU": ["Moscow", "Saint Petersburg"],
    "CN": ["Beijing", "Shanghai"],
}

ENTRY_MODES = ["chip", "swipe", "online", "contactless"]
CARD_TYPES = ["credit", "debit"]


@dataclass
class Transaction:
    transaction_id: str
    timestamp: str
    card_id: str
    user_id: str
    merchant_id: str
    merchant_category: str
    amount: float
    currency: str
    card_type: str
    entry_mode: str
    country: str
    city: str
    latitude: float
    longitude: float
    is_international: bool
    device_id: str
    ip_address: str
    is_fraud: int


class UserProfile:
    """Simulates a cardholder with consistent spending behavior."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.card_id = f"card_{uuid.uuid4().hex[:12]}"
        self.card_type = random.choice(CARD_TYPES)
        self.home_country = "US"
        self.home_city = random.choice(CITIES_BY_COUNTRY["US"])
        self.device_id = f"dev_{uuid.uuid4().hex[:8]}"

        # Spending profile
        self.avg_amount = np.random.lognormal(mean=3.5, sigma=0.8)  # ~$33 median
        self.amount_std = self.avg_amount * random.uniform(0.3, 0.8)
        self.preferred_categories = random.sample(MERCHANT_CATEGORIES, k=random.randint(3, 6))
        self.preferred_entry_modes = random.sample(ENTRY_MODES, k=random.randint(1, 3))

        # Activity profile
        self.avg_txns_per_day = random.uniform(1, 8)
        self.last_txn_time = None
        self.last_country = self.home_country


class FraudPatternGenerator:
    """Generates different fraud patterns on top of legitimate user behavior."""

    @staticmethod
    def card_not_present_fraud(user: UserProfile, timestamp: datetime) -> Transaction:
        """Online fraud with unusual amount, often from a different country."""
        fraud_country = random.choice(["NG", "RU", "CN", "BR"])
        fraud_city = random.choice(CITIES_BY_COUNTRY[fraud_country])
        amount = user.avg_amount * random.uniform(3, 15)  # Much higher than normal

        return Transaction(
            transaction_id=f"txn_{uuid.uuid4().hex[:16]}",
            timestamp=timestamp.isoformat(),
            card_id=user.card_id,
            user_id=user.user_id,
            merchant_id=f"merch_{uuid.uuid4().hex[:8]}",
            merchant_category=random.choice(["electronics", "jewelry", "online_retail", "money_transfer"]),
            amount=round(amount, 2),
            currency="USD",
            card_type=user.card_type,
            entry_mode="online",
            country=fraud_country,
            city=fraud_city,
            latitude=round(random.uniform(-30, 60), 6),
            longitude=round(random.uniform(-120, 120), 6),
            is_international=True,
            device_id=f"dev_{uuid.uuid4().hex[:8]}",  # Different device
            ip_address=f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            is_fraud=1,
        )

    @staticmethod
    def card_testing_fraud(user: UserProfile, timestamp: datetime) -> list[Transaction]:
        """Rapid sequence of small transactions to verify stolen card details."""
        txns = []
        for i in range(random.randint(3, 8)):
            t = timestamp + timedelta(seconds=random.randint(5, 60) * i)
            txns.append(Transaction(
                transaction_id=f"txn_{uuid.uuid4().hex[:16]}",
                timestamp=t.isoformat(),
                card_id=user.card_id,
                user_id=user.user_id,
                merchant_id=f"merch_{uuid.uuid4().hex[:8]}",
                merchant_category="online_retail",
                amount=round(random.uniform(0.50, 5.00), 2),  # Very small amounts
                currency="USD",
                card_type=user.card_type,
                entry_mode="online",
                country="US",
                city=random.choice(CITIES_BY_COUNTRY["US"]),
                latitude=round(random.uniform(25, 48), 6),
                longitude=round(random.uniform(-125, -70), 6),
                is_international=False,
                device_id=f"dev_{uuid.uuid4().hex[:8]}",
                ip_address=f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
                is_fraud=1,
            ))
        return txns

    @staticmethod
    def bust_out_fraud(user: UserProfile, timestamp: datetime) -> Transaction:
        """Large transaction attempting to max out credit before disappearing."""
        amount = random.uniform(2000, 10000)
        return Transaction(
            transaction_id=f"txn_{uuid.uuid4().hex[:16]}",
            timestamp=timestamp.isoformat(),
            card_id=user.card_id,
            user_id=user.user_id,
            merchant_id=f"merch_{uuid.uuid4().hex[:8]}",
            merchant_category=random.choice(["jewelry", "electronics", "money_transfer", "gambling"]),
            amount=round(amount, 2),
            currency="USD",
            card_type="credit",
            entry_mode=random.choice(["chip", "swipe"]),
            country=user.home_country,
            city=user.home_city,
            latitude=round(random.uniform(25, 48), 6),
            longitude=round(random.uniform(-125, -70), 6),
            is_international=False,
            device_id=user.device_id,
            ip_address=f"192.168.{random.randint(0,255)}.{random.randint(0,255)}",
            is_fraud=1,
        )


class TransactionGenerator:
    """
    Main generator that produces a realistic mix of legitimate and fraudulent transactions.

    Args:
        num_users: Number of unique cardholders to simulate.
        fraud_rate: Proportion of transactions that are fraudulent (default 3.5%).
        seed: Random seed for reproducibility.
    """

    def __init__(self, num_users: int = 1000, fraud_rate: float = 0.035, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        self.fraud_rate = fraud_rate
        self.fraud_gen = FraudPatternGenerator()
        self.users = [
            UserProfile(user_id=f"user_{i:05d}")
            for i in range(num_users)
        ]

    def _generate_legitimate_txn(self, user: UserProfile, timestamp: datetime) -> Transaction:
        """Generate a single legitimate transaction for a user."""
        country = user.home_country if random.random() > 0.05 else random.choice(["CA", "GB", "DE", "FR"])
        city = random.choice(CITIES_BY_COUNTRY.get(country, ["Unknown"]))
        is_international = country != user.home_country

        amount = max(0.01, np.random.normal(user.avg_amount, user.amount_std))

        return Transaction(
            transaction_id=f"txn_{uuid.uuid4().hex[:16]}",
            timestamp=timestamp.isoformat(),
            card_id=user.card_id,
            user_id=user.user_id,
            merchant_id=f"merch_{random.randint(1, 5000):05d}",
            merchant_category=random.choice(user.preferred_categories),
            amount=round(amount, 2),
            currency="USD",
            card_type=user.card_type,
            entry_mode=random.choice(user.preferred_entry_modes),
            country=country,
            city=city,
            latitude=round(random.uniform(25, 48), 6),
            longitude=round(random.uniform(-125, -70), 6),
            is_international=is_international,
            device_id=user.device_id,
            ip_address=f"192.168.{random.randint(0,255)}.{random.randint(1,255)}",
            is_fraud=0,
        )

    def generate_batch(
        self,
        num_transactions: int,
        start_time: datetime | None = None,
    ) -> list[dict]:
        """
        Generate a batch of transactions.

        Args:
            num_transactions: Number of transactions to generate.
            start_time: Starting timestamp (defaults to 30 days ago).

        Returns:
            List of transaction dictionaries sorted by timestamp.
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)

        transactions = []
        time_span_seconds = 30 * 24 * 3600  # 30 days in seconds

        for _ in range(num_transactions):
            user = random.choice(self.users)
            offset = random.uniform(0, time_span_seconds)
            txn_time = start_time + timedelta(seconds=offset)

            if random.random() < self.fraud_rate:
                # Generate a fraudulent transaction
                fraud_type = random.choices(
                    ["cnp", "card_testing", "bust_out"],
                    weights=[0.5, 0.3, 0.2],
                    k=1
                )[0]

                if fraud_type == "cnp":
                    txn = self.fraud_gen.card_not_present_fraud(user, txn_time)
                    transactions.append(asdict(txn))
                elif fraud_type == "card_testing":
                    for txn in self.fraud_gen.card_testing_fraud(user, txn_time):
                        transactions.append(asdict(txn))
                else:
                    txn = self.fraud_gen.bust_out_fraud(user, txn_time)
                    transactions.append(asdict(txn))
            else:
                txn = self._generate_legitimate_txn(user, txn_time)
                transactions.append(asdict(txn))

        # Sort by timestamp
        transactions.sort(key=lambda x: x["timestamp"])
        return transactions

    def stream(self, rate: int = 10) -> Generator[dict, None, None]:
        """
        Generate transactions as an infinite stream.

        Args:
            rate: Transactions per second.

        Yields:
            Transaction dictionaries one at a time.
        """
        interval = 1.0 / rate
        while True:
            user = random.choice(self.users)
            now = datetime.now()

            if random.random() < self.fraud_rate:
                fraud_type = random.choice(["cnp", "card_testing", "bust_out"])
                if fraud_type == "cnp":
                    txn = self.fraud_gen.card_not_present_fraud(user, now)
                    yield asdict(txn)
                elif fraud_type == "card_testing":
                    for txn in self.fraud_gen.card_testing_fraud(user, now):
                        yield asdict(txn)
                        time.sleep(interval / 3)
                else:
                    txn = self.fraud_gen.bust_out_fraud(user, now)
                    yield asdict(txn)
            else:
                txn = self._generate_legitimate_txn(user, now)
                yield asdict(txn)

            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data")
    parser.add_argument("--num-transactions", type=int, default=50000,
                        help="Number of transactions to generate (batch mode)")
    parser.add_argument("--num-users", type=int, default=1000,
                        help="Number of unique cardholders")
    parser.add_argument("--fraud-rate", type=float, default=0.035,
                        help="Proportion of fraudulent transactions")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (CSV or JSON). Prints to stdout if not set.")
    parser.add_argument("--stream", action="store_true",
                        help="Stream mode: emit transactions continuously")
    parser.add_argument("--rate", type=int, default=10,
                        help="Transactions per second (stream mode)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration in seconds (stream mode)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    generator = TransactionGenerator(
        num_users=args.num_users,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )

    if args.stream:
        # Streaming mode
        print(f"Streaming at {args.rate} txn/sec for {args.duration}s...", file=sys.stderr)
        start = time.time()
        count = 0
        for txn in generator.stream(rate=args.rate):
            print(json.dumps(txn))
            sys.stdout.flush()
            count += 1
            if time.time() - start > args.duration:
                break
        print(f"\nStreamed {count} transactions.", file=sys.stderr)

    else:
        # Batch mode
        print(f"Generating {args.num_transactions} transactions...", file=sys.stderr)
        transactions = generator.generate_batch(args.num_transactions)

        if args.output:
            import pandas as pd
            df = pd.DataFrame(transactions)
            if args.output.endswith(".csv"):
                df.to_csv(args.output, index=False)
            elif args.output.endswith(".json"):
                df.to_json(args.output, orient="records", lines=True)
            else:
                df.to_parquet(args.output, index=False)
            print(f"Saved {len(transactions)} transactions to {args.output}", file=sys.stderr)

            # Print summary stats
            fraud_count = sum(1 for t in transactions if t["is_fraud"] == 1)
            print(f"  Legitimate: {len(transactions) - fraud_count}", file=sys.stderr)
            print(f"  Fraudulent: {fraud_count} ({fraud_count/len(transactions)*100:.1f}%)", file=sys.stderr)
        else:
            for txn in transactions:
                print(json.dumps(txn))


if __name__ == "__main__":
    main()
