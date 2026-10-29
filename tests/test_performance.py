"""Performance benchmarks and 100,000+ transaction throughput verification."""

import time
import pytest

from src.features import FeatureEngineer
from src.generator import SyntheticDataGenerator


def test_performance_100k_transactions():
    """Benchmark vectorized feature calculation for 100,000 transactions."""
    num_customers = 1_000
    num_transactions = 100_000

    gen = SyntheticDataGenerator(seed=42)
    cust_df = gen.generate_customers(num_customers=num_customers)
    tx_df = gen.generate_transactions(
        customer_ids=cust_df["customer_id"].tolist(),
        num_transactions=num_transactions,
        inject_anomalies=False,
    )

    engineer = FeatureEngineer(windows_days=[7, 30, 90])

    start_time = time.perf_counter()
    features = engineer.compute_features(
        customers_df=cust_df,
        transactions_df=tx_df,
        observation_timestamp="2026-06-01T00:00:00Z",
    )
    elapsed = time.perf_counter() - start_time

    assert len(features) == num_customers
    # Assert performance threshold: 100k records must be processed in under 10 seconds
    assert elapsed < 10.0, f"Vectorized feature computation took {elapsed:.2f}s (exceeded 10s budget)!"
    print(f"\n[Performance Benchmark] 100,000 transactions processed in {elapsed:.3f} seconds ({num_transactions / elapsed:.1f} tx/sec).")
