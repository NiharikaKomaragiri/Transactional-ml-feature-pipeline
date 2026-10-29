"""Tests for synthetic data generator reproducibility, scale, and anomaly injection."""

import pandas as pd
import pytest

from src.generator import SyntheticDataGenerator, generate_and_save_data


def test_generator_reproducibility():
    """Verify that identical random seeds produce byte-for-byte identical datasets."""
    gen1 = SyntheticDataGenerator(seed=123)
    gen2 = SyntheticDataGenerator(seed=123)

    cust1 = gen1.generate_customers(num_customers=100)
    cust2 = gen2.generate_customers(num_customers=100)
    pd.testing.assert_frame_equal(cust1, cust2)

    tx1 = gen1.generate_transactions(
        customer_ids=cust1["customer_id"].tolist(),
        num_transactions=500,
        inject_anomalies=False,
    )
    tx2 = gen2.generate_transactions(
        customer_ids=cust2["customer_id"].tolist(),
        num_transactions=500,
        inject_anomalies=False,
    )
    pd.testing.assert_frame_equal(tx1, tx2)


def test_generator_different_seeds():
    """Verify that different random seeds produce different datasets."""
    gen1 = SyntheticDataGenerator(seed=42)
    gen2 = SyntheticDataGenerator(seed=99)

    cust1 = gen1.generate_customers(num_customers=50)
    cust2 = gen2.generate_customers(num_customers=50)

    # Country codes or timestamps should differ
    assert not cust1["signup_timestamp"].equals(cust2["signup_timestamp"])


def test_small_sample_and_configurable_scale():
    """Test generating small samples vs configurable sizes."""
    gen = SyntheticDataGenerator(seed=42)
    cust = gen.generate_customers(num_customers=25)
    tx = gen.generate_transactions(customer_ids=cust["customer_id"].tolist(), num_transactions=75)

    assert len(cust) == 25
    assert len(tx) == 75
    assert set(CustomerSchemaRequired := ["customer_id", "signup_timestamp", "is_active"]).issubset(cust.columns)
    assert set(["transaction_id", "customer_id", "amount", "event_timestamp"]).issubset(tx.columns)


def test_anomaly_injection():
    """Verify that anomaly injection introduces invalid records."""
    gen = SyntheticDataGenerator(seed=42)
    cust = gen.generate_customers(num_customers=50)
    tx = gen.generate_transactions(
        customer_ids=cust["customer_id"].tolist(),
        num_transactions=200,
        inject_anomalies=True,
        anomaly_rate=0.10,
    )

    # Should contain some nulls, negative amounts, or orphan keys
    has_nulls = tx["customer_id"].isna().any()
    has_bad_amounts = (tx["amount"] <= 0).any()
    has_orphans = (tx["customer_id"] == "CUST_999999").any()
    has_bad_channel = (tx["channel"] == "BITCOIN_KIOSK").any()

    assert any([has_nulls, has_bad_amounts, has_orphans, has_bad_channel])


def test_target_generation_isolation():
    """Verify ground truth target generation produces clean binary labels and full customer coverage."""
    gen = SyntheticDataGenerator(seed=42)
    cust = gen.generate_customers(num_customers=50)
    tx = gen.generate_transactions(customer_ids=cust["customer_id"].tolist(), num_transactions=200)

    # 1. With customers_df: exactly one target per customer in customer dataset
    targets_with_cust = gen.generate_observation_targets(transactions_df=tx, customers_df=cust)
    assert len(targets_with_cust) == 50
    assert set(targets_with_cust["customer_id"]) == set(cust["customer_id"])
    assert set(targets_with_cust["target_label"].unique()).issubset({0, 1})
    assert not targets_with_cust["target_label"].isna().any()
    assert "amount" not in targets_with_cust.columns

    # 2. Backward compatibility: when only transactions_df is passed
    targets_legacy = gen.generate_observation_targets(tx)
    assert len(targets_legacy) > 0
    assert set(targets_legacy["target_label"].unique()).issubset({0, 1})
