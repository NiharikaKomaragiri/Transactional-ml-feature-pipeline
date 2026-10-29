"""Automated tests for target generation, horizon boundaries, and leakage prevention."""

from datetime import datetime, timezone
import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.features import FeatureEngineer
from src.generator import SyntheticDataGenerator
from src.pipeline import FeaturePipeline


@pytest.fixture
def target_test_customers():
    """Deterministic customer dataset for target generation tests."""
    return pd.DataFrame([
        {"customer_id": "CUST_001", "signup_timestamp": "2025-01-01T00:00:00Z", "country_code": "US", "kyc_status": "VERIFIED", "account_tier": "VIP", "is_active": True},
        {"customer_id": "CUST_002", "signup_timestamp": "2025-01-01T00:00:00Z", "country_code": "GB", "kyc_status": "VERIFIED", "account_tier": "STANDARD", "is_active": True},
        {"customer_id": "CUST_003", "signup_timestamp": "2025-01-01T00:00:00Z", "country_code": "IN", "kyc_status": "VERIFIED", "account_tier": "PREMIUM", "is_active": True},
        {"customer_id": "CUST_004", "signup_timestamp": "2025-01-01T00:00:00Z", "country_code": "CA", "kyc_status": "VERIFIED", "account_tier": "STANDARD", "is_active": True},
        {"customer_id": "CUST_005", "signup_timestamp": "2025-01-01T00:00:00Z", "country_code": "DE", "kyc_status": "VERIFIED", "account_tier": "VIP", "is_active": True},
    ])


def test_every_valid_customer_receives_exactly_one_target(target_test_customers):
    """Every customer in the customer dataset must receive exactly one observation target."""
    gen = SyntheticDataGenerator(seed=42)
    # Transactions exist for ONLY CUST_001
    tx_df = pd.DataFrame([
        {
            "transaction_id": "TX_01",
            "customer_id": "CUST_001",
            "event_timestamp": "2026-06-05T12:00:00Z",
            "available_timestamp": "2026-06-05T12:05:00Z",
            "amount": 100.0,
            "status": "COMPLETED",
        }
    ])

    targets = gen.generate_observation_targets(
        transactions_df=tx_df,
        customers_df=target_test_customers,
        observation_date="2026-06-01T00:00:00Z",
        horizon_days=30,
    )

    # Must contain all 5 customers
    assert len(targets) == len(target_test_customers)
    assert set(targets["customer_id"]) == set(target_test_customers["customer_id"])
    assert targets["customer_id"].is_unique
    assert targets["observation_id"].is_unique


def test_customers_with_no_transactions_receive_label_zero(target_test_customers):
    """Customers with zero transactions must receive a target label of 0."""
    gen = SyntheticDataGenerator(seed=42)
    empty_tx = pd.DataFrame(columns=["transaction_id", "customer_id", "event_timestamp", "amount", "status"])

    targets = gen.generate_observation_targets(
        transactions_df=empty_tx,
        customers_df=target_test_customers,
        observation_date="2026-06-01T00:00:00Z",
        horizon_days=30,
    )

    assert len(targets) == 5
    assert (targets["target_label"] == 0).all()


def test_future_spending_above_threshold_receives_label_one(target_test_customers):
    """Customer with total future spending > $1,500 receives label 1; <= $1,500 receives 0."""
    gen = SyntheticDataGenerator(seed=42)
    tx_df = pd.DataFrame([
        # CUST_001: $1,600 total future spend (> $1,500)
        {
            "transaction_id": "TX_01",
            "customer_id": "CUST_001",
            "event_timestamp": "2026-06-10T10:00:00Z",
            "amount": 1000.0,
            "status": "COMPLETED",
        },
        {
            "transaction_id": "TX_02",
            "customer_id": "CUST_001",
            "event_timestamp": "2026-06-15T10:00:00Z",
            "amount": 600.0,
            "status": "COMPLETED",
        },
        # CUST_002: exactly $1,500 future spend (not > $1,500)
        {
            "transaction_id": "TX_03",
            "customer_id": "CUST_002",
            "event_timestamp": "2026-06-10T10:00:00Z",
            "amount": 1500.0,
            "status": "COMPLETED",
        },
        # CUST_003: $250 future spend
        {
            "transaction_id": "TX_04",
            "customer_id": "CUST_003",
            "event_timestamp": "2026-06-10T10:00:00Z",
            "amount": 250.0,
            "status": "COMPLETED",
        },
    ])

    targets = gen.generate_observation_targets(
        transactions_df=tx_df,
        customers_df=target_test_customers,
        observation_date="2026-06-01T00:00:00Z",
        horizon_days=30,
    )

    t_map = dict(zip(targets["customer_id"], targets["target_label"]))
    assert t_map["CUST_001"] == 1  # 1600 > 1500
    assert t_map["CUST_002"] == 0  # 1500 is not > 1500
    assert t_map["CUST_003"] == 0  # 250 <= 1500
    assert t_map["CUST_004"] == 0  # 0 transactions
    assert t_map["CUST_005"] == 0  # 0 transactions


def test_declined_transaction_in_future_horizon_receives_label_one(target_test_customers):
    """Customer with at least one DECLINED transaction in future horizon receives label 1."""
    gen = SyntheticDataGenerator(seed=42)
    tx_df = pd.DataFrame([
        # CUST_001 has only $25 spend, but has a DECLINED transaction
        {
            "transaction_id": "TX_01",
            "customer_id": "CUST_001",
            "event_timestamp": "2026-06-12T14:00:00Z",
            "amount": 25.0,
            "status": "DECLINED",
        },
        # CUST_002 has $100 completed, no decline
        {
            "transaction_id": "TX_02",
            "customer_id": "CUST_002",
            "event_timestamp": "2026-06-12T14:00:00Z",
            "amount": 100.0,
            "status": "COMPLETED",
        },
    ])

    targets = gen.generate_observation_targets(
        transactions_df=tx_df,
        customers_df=target_test_customers,
        observation_date="2026-06-01T00:00:00Z",
        horizon_days=30,
    )

    t_map = dict(zip(targets["customer_id"], targets["target_label"]))
    assert t_map["CUST_001"] == 1  # Has declined transaction
    assert t_map["CUST_002"] == 0  # No decline and spend <= 1500


def test_transactions_outside_future_horizon_do_not_affect_labels(target_test_customers):
    """Transactions before observation cutoff or after horizon_end must NOT affect target."""
    gen = SyntheticDataGenerator(seed=42)
    obs_time = "2026-06-01T00:00:00Z"

    tx_df = pd.DataFrame([
        # Past transaction before obs (1 second prior): $5,000 spend & DECLINED
        {
            "transaction_id": "TX_PAST",
            "customer_id": "CUST_001",
            "event_timestamp": "2026-05-31T23:59:59Z",
            "amount": 5000.0,
            "status": "DECLINED",
        },
        # Exact start of horizon (2026-06-01T00:00:00Z) - INCLUSIVE: $1,600 spend
        {
            "transaction_id": "TX_EXACT_START",
            "customer_id": "CUST_002",
            "event_timestamp": "2026-06-01T00:00:00Z",
            "amount": 1600.0,
            "status": "COMPLETED",
        },
        # Far future transaction after horizon (2026-07-01T00:00:00Z) - EXCLUSIVE: $5,000 spend & DECLINED
        {
            "transaction_id": "TX_FAR_FUTURE",
            "customer_id": "CUST_003",
            "event_timestamp": "2026-07-01T00:00:00Z",
            "amount": 5000.0,
            "status": "DECLINED",
        },
    ])

    targets = gen.generate_observation_targets(
        transactions_df=tx_df,
        customers_df=target_test_customers,
        observation_date=obs_time,
        horizon_days=30,
    )

    t_map = dict(zip(targets["customer_id"], targets["target_label"]))
    assert t_map["CUST_001"] == 0, "Past transactions must not count toward future target!"
    assert t_map["CUST_002"] == 1, "Transaction at exact horizon start must be included (inclusive start)!"
    assert t_map["CUST_003"] == 0, "Transaction at horizon end must be excluded (exclusive end)!"


def test_invalid_and_orphan_transactions_do_not_affect_labels(target_test_customers):
    """Orphan customers, negative amounts, and duplicates must not distort target labels."""
    gen = SyntheticDataGenerator(seed=42)

    tx_df = pd.DataFrame([
        # Orphan customer not in target_test_customers
        {
            "transaction_id": "TX_ORPHAN",
            "customer_id": "CUST_ORPHAN_999",
            "event_timestamp": "2026-06-10T12:00:00Z",
            "amount": 9999.0,
            "status": "DECLINED",
        },
        # CUST_001 has valid $1,000 spend and a duplicate $1,000 spend with the same transaction_id
        {
            "transaction_id": "TX_DUP",
            "customer_id": "CUST_001",
            "event_timestamp": "2026-06-10T12:00:00Z",
            "amount": 1000.0,
            "status": "COMPLETED",
        },
        {
            "transaction_id": "TX_DUP",  # Duplicate transaction_id!
            "customer_id": "CUST_001",
            "event_timestamp": "2026-06-10T12:00:00Z",
            "amount": 1000.0,
            "status": "COMPLETED",
        },
        # CUST_002 has negative amount which should be clipped to 0
        {
            "transaction_id": "TX_NEG",
            "customer_id": "CUST_002",
            "event_timestamp": "2026-06-10T12:00:00Z",
            "amount": -5000.0,
            "status": "COMPLETED",
        },
    ])

    targets = gen.generate_observation_targets(
        transactions_df=tx_df,
        customers_df=target_test_customers,
        observation_date="2026-06-01T00:00:00Z",
        horizon_days=30,
    )

    assert "CUST_ORPHAN_999" not in set(targets["customer_id"]), "Orphan customer must not appear in target dataset!"
    assert len(targets) == len(target_test_customers)

    t_map = dict(zip(targets["customer_id"], targets["target_label"]))
    # CUST_001 should only have $1,000 counted once (deduplicated), so label = 0 (not > 1500)
    assert t_map["CUST_001"] == 0, "Duplicate transaction must not be counted twice to exceed $1500 threshold!"
    assert t_map["CUST_002"] == 0


def test_future_transactions_do_not_leak_into_historical_features(target_test_customers):
    """Transactions in the future horizon used for targets must NEVER leak into historical features."""
    obs_time = "2026-06-01T00:00:00Z"
    
    tx_df = pd.DataFrame([
        # Past transaction (eligible for features)
        {
            "transaction_id": "TX_HIST_01",
            "customer_id": "CUST_001",
            "event_timestamp": "2026-05-25T10:00:00Z",
            "available_timestamp": "2026-05-25T10:05:00Z",
            "amount": 50.0,
            "status": "COMPLETED",
        },
        # Future transaction (target horizon: June 1 to July 1)
        {
            "transaction_id": "TX_FUTR_01",
            "customer_id": "CUST_001",
            "event_timestamp": "2026-06-15T10:00:00Z",
            "available_timestamp": "2026-06-15T10:05:00Z",
            "amount": 2500.0,
            "status": "COMPLETED",
        },
    ])

    # 1. Historical Feature Engineering as of obs_time
    engineer = FeatureEngineer(windows_days=[7, 30, 90])
    features = engineer.compute_features(
        customers_df=target_test_customers,
        transactions_df=tx_df,
        observation_timestamp=obs_time,
    )

    c1_feat = features[features["customer_id"] == "CUST_001"].iloc[0]
    # Future transaction of $2,500 MUST NOT appear in 7d, 30d, or 90d features!
    assert c1_feat["tx_count_7d"] == 1
    assert c1_feat["tx_sum_amount_7d"] == 50.0
    assert c1_feat["tx_count_30d"] == 1
    assert c1_feat["tx_sum_amount_30d"] == 50.0
    assert c1_feat["tx_max_amount_30d"] == 50.0

    # 2. Target Generation uses the future transaction
    gen = SyntheticDataGenerator(seed=42)
    targets = gen.generate_observation_targets(
        transactions_df=tx_df,
        customers_df=target_test_customers,
        observation_date=obs_time,
        horizon_days=30,
    )
    c1_target = targets[targets["customer_id"] == "CUST_001"].iloc[0]
    # CUST_001 spent $2,500 in future horizon, so target is 1
    assert c1_target["target_label"] == 1


def test_repeated_runs_target_idempotency(temp_config, sample_customers_df, sample_transactions_df):
    """Running the pipeline repeatedly produces consistent, idempotent target records with no duplicates."""
    pipeline = FeaturePipeline(config=temp_config)

    metrics_1 = pipeline.run(
        run_id="RUN_TARGET_IDEMPOTENCY_01",
        customers_df=sample_customers_df,
        transactions_df=sample_transactions_df,
    )
    assert metrics_1.status == "SUCCESS"

    conn = sqlite3.connect(str(temp_config.db_path))
    target_count_1 = conn.execute("SELECT COUNT(*) FROM observation_targets").fetchone()[0]
    distinct_cust_1 = conn.execute("SELECT COUNT(DISTINCT customer_id) FROM observation_targets").fetchone()[0]
    conn.close()

    metrics_2 = pipeline.run(
        run_id="RUN_TARGET_IDEMPOTENCY_02",
        customers_df=sample_customers_df,
        transactions_df=sample_transactions_df,
    )
    assert metrics_2.status == "SUCCESS"

    conn = sqlite3.connect(str(temp_config.db_path))
    target_count_2 = conn.execute("SELECT COUNT(*) FROM observation_targets").fetchone()[0]
    distinct_cust_2 = conn.execute("SELECT COUNT(DISTINCT customer_id) FROM observation_targets").fetchone()[0]
    conn.close()

    assert target_count_1 == len(sample_customers_df)
    assert target_count_2 == len(sample_customers_df)
    assert distinct_cust_1 == len(sample_customers_df)
    assert distinct_cust_2 == len(sample_customers_df)
