"""Tests for rolling window feature calculations and mathematical correctness."""

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureEngineer


def test_feature_window_aggregations(sample_customers_df, sample_transactions_df):
    """Verify exact numerical feature calculations across 7d, 30d, and 90d windows."""
    engineer = FeatureEngineer(windows_days=[7, 30, 90])
    features = engineer.compute_features(
        customers_df=sample_customers_df,
        transactions_df=sample_transactions_df,
        observation_timestamp="2026-06-01T00:00:00Z",
    )

    assert len(features) == len(sample_customers_df)

    # 1. Customer 1 (CUST_000001) has 3 transactions: May 30 ($100), May 20 ($200), Apr 1 ($300)
    c1 = features[features["customer_id"] == "CUST_000001"].iloc[0]
    
    # 7-day window (only May 30)
    assert c1["tx_count_7d"] == 1
    assert c1["tx_sum_amount_7d"] == 100.0
    assert c1["tx_avg_amount_7d"] == 100.0
    assert c1["tx_max_amount_7d"] == 100.0
    assert c1["tx_std_amount_7d"] == 0.0

    # 30-day window (May 30 and May 20)
    assert c1["tx_count_30d"] == 2
    assert c1["tx_sum_amount_30d"] == 300.0
    assert c1["tx_avg_amount_30d"] == 150.0
    assert c1["tx_max_amount_30d"] == 200.0
    # Sample std of [100, 200] is ~70.71
    assert np.isclose(c1["tx_std_amount_30d"], 70.71, atol=0.05)

    # 90-day window (May 30, May 20, Apr 1)
    assert c1["tx_count_90d"] == 3
    assert c1["tx_sum_amount_90d"] == 600.0
    assert c1["tx_avg_amount_90d"] == 200.0
    assert c1["tx_max_amount_90d"] == 300.0

    # Recency (May 30 is ~1.58 days before June 1 00:00:00 UTC)
    assert 1.0 < c1["days_since_last_tx"] < 2.0


def test_customer_without_transactions(sample_customers_df, sample_transactions_df):
    """Verify that a customer with 0 transactions receives cleanly defaulted zero-features."""
    engineer = FeatureEngineer(windows_days=[7, 30, 90])
    features = engineer.compute_features(
        customers_df=sample_customers_df,
        transactions_df=sample_transactions_df,
        observation_timestamp="2026-06-01T00:00:00Z",
    )

    # Customer 3 (CUST_000003) has 0 transactions
    c3 = features[features["customer_id"] == "CUST_000003"].iloc[0]
    assert c3["tx_count_7d"] == 0
    assert c3["tx_sum_amount_7d"] == 0.0
    assert c3["tx_avg_amount_7d"] == 0.0
    assert c3["tx_max_amount_7d"] == 0.0
    assert c3["tx_count_30d"] == 0
    assert c3["tx_sum_amount_30d"] == 0.0
    assert c3["tx_count_90d"] == 0
    assert c3["tx_sum_amount_90d"] == 0.0
    assert c3["days_since_last_tx"] == 999.0
    assert c3["tx_spend_velocity_7d_to_30d"] == 0.0


def test_spend_velocity_calculation(sample_customers_df, sample_transactions_df):
    """Verify cross-window velocity calculation (7d spend relative to weekly 30d baseline)."""
    engineer = FeatureEngineer(windows_days=[7, 30])
    features = engineer.compute_features(
        customers_df=sample_customers_df,
        transactions_df=sample_transactions_df,
        observation_timestamp="2026-06-01T00:00:00Z",
    )

    c1 = features[features["customer_id"] == "CUST_000001"].iloc[0]
    # c1 7d spend = 100, 30d spend = 300 -> baseline weekly = 300 / 4.2857 = 70.00
    # velocity = 100 / 70.00 = ~1.428
    assert "tx_spend_velocity_7d_to_30d" in c1
    assert c1["tx_spend_velocity_7d_to_30d"] > 1.0
