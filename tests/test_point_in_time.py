"""Tests for Point-in-Time (PIT) correctness, ingestion lag, and target isolation."""

from datetime import datetime, timezone
import pandas as pd
import pytest

from src.point_in_time import PointInTimeEngine


def test_future_events_excluded():
    """Verify transactions occurring after observation_timestamp are strictly excluded."""
    obs_time = "2026-06-01T00:00:00Z"
    tx_df = pd.DataFrame([
        {
            "transaction_id": "TX_PAST",
            "customer_id": "CUST_1",
            "event_timestamp": "2026-05-31T23:59:59Z",
            "available_timestamp": "2026-05-31T23:59:59Z",
            "amount": 100.0,
            "status": "COMPLETED",
        },
        {
            "transaction_id": "TX_FUTURE",
            "customer_id": "CUST_1",
            "event_timestamp": "2026-06-01T00:00:01Z",  # 1 second in future!
            "available_timestamp": "2026-06-01T00:00:01Z",
            "amount": 200.0,
            "status": "COMPLETED",
        },
    ])

    pit_tx = PointInTimeEngine.filter_transactions_as_of(tx_df, observation_timestamp=obs_time)
    assert len(pit_tx) == 1
    assert pit_tx.iloc[0]["transaction_id"] == "TX_PAST"


def test_late_arriving_events_excluded_then_included():
    """Verify delayed transactions (lag leakage prevention) as-of past vs future dates."""
    # Transaction happened on May 30, but only arrived in DB on June 2 (e.g. offline store sync)
    tx_df = pd.DataFrame([
        {
            "transaction_id": "TX_LATE",
            "customer_id": "CUST_1",
            "event_timestamp": "2026-05-30T12:00:00Z",
            "available_timestamp": "2026-06-02T12:00:00Z",  # Ingested June 2
            "amount": 350.0,
            "status": "COMPLETED",
        }
    ])

    # 1. As of June 1: Event happened before June 1, BUT was NOT available yet!
    pit_june1 = PointInTimeEngine.filter_transactions_as_of(
        tx_df, observation_timestamp="2026-06-01T00:00:00Z"
    )
    assert len(pit_june1) == 0, "Lagged transaction must not leak into June 1 features!"

    # 2. As of June 3: Transaction is now available in the system!
    pit_june3 = PointInTimeEngine.filter_transactions_as_of(
        tx_df, observation_timestamp="2026-06-03T00:00:00Z"
    )
    assert len(pit_june3) == 1, "Transaction should be eligible once available_timestamp is reached."


def test_lookback_boundary():
    """Verify events older than max_lookback_days are excluded."""
    obs_time = "2026-06-01T00:00:00Z"
    tx_df = pd.DataFrame([
        {
            "transaction_id": "TX_WITHIN_90D",
            "customer_id": "CUST_1",
            "event_timestamp": "2026-04-01T00:00:00Z",  # ~61 days ago
            "available_timestamp": "2026-04-01T00:05:00Z",
            "amount": 100.0,
            "status": "COMPLETED",
        },
        {
            "transaction_id": "TX_BEYOND_90D",
            "customer_id": "CUST_1",
            "event_timestamp": "2026-02-01T00:00:00Z",  # ~120 days ago
            "available_timestamp": "2026-02-01T00:05:00Z",
            "amount": 200.0,
            "status": "COMPLETED",
        },
    ])

    pit_tx = PointInTimeEngine.filter_transactions_as_of(
        tx_df, observation_timestamp=obs_time, max_lookback_days=90
    )
    assert len(pit_tx) == 1
    assert pit_tx.iloc[0]["transaction_id"] == "TX_WITHIN_90D"


def test_declined_status_filter():
    """Verify declined/pending transactions are excluded from completed spend features."""
    obs_time = "2026-06-01T00:00:00Z"
    tx_df = pd.DataFrame([
        {
            "transaction_id": "TX_COMPLETED",
            "customer_id": "CUST_1",
            "event_timestamp": "2026-05-25T00:00:00Z",
            "available_timestamp": "2026-05-25T00:05:00Z",
            "amount": 100.0,
            "status": "COMPLETED",
        },
        {
            "transaction_id": "TX_DECLINED",
            "customer_id": "CUST_1",
            "event_timestamp": "2026-05-25T00:00:00Z",
            "available_timestamp": "2026-05-25T00:05:00Z",
            "amount": 500.0,
            "status": "DECLINED",
        },
    ])

    pit_tx = PointInTimeEngine.filter_transactions_as_of(
        tx_df, observation_timestamp=obs_time, only_completed=True
    )
    assert len(pit_tx) == 1
    assert pit_tx.iloc[0]["transaction_id"] == "TX_COMPLETED"
