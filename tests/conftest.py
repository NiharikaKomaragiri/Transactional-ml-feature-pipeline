"""Shared pytest fixtures."""

import os
from pathlib import Path
import tempfile
import pytest
import pandas as pd

from src.config import PipelineConfig
from src.database import DatabaseManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for isolated test runs."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_config(temp_dir):
    """Create a PipelineConfig pointing to temporary directories."""
    config = PipelineConfig(
        seed=42,
        num_customers=50,
        num_transactions=500,
        feature_windows_days=[7, 30, 90],
        observation_date="2026-06-01T00:00:00Z",
        project_root=temp_dir,
        data_dir=temp_dir / "data",
        raw_dir=temp_dir / "data" / "raw",
        quarantine_dir=temp_dir / "data" / "quarantine",
        features_dir=temp_dir / "data" / "features",
        logs_dir=temp_dir / "logs",
        db_path=temp_dir / "data" / "test_pipeline.db",
    )
    config.ensure_directories()
    return config


@pytest.fixture
def db_manager(temp_config):
    """Initialize DatabaseManager with temporary test database."""
    return DatabaseManager(temp_config.db_path)


@pytest.fixture
def sample_customers_df():
    """Small deterministic customers dataframe."""
    return pd.DataFrame([
        {
            "customer_id": "CUST_000001",
            "signup_timestamp": "2025-01-10T12:00:00Z",
            "country_code": "US",
            "kyc_status": "VERIFIED",
            "account_tier": "VIP",
            "is_active": True,
        },
        {
            "customer_id": "CUST_000002",
            "signup_timestamp": "2025-02-15T15:30:00Z",
            "country_code": "IN",
            "kyc_status": "VERIFIED",
            "account_tier": "STANDARD",
            "is_active": True,
        },
        {
            "customer_id": "CUST_000003",
            "signup_timestamp": "2025-03-20T08:00:00Z",
            "country_code": "GB",
            "kyc_status": "PENDING",
            "account_tier": "PREMIUM",
            "is_active": False,
        },
    ])


@pytest.fixture
def sample_transactions_df():
    """Small deterministic transactions dataframe."""
    return pd.DataFrame([
        {
            "transaction_id": "TX_000000001",
            "customer_id": "CUST_000001",
            "event_timestamp": "2026-05-30T10:00:00Z",      # 2 days before obs (in 7d, 30d, 90d)
            "available_timestamp": "2026-05-30T10:05:00Z",
            "amount": 100.0,
            "currency": "USD",
            "channel": "MOBILE",
            "category": "RETAIL",
            "status": "COMPLETED",
        },
        {
            "transaction_id": "TX_000000002",
            "customer_id": "CUST_000001",
            "event_timestamp": "2026-05-20T14:00:00Z",      # 12 days before obs (in 30d, 90d)
            "available_timestamp": "2026-05-20T14:10:00Z",
            "amount": 200.0,
            "currency": "USD",
            "channel": "WEB",
            "category": "GROCERY",
            "status": "COMPLETED",
        },
        {
            "transaction_id": "TX_000000003",
            "customer_id": "CUST_000001",
            "event_timestamp": "2026-04-01T12:00:00Z",      # 61 days before obs (in 90d)
            "available_timestamp": "2026-04-01T12:02:00Z",
            "amount": 300.0,
            "currency": "USD",
            "channel": "POS",
            "category": "TRAVEL",
            "status": "COMPLETED",
        },
        {
            "transaction_id": "TX_000000004",
            "customer_id": "CUST_000002",
            "event_timestamp": "2026-05-28T09:00:00Z",      # 4 days before obs (in 7d, 30d, 90d)
            "available_timestamp": "2026-05-28T09:01:00Z",
            "amount": 50.0,
            "currency": "USD",
            "channel": "MOBILE",
            "category": "RETAIL",
            "status": "COMPLETED",
        },
    ])
