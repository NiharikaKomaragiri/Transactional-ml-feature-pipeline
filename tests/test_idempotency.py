"""Tests for idempotency of repeated pipeline runs and atomic failure recovery."""

import sqlite3
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.database import DatabaseManager
from src.pipeline import FeaturePipeline


def test_repeated_pipeline_runs_are_idempotent(temp_config, sample_customers_df, sample_transactions_df):
    """Running the pipeline twice with the same inputs must yield identical database state."""
    pipeline = FeaturePipeline(config=temp_config)

    # First Run
    metrics_1 = pipeline.run(
        run_id="RUN_IDEMPOTENCY_01",
        customers_df=sample_customers_df,
        transactions_df=sample_transactions_df,
    )
    assert metrics_1.status == "SUCCESS"

    conn1 = sqlite3.connect(str(temp_config.db_path))
    count_cust_1 = conn1.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    count_tx_1 = conn1.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    count_feat_1 = conn1.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    conn1.close()

    # Second Run with identical data and different run_id
    metrics_2 = pipeline.run(
        run_id="RUN_IDEMPOTENCY_02",
        customers_df=sample_customers_df,
        transactions_df=sample_transactions_df,
    )
    assert metrics_2.status == "SUCCESS"

    conn2 = sqlite3.connect(str(temp_config.db_path))
    count_cust_2 = conn2.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    count_tx_2 = conn2.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    count_feat_2 = conn2.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    conn2.close()

    # Row counts in primary tables must remain exactly the same (no duplicates)
    assert count_cust_1 == count_cust_2 == len(sample_customers_df)
    assert count_tx_1 == count_tx_2 == len(sample_transactions_df)
    assert count_feat_1 == count_feat_2 == len(sample_customers_df)


def test_atomic_rollback_on_failure(temp_config):
    """Verify that an exception inside get_connection() triggers an automatic rollback."""
    db = DatabaseManager(temp_config.db_path)

    # Initial state
    with db.get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO customers VALUES ('C_BASE', '2025-01-01', 'US', 'VERIFIED', 'VIP', 1)")

    # Attempt a transaction that raises an error after inserting
    with pytest.raises(ValueError, match="Simulated crash"):
        with db.get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO customers VALUES ('C_CRASH', '2025-01-01', 'US', 'VERIFIED', 'VIP', 1)")
            raise ValueError("Simulated crash mid-transaction!")

    # Verify that 'C_CRASH' was rolled back and is NOT present in the database
    conn3 = sqlite3.connect(str(temp_config.db_path))
    row = conn3.execute("SELECT customer_id FROM customers WHERE customer_id = 'C_CRASH'").fetchone()
    conn3.close()
    assert row is None, "Corrupted record must be rolled back on failure!"
