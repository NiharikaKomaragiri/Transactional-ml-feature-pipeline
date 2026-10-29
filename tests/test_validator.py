"""Tests for data quality validation, constraints, and quarantine routing."""

import pandas as pd
import pytest

from src.schemas import RejectionReason
from src.validator import DataQualityValidator


def test_clean_data_passes(sample_customers_df, sample_transactions_df):
    """Verify clean valid datasets produce zero quarantined records."""
    validator = DataQualityValidator(run_id="TEST_RUN_001")
    valid_cust, quaran_cust = validator.validate_customers(sample_customers_df)
    assert len(quaran_cust) == 0
    assert len(valid_cust) == len(sample_customers_df)

    valid_cust_ids = set(valid_cust["customer_id"])
    valid_tx, quaran_tx = validator.validate_transactions(sample_transactions_df, valid_cust_ids)
    assert len(quaran_tx) == 0
    assert len(valid_tx) == len(sample_transactions_df)


def test_missing_required_fields(sample_transactions_df):
    """Verify records with missing required fields are quarantined."""
    validator = DataQualityValidator(run_id="TEST_RUN_002")
    bad_df = sample_transactions_df.copy()
    bad_df.at[0, "customer_id"] = None
    bad_df.at[1, "amount"] = None

    valid_tx, quaran_tx = validator.validate_transactions(
        bad_df, valid_customer_ids={"CUST_000001", "CUST_000002", "CUST_000003"}
    )
    assert len(quaran_tx) == 2
    assert (quaran_tx["rejection_reason"] == RejectionReason.MISSING_REQUIRED_FIELD.value).all()
    assert len(valid_tx) == len(bad_df) - 2


def test_invalid_amount(sample_transactions_df):
    """Verify negative and zero amounts are quarantined."""
    validator = DataQualityValidator(run_id="TEST_RUN_003")
    bad_df = sample_transactions_df.copy()
    bad_df.at[0, "amount"] = -50.0
    bad_df.at[1, "amount"] = 0.0

    valid_tx, quaran_tx = validator.validate_transactions(
        bad_df, valid_customer_ids={"CUST_000001", "CUST_000002"}
    )
    assert len(quaran_tx) == 2
    assert (quaran_tx["rejection_reason"] == RejectionReason.INVALID_AMOUNT.value).all()


def test_negative_timestamp_lag(sample_transactions_df):
    """Verify available_timestamp < event_timestamp is caught as causality violation."""
    validator = DataQualityValidator(run_id="TEST_RUN_004")
    bad_df = sample_transactions_df.copy()
    # Event is 2026-05-30T10:00:00Z, set available earlier
    bad_df.at[0, "available_timestamp"] = "2026-05-30T09:00:00Z"

    valid_tx, quaran_tx = validator.validate_transactions(
        bad_df, valid_customer_ids={"CUST_000001", "CUST_000002"}
    )
    assert len(quaran_tx) == 1
    assert quaran_tx.iloc[0]["rejection_reason"] == RejectionReason.NEGATIVE_TIMESTAMP_LAG.value


def test_orphan_foreign_keys(sample_transactions_df):
    """Verify transactions referencing non-existent customers are quarantined."""
    validator = DataQualityValidator(run_id="TEST_RUN_005")
    bad_df = sample_transactions_df.copy()
    bad_df.at[0, "customer_id"] = "CUST_UNKNOWN_999"

    valid_tx, quaran_tx = validator.validate_transactions(
        bad_df, valid_customer_ids={"CUST_000001", "CUST_000002"}
    )
    assert len(quaran_tx) == 1
    assert quaran_tx.iloc[0]["rejection_reason"] == RejectionReason.ORPHAN_FOREIGN_KEY.value


def test_duplicate_primary_keys(sample_transactions_df):
    """Verify duplicate transaction_ids are identified and quarantined."""
    validator = DataQualityValidator(run_id="TEST_RUN_006")
    bad_df = sample_transactions_df.copy()
    # Make row 1 have the same transaction_id as row 0
    bad_df.at[1, "transaction_id"] = bad_df.at[0, "transaction_id"]

    valid_tx, quaran_tx = validator.validate_transactions(
        bad_df, valid_customer_ids={"CUST_000001", "CUST_000002"}
    )
    assert len(quaran_tx) == 1
    assert quaran_tx.iloc[0]["rejection_reason"] == RejectionReason.DUPLICATE_PRIMARY_KEY.value
    assert len(valid_tx) == len(bad_df) - 1


def test_malformed_timestamp(sample_transactions_df):
    """Verify invalid date strings are caught."""
    validator = DataQualityValidator(run_id="TEST_RUN_007")
    bad_df = sample_transactions_df.copy()
    bad_df.at[0, "event_timestamp"] = "NOT_A_DATE"

    valid_tx, quaran_tx = validator.validate_transactions(
        bad_df, valid_customer_ids={"CUST_000001", "CUST_000002"}
    )
    assert len(quaran_tx) == 1
    assert quaran_tx.iloc[0]["rejection_reason"] == RejectionReason.MALFORMED_TIMESTAMP.value


def test_empty_dataframe():
    """Verify validator handles 0-row inputs gracefully without raising."""
    validator = DataQualityValidator(run_id="TEST_RUN_008")
    empty_df = pd.DataFrame()

    valid_c, quaran_c = validator.validate_customers(empty_df)
    assert valid_c.empty
    assert quaran_c.empty

    valid_t, quaran_t = validator.validate_transactions(empty_df, valid_customer_ids=set())
    assert valid_t.empty
    assert quaran_t.empty


def test_schema_mismatch():
    """Verify missing required columns causes rejection."""
    validator = DataQualityValidator(run_id="TEST_RUN_009")
    bad_df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})

    valid_t, quaran_t = validator.validate_transactions(bad_df, valid_customer_ids=set())
    assert valid_t.empty
    assert len(quaran_t) == 2
    assert (quaran_t["rejection_reason"] == RejectionReason.SCHEMA_MISMATCH.value).all()
