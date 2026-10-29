"""Tests for reconciliation reporting and mathematical balance invariants."""

from pathlib import Path
import pandas as pd
import pytest

from src.reconciliation import ReconciliationEngine


def test_reconciliation_math_balance():
    """Verify is_balanced is True when Raw == Valid + Quarantined, and False otherwise."""
    quaran_df = pd.DataFrame([
        {"rejection_reason": "INVALID_AMOUNT"},
        {"rejection_reason": "INVALID_AMOUNT"},
        {"rejection_reason": "MISSING_REQUIRED_FIELD"},
    ])

    # Balanced scenario: 100 raw = 97 valid + 3 quarantined
    balanced_metrics = ReconciliationEngine.generate_metrics(
        run_id="RUN_REC_01",
        started_at="2026-06-01T12:00:00Z",
        completed_at="2026-06-01T12:00:02Z",
        duration_seconds=2.0,
        raw_tx_count=100,
        valid_tx_count=97,
        quarantine_tx_df=quaran_df,
        raw_cust_count=10,
        valid_cust_count=10,
        quarantine_cust_df=pd.DataFrame(),
        features_count=10,
    )
    assert balanced_metrics.is_balanced is True
    assert balanced_metrics.rejection_breakdown == {"INVALID_AMOUNT": 2, "MISSING_REQUIRED_FIELD": 1}
    assert balanced_metrics.throughput_records_per_sec == 50.0

    # Unbalanced scenario (e.g. data dropped silently)
    unbalanced_metrics = ReconciliationEngine.generate_metrics(
        run_id="RUN_REC_02",
        started_at="2026-06-01T12:00:00Z",
        completed_at="2026-06-01T12:00:02Z",
        duration_seconds=2.0,
        raw_tx_count=100,
        valid_tx_count=95,  # 95 + 3 = 98 != 100!
        quarantine_tx_df=quaran_df,
        raw_cust_count=10,
        valid_cust_count=10,
        quarantine_cust_df=pd.DataFrame(),
        features_count=10,
    )
    assert unbalanced_metrics.is_balanced is False


def test_reconciliation_report_formatting_and_saving(temp_dir):
    """Verify formatted report text contains all audit fields and saves atomically."""
    metrics = ReconciliationEngine.generate_metrics(
        run_id="RUN_AUDIT_REPORT",
        started_at="2026-06-01T12:00:00Z",
        completed_at="2026-06-01T12:00:01Z",
        duration_seconds=1.234,
        raw_tx_count=500,
        valid_tx_count=490,
        quarantine_tx_df=pd.DataFrame([{"rejection_reason": "INVALID_AMOUNT"}] * 10),
        raw_cust_count=50,
        valid_cust_count=50,
        quarantine_cust_df=pd.DataFrame(),
        features_count=50,
    )

    report_text = ReconciliationEngine.format_report_text(metrics)
    assert "RUN_AUDIT_REPORT" in report_text
    assert "PASSED (BALANCED)" in report_text
    assert "INVALID_AMOUNT" in report_text
    assert "Processing Throughput" in report_text

    report_file = temp_dir / "audit_report.txt"
    ReconciliationEngine.save_report(report_text, report_file)
    assert report_file.exists()
    assert report_file.read_text(encoding="utf-8") == report_text
