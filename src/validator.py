"""Data quality validation, schema enforcement, and quarantine routing."""

from datetime import datetime, timezone
import json
import logging
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from src.schemas import (
    CustomerSchema,
    RejectionReason,
    TransactionSchema,
)

logger = logging.getLogger("transactional_pipeline.validator")


class DataQualityValidator:
    """Validates raw customer and transaction data and isolates defective records into quarantine."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._quarantine_counter = 0

    def validate_customers(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Validate customer records against CustomerSchema.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (valid_customers, quarantined_customers)
        """
        if df.empty:
            logger.warning("Empty customer DataFrame received.")
            return df.copy(), self._empty_quarantine_df()

        # Check required columns presence
        missing_cols = set(CustomerSchema.REQUIRED_FIELDS) - set(df.columns)
        if missing_cols:
            logger.error(f"Customer dataset missing required columns: {missing_cols}")
            # Quarantine all rows due to schema mismatch
            return pd.DataFrame(columns=CustomerSchema.REQUIRED_FIELDS), self._quarantine_all_rows(
                df, "customers", RejectionReason.SCHEMA_MISMATCH, "customer_id"
            )

        valid_mask = pd.Series(True, index=df.index)
        rejection_reasons = pd.Series(None, index=df.index, dtype="object")

        # 1. Null check on required fields
        for col in CustomerSchema.REQUIRED_FIELDS:
            null_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            rejection_reasons[valid_mask & null_mask] = RejectionReason.MISSING_REQUIRED_FIELD.value
            valid_mask = valid_mask & (~null_mask)

        # 2. Check KYC Status and Account Tier enums
        invalid_kyc = ~df["kyc_status"].astype(str).isin(CustomerSchema.ALLOWED_KYC_STATUS)
        rejection_reasons[valid_mask & invalid_kyc] = RejectionReason.INVALID_ENUM_VALUE.value
        valid_mask = valid_mask & (~invalid_kyc)

        invalid_tier = ~df["account_tier"].astype(str).isin(CustomerSchema.ALLOWED_TIERS)
        rejection_reasons[valid_mask & invalid_tier] = RejectionReason.INVALID_ENUM_VALUE.value
        valid_mask = valid_mask & (~invalid_tier)

        # 3. Check Duplicate Primary Key
        dup_mask = df.duplicated(subset=["customer_id"], keep="first")
        rejection_reasons[valid_mask & dup_mask] = RejectionReason.DUPLICATE_PRIMARY_KEY.value
        valid_mask = valid_mask & (~dup_mask)

        # Separate valid and quarantined
        valid_df = df[valid_mask].copy()
        invalid_df = df[~valid_mask].copy()
        invalid_reasons = rejection_reasons[~valid_mask]

        quarantine_df = self._build_quarantine_df(
            invalid_df, invalid_reasons, "customers", "customer_id"
        )

        logger.info(
            f"Validated customers: {len(valid_df)} valid, {len(quarantine_df)} quarantined."
        )
        return valid_df, quarantine_df

    def validate_transactions(
        self, df: pd.DataFrame, valid_customer_ids: Set[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Validate transaction records against TransactionSchema and foreign keys.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (valid_transactions, quarantined_transactions)
        """
        if df.empty:
            logger.warning("Empty transaction DataFrame received.")
            return df.copy(), self._empty_quarantine_df()

        # Check required columns
        missing_cols = set(TransactionSchema.REQUIRED_FIELDS) - set(df.columns)
        if missing_cols:
            logger.error(f"Transaction dataset missing required columns: {missing_cols}")
            return pd.DataFrame(columns=TransactionSchema.REQUIRED_FIELDS), self._quarantine_all_rows(
                df, "transactions", RejectionReason.SCHEMA_MISMATCH, "transaction_id"
            )

        valid_mask = pd.Series(True, index=df.index)
        rejection_reasons = pd.Series(None, index=df.index, dtype="object")

        # 1. Null check on required fields
        for col in TransactionSchema.REQUIRED_FIELDS:
            null_mask = df[col].isna() | (df[col].astype(str).str.strip().isin(["", "nan", "None", "null"]))
            newly_invalid = valid_mask & null_mask
            rejection_reasons[newly_invalid] = RejectionReason.MISSING_REQUIRED_FIELD.value
            valid_mask = valid_mask & (~null_mask)

        # 2. Amount numeric and strictly positive (> 0.00)
        numeric_amounts = pd.to_numeric(df["amount"], errors="coerce")
        bad_amount = numeric_amounts.isna() | (numeric_amounts <= 0.0)
        newly_invalid = valid_mask & bad_amount
        rejection_reasons[newly_invalid] = RejectionReason.INVALID_AMOUNT.value
        valid_mask = valid_mask & (~bad_amount)

        # 3. Enum validations (channel, status, category)
        bad_channel = ~df["channel"].astype(str).isin(TransactionSchema.ALLOWED_CHANNELS)
        newly_invalid = valid_mask & bad_channel
        rejection_reasons[newly_invalid] = RejectionReason.INVALID_ENUM_VALUE.value
        valid_mask = valid_mask & (~bad_channel)

        bad_status = ~df["status"].astype(str).isin(TransactionSchema.ALLOWED_STATUSES)
        newly_invalid = valid_mask & bad_status
        rejection_reasons[newly_invalid] = RejectionReason.INVALID_ENUM_VALUE.value
        valid_mask = valid_mask & (~bad_status)

        bad_category = ~df["category"].astype(str).isin(TransactionSchema.ALLOWED_CATEGORIES)
        newly_invalid = valid_mask & bad_category
        rejection_reasons[newly_invalid] = RejectionReason.INVALID_ENUM_VALUE.value
        valid_mask = valid_mask & (~bad_category)

        # 4. Timestamps valid ISO and Causality Check (available_timestamp >= event_timestamp)
        ev_dt = pd.to_datetime(df["event_timestamp"], format="ISO8601", errors="coerce", utc=True)
        av_dt = pd.to_datetime(df["available_timestamp"], format="ISO8601", errors="coerce", utc=True)

        bad_ts = ev_dt.isna() | av_dt.isna()
        newly_invalid = valid_mask & bad_ts
        rejection_reasons[newly_invalid] = RejectionReason.MALFORMED_TIMESTAMP.value
        valid_mask = valid_mask & (~bad_ts)

        # Causality check: available must not precede event
        negative_lag = av_dt < ev_dt
        newly_invalid = valid_mask & negative_lag
        rejection_reasons[newly_invalid] = RejectionReason.NEGATIVE_TIMESTAMP_LAG.value
        valid_mask = valid_mask & (~negative_lag)

        # 5. Foreign Key check: customer_id must exist in valid_customer_ids
        is_orphan = ~df["customer_id"].astype(str).isin(valid_customer_ids)
        newly_invalid = valid_mask & is_orphan
        rejection_reasons[newly_invalid] = RejectionReason.ORPHAN_FOREIGN_KEY.value
        valid_mask = valid_mask & (~is_orphan)

        # 6. Duplicate Primary Key check (transaction_id uniqueness)
        is_duplicate = df.duplicated(subset=["transaction_id"], keep="first")
        newly_invalid = valid_mask & is_duplicate
        rejection_reasons[newly_invalid] = RejectionReason.DUPLICATE_PRIMARY_KEY.value
        valid_mask = valid_mask & (~is_duplicate)

        # Build output frames
        valid_df = df[valid_mask].copy()
        # Ensure proper types on valid_df
        valid_df["amount"] = pd.to_numeric(valid_df["amount"])
        
        invalid_df = df[~valid_mask].copy()
        invalid_reasons = rejection_reasons[~valid_mask]

        quarantine_df = self._build_quarantine_df(
            invalid_df, invalid_reasons, "transactions", "transaction_id"
        )

        logger.info(
            f"Validated transactions: {len(valid_df)} valid, {len(quarantine_df)} quarantined."
        )
        return valid_df, quarantine_df

    def _build_quarantine_df(
        self,
        invalid_df: pd.DataFrame,
        rejection_reasons: pd.Series,
        source_table: str,
        id_column: str,
    ) -> pd.DataFrame:
        """Construct standard quarantine DataFrame with traceability fields."""
        if invalid_df.empty:
            return self._empty_quarantine_df()

        records = []
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for idx, row in invalid_df.iterrows():
            self._quarantine_counter += 1
            raw_id = str(row[id_column]) if id_column in row and pd.notna(row[id_column]) else "UNKNOWN"
            reason = rejection_reasons.loc[idx] or RejectionReason.INVALID_DATA_TYPE.value
            
            # Serialize payload to JSON for triage
            payload = json.dumps(row.to_dict(), default=str)
            records.append({
                "quarantine_id": f"Q_{self.run_id}_{self._quarantine_counter:06d}",
                "run_id": self.run_id,
                "source_table": source_table,
                "raw_record_id": raw_id,
                "rejection_reason": reason,
                "rejected_at": now_str,
                "raw_payload": payload,
            })

        return pd.DataFrame(records)

    def _quarantine_all_rows(
        self, df: pd.DataFrame, source_table: str, reason: RejectionReason, id_column: str
    ) -> pd.DataFrame:
        reasons = pd.Series(reason.value, index=df.index)
        return self._build_quarantine_df(df, reasons, source_table, id_column)

    def _empty_quarantine_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "quarantine_id",
                "run_id",
                "source_table",
                "raw_record_id",
                "rejection_reason",
                "rejected_at",
                "raw_payload",
            ]
        )
