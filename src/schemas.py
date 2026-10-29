"""Schema definitions, allowed values, constraints, and rejection error codes."""

from enum import Enum
from typing import Dict, List, Set


class RejectionReason(str, Enum):
    """Standardized reasons for record rejection and quarantine."""
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_DATA_TYPE = "INVALID_DATA_TYPE"
    DUPLICATE_PRIMARY_KEY = "DUPLICATE_PRIMARY_KEY"
    ORPHAN_FOREIGN_KEY = "ORPHAN_FOREIGN_KEY"
    INVALID_AMOUNT = "INVALID_AMOUNT"
    NEGATIVE_TIMESTAMP_LAG = "NEGATIVE_TIMESTAMP_LAG"
    INVALID_ENUM_VALUE = "INVALID_ENUM_VALUE"
    MALFORMED_TIMESTAMP = "MALFORMED_TIMESTAMP"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"


class CustomerSchema:
    """Schema and rules for Customer entity."""
    REQUIRED_FIELDS: List[str] = [
        "customer_id",
        "signup_timestamp",
        "country_code",
        "kyc_status",
        "account_tier",
        "is_active",
    ]
    
    ALLOWED_KYC_STATUS: Set[str] = {"VERIFIED", "PENDING", "REJECTED"}
    ALLOWED_TIERS: Set[str] = {"STANDARD", "PREMIUM", "VIP"}
    
    EXPECTED_DTYPES: Dict[str, str] = {
        "customer_id": "string",
        "signup_timestamp": "string",  # ISO 8601
        "country_code": "string",
        "kyc_status": "string",
        "account_tier": "string",
        "is_active": "bool",
    }


class TransactionSchema:
    """Schema and rules for Transaction entity."""
    REQUIRED_FIELDS: List[str] = [
        "transaction_id",
        "customer_id",
        "event_timestamp",
        "available_timestamp",
        "amount",
        "currency",
        "channel",
        "category",
        "status",
    ]
    
    ALLOWED_CHANNELS: Set[str] = {"MOBILE", "WEB", "POS", "ATM"}
    ALLOWED_CATEGORIES: Set[str] = {
        "RETAIL",
        "GROCERY",
        "TRAVEL",
        "ENTERTAINMENT",
        "UTILITIES",
        "TRANSFER",
    }
    ALLOWED_STATUSES: Set[str] = {"COMPLETED", "PENDING", "DECLINED", "REVERSED"}
    
    EXPECTED_DTYPES: Dict[str, str] = {
        "transaction_id": "string",
        "customer_id": "string",
        "event_timestamp": "string",  # ISO 8601
        "available_timestamp": "string",  # ISO 8601
        "amount": "float64",
        "currency": "string",
        "channel": "string",
        "category": "string",
        "status": "string",
    }


class ObservationTargetSchema:
    """Schema for ground truth ML targets."""
    REQUIRED_FIELDS: List[str] = [
        "observation_id",
        "customer_id",
        "observation_timestamp",
        "target_label",
    ]
