# Data Dictionary

This document provides the formal data dictionary for all data assets, database entities, and feature outputs in the **Reproducible ML Feature Engineering Pipeline**.

---

## 1. `customers` (Dimension Table)

Stores customer demographic and account profile attributes.

| Field Name | Type | Nullable | Primary / Foreign Key | Description | Example |
|---|---|---|---|---|---|
| `customer_id` | TEXT | No | PRIMARY KEY | Unique identifier for the customer | `CUST_000042` |
| `signup_timestamp` | TEXT | No | None | ISO 8601 UTC timestamp of account creation | `2025-03-15T14:30:00Z` |
| `country_code` | TEXT | No | None | ISO 2-letter country code (`US`, `GB`, `IN`, `CA`, `DE`, `AU`, `SG`) | `US` |
| `kyc_status` | TEXT | No | None | KYC verification status (`VERIFIED`, `PENDING`, `REJECTED`) | `VERIFIED` |
| `account_tier` | TEXT | No | None | Customer service tier (`STANDARD`, `PREMIUM`, `VIP`) | `VIP` |
| `is_active` | INTEGER | No | None | Account status flag (`1` for active, `0` for inactive) | `1` |

---

## 2. `transactions` (Fact / Event Log Table)

Stores financial event records with dual timestamps for point-in-time correctness.

| Field Name | Type | Nullable | Primary / Foreign Key | Description | Example |
|---|---|---|---|---|---|
| `transaction_id` | TEXT | No | PRIMARY KEY | Unique identifier for the transaction event | `TX_000000105` |
| `customer_id` | TEXT | No | FOREIGN KEY (`customers.customer_id`) | Reference to purchasing customer | `CUST_000042` |
| `event_timestamp` | TEXT | No | None | ISO 8601 UTC time the transaction took place | `2026-05-10T11:24:00Z` |
| `available_timestamp` | TEXT | No | None | ISO 8601 UTC time record was ingested and accessible in systems | `2026-05-10T11:27:15Z` |
| `amount` | REAL | No | None | Transaction value in USD (strictly positive $> 0.00$) | `49.99` |
| `currency` | TEXT | No | None | Transaction currency code (`USD`) | `USD` |
| `channel` | TEXT | No | None | Transaction payment medium (`MOBILE`, `WEB`, `POS`, `ATM`) | `MOBILE` |
| `category` | TEXT | No | None | Merchant category code (`RETAIL`, `GROCERY`, `TRAVEL`, `ENTERTAINMENT`, `UTILITIES`, `TRANSFER`) | `GROCERY` |
| `status` | TEXT | No | None | Settlement status (`COMPLETED`, `PENDING`, `DECLINED`, `REVERSED`) | `COMPLETED` |

---

## 3. `quarantine_records` (Data Quality Quarantine)

Isolates malformed or non-compliant records with complete audit traceability.

| Field Name | Type | Nullable | Description | Example |
|---|---|---|---|---|
| `quarantine_id` | TEXT | No | Unique tracking ID (`Q_{run_id}_{counter}`) | `Q_RUN_20261029_000001` |
| `run_id` | TEXT | No | Ingestion pipeline run identifier | `RUN_20261029_073507_1b3f86` |
| `source_table` | TEXT | No | Source entity name (`transactions` or `customers`) | `transactions` |
| `raw_record_id` | TEXT | No | Primary key of offending record if present, else `UNKNOWN` | `TX_000000139` |
| `rejection_reason` | TEXT | No | Standardized rejection reason code | `NEGATIVE_TIMESTAMP_LAG` |
| `rejected_at` | TEXT | No | Ingestion UTC timestamp when quarantine occurred | `2026-10-29T07:35:08Z` |
| `raw_payload` | TEXT | No | Complete JSON-serialized raw row payload for audit triage | `{"transaction_id": "TX_...", ...}` |

### Standard Rejection Reasons:
* `MISSING_REQUIRED_FIELD`: Null, empty, or whitespace value in a mandatory column.
* `INVALID_AMOUNT`: Amount is non-numeric, zero, or negative.
* `INVALID_ENUM_VALUE`: Category, channel, tier, or status not in allowed enum list.
* `NEGATIVE_TIMESTAMP_LAG`: Causality violation where `available_timestamp < event_timestamp`.
* `MALFORMED_TIMESTAMP`: Date string cannot be parsed as ISO 8601 UTC.
* `ORPHAN_FOREIGN_KEY`: Transaction references a `customer_id` not found in valid customers.
* `DUPLICATE_PRIMARY_KEY`: Record possesses a primary key that appeared earlier in the batch.
* `SCHEMA_MISMATCH`: File header is missing required columns.

---

## 4. `features` (Customer Point-in-Time Feature Store)

Published customer features computed strictly as-of `observation_timestamp`.

| Field Name | Type | Description |
|---|---|---|
| `customer_id` | TEXT | Customer identifier (Composite PK component 1) |
| `observation_timestamp` | TEXT | Point-in-time prediction timestamp (Composite PK component 2) |
| `country_code` | TEXT | Customer country of residence |
| `account_tier` | TEXT | Customer account tier at observation time |
| `is_active` | INTEGER | Active status flag at observation time |
| `days_since_last_tx` | REAL | Number of days between observation time and most recent transaction (`999.0` if 0 transactions) |
| `tx_count_7d` | INTEGER | Number of completed transactions in last 7 days $[T_{\text{obs}} - 7\text{d}, T_{\text{obs}}]$ |
| `tx_sum_amount_7d` | REAL | Total dollar spend in last 7 days |
| `tx_avg_amount_7d` | REAL | Average transaction amount in last 7 days (`0.0` if count = 0) |
| `tx_max_amount_7d` | REAL | Largest single transaction in last 7 days (`0.0` if count = 0) |
| `tx_std_amount_7d` | REAL | Standard deviation of transaction amounts in last 7 days (`0.0` if count < 2) |
| `tx_count_30d` | INTEGER | Number of completed transactions in last 30 days |
| `tx_sum_amount_30d` | REAL | Total dollar spend in last 30 days |
| `tx_avg_amount_30d` | REAL | Average transaction amount in last 30 days |
| `tx_max_amount_30d` | REAL | Largest single transaction in last 30 days |
| `tx_std_amount_30d` | REAL | Standard deviation of transaction amounts in last 30 days |
| `tx_count_90d` | INTEGER | Number of completed transactions in last 90 days |
| `tx_sum_amount_90d` | REAL | Total dollar spend in last 90 days |
| `tx_avg_amount_90d` | REAL | Average transaction amount in last 90 days |
| `tx_max_amount_90d` | REAL | Largest single transaction in last 90 days |
| `tx_std_amount_90d` | REAL | Standard deviation of transaction amounts in last 90 days |
| `tx_spend_velocity_7d_to_30d` | REAL | Ratio of 7-day spending rate to normalized 30-day weekly spending rate |

---

## 5. `observation_targets` (Isolated ML Target Dataset)

Physically separated ground truth prediction targets computed over the future horizon $[T_{\text{obs}}, T_{\text{obs}} + 30\text{d})$ (inclusive start, exclusive end).

### Population & Correctness Invariants:
* **Complete Customer Coverage**: Every valid customer present in the `customers` dataset receives **exactly one** observation target at $T_{\text{obs}}$, including customers with zero transactions.
* **Target Label Calculation**:
  * `target_label = 1` if customer total transaction spending in the future horizon $> \$1,500.00$ **OR** the customer has $\ge 1$ transaction with `status = 'DECLINED'` during that horizon.
  * `target_label = 0` otherwise (including customers with no transactions).
* **Isolation Guarantee**: Targets are maintained in a physically separated table and CSV file (`data/features/observation_targets.csv`), ensuring zero leakage into historical feature aggregations.

| Field Name | Type | Description |
|---|---|---|
| `observation_id` | TEXT | Unique prediction instance identifier (`OBS_{sequence}`, PK) |
| `customer_id` | TEXT | Customer identifier (FK to `customers.customer_id`, unique per $T_{\text{obs}}$) |
| `observation_timestamp` | TEXT | Timestamp when prediction would have been requested (`2026-06-01T00:00:00Z`) |
| `target_label` | INTEGER | Binary classification outcome (`1` or `0`) |

---

## 6. `pipeline_runs` (Pipeline Telemetry & Audit Log)

Records execution telemetry and reconciliation metrics for each batch run.

| Field Name | Type | Description |
|---|---|---|
| `run_id` | TEXT | Unique pipeline execution identifier (PK) |
| `started_at` | TEXT | Pipeline start timestamp |
| `completed_at` | TEXT | Pipeline completion timestamp |
| `duration_seconds` | REAL | Total execution duration in seconds |
| `input_records_count` | INTEGER | Total raw input records processed |
| `valid_records_count` | INTEGER | Total records validated into curated tables |
| `quarantined_records_count` | INTEGER | Total defective records routed to quarantine |
| `features_generated_count` | INTEGER | Total customer feature vectors published |
| `status` | TEXT | Execution status (`STARTED`, `SUCCESS`, `FAILED`) |
| `error_message` | TEXT | Exception traceback if execution failed (nullable) |
