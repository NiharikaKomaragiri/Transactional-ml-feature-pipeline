# Reproducible ML Feature Engineering Pipeline for Transactional Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-34%20passed%20(1.82s)-brightgreen.svg)]()
[![Database](https://img.shields.io/badge/database-SQLite%20WAL-lightgrey.svg)]()

A reproducible, production-grade Python data engineering pipeline that processes transactional event data, validates data quality and schema constraints, quarantines malformed records with traceable rejection reasons, guards against target and lookahead leakage via dual-timestamp point-in-time (PIT) filtering, ensures idempotency across executions, and publishes machine learning features.

Developed as an individual technical assessment project for **YottaFlex AI**.

---

## 1. Business Objective

Machine learning models trained on transactional data frequently fail in production due to two critical data engineering challenges:
1. **Target and Lookahead Leakage**: Incorporating information from future events or delayed pipeline data that would not have been available at the simulated prediction moment.
2. **Data Ingestion Fragility**: Pipelines silently dropping defective records without audit traceability, creating duplicate records on retries, or corrupting historical feature tables.

This project delivers a complete, modular, and self-contained solution built entirely with standard Python, Pandas, and SQLite to solve these challenges without requiring external or paid cloud infrastructure.

---

## 2. Architecture Overview

```
                      +----------------------------------+
                      |   Synthetic Data Generator       |
                      |  (Fixed Random Seed / Reproducible)
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |     Raw Data Ingestion (CSV)     |
                      |   (customers.csv, transactions)  |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      | Data Quality & Schema Validator  |
                      +----------------------------------+
                                /              \
                     (Valid)   /                \ (Violations)
                              v                  v
         +--------------------------+   +---------------------------------+
         |   Curated SQLite Store   |   |   Quarantine Audit Store        |
         | (Atomic Upsert / WAL)    |   |  (Traceable Rejection Codes)    |
         +--------------------------+   +---------------------------------+
                      |
                      v
         +--------------------------+
         |   Point-in-Time Engine   |
         | (event_ts & available_ts)|
         +--------------------------+
                      |
                      v
         +--------------------------+
         | Vectorized Feature Eng.  |
         | (7d, 30d, 90d Windows)   |
         +--------------------------+
                  /          \
                 v            v
+------------------------+  +----------------------------+
| Published Feature Store|  | Isolated ML Target Store   |
| (customer_features.csv)|  | (observation_targets.csv)  |
+------------------------+  +----------------------------+
                 \            /
                  v          v
         +--------------------------+
         |  Reconciliation Engine   |
         |  Raw = Valid + Quarantined
         +--------------------------+
```

---

## 3. Technology Stack

* **Language**: Python 3.12 (compatible with Python 3.10+)
* **Data Manipulation**: Pandas (vectorized grouped aggregations and boolean filtering)
* **Storage Engine**: SQLite with Write-Ahead Logging (WAL) and B-Tree indexes
* **Testing Framework**: `pytest` (34 automated unit, integration, and performance tests)
* **File Formats**: Standard CSV with atomic temporary-staging and replacement
* **Telemetry**: Python standard library `logging` and structured audit table (`pipeline_runs`)

---

## 4. Assessment Requirements Compliance Matrix

| Requirement # | Assessment Requirement | Status | Implementation Details |
|---|---|---|---|
| **1** | Reproducible synthetic data using fixed random seed | **PASSED** | Implemented in [`src/generator.py`](src/generator.py); verified in [`tests/test_generator.py`](tests/test_generator.py). |
| **2** | Support small sample & $\ge 100,000$ transactions | **PASSED** | Validated on 100,000 transactions and 2,500 customers (full pipeline completed in 2.883s in verified local test run). |
| **3** | Validate schemas, types, required fields, PKs, duplicates, and FKs | **PASSED** | Implemented in [`src/validator.py`](src/validator.py); verified in [`tests/test_validator.py`](tests/test_validator.py). |
| **4** | Quarantine invalid records with traceable rejection reasons | **PASSED** | 8 standardized error codes logged to SQLite and [`data/quarantine/quarantined_records.csv`](data/quarantine/quarantined_records.csv). |
| **5** | Customer features over configurable time windows (7d, 30d, 90d) | **PASSED** | Aggregations (count, sum, mean, max, std, recency, velocity) in [`src/features.py`](src/features.py). |
| **6** | Point-in-time correctness using event and available timestamps | **PASSED** | Dual-timestamp filter engine in [`src/point_in_time.py`](src/point_in_time.py); mitigates lookahead and ingestion lag leakage. |
| **7** | Keep future target labels separate from feature inputs | **PASSED** | Target dataset saved to distinct table and file [`data/features/observation_targets.csv`](data/features/observation_targets.csv). |
| **8** | Make repeated executions idempotent | **PASSED** | SQLite `INSERT OR REPLACE` and atomic `.tmp` file swaps; verified in [`tests/test_idempotency.py`](tests/test_idempotency.py). |
| **9** | Record run IDs, timestamps, counts, duration, and failures | **PASSED** | Tracked in SQLite table `pipeline_runs` and logged during every pipeline execution. |
| **10** | Support reconciliation reports and automated tests | **PASSED** | Generates executive report verifying $N_{\text{raw}} = N_{\text{valid}} + N_{\text{quarantined}}$. |
| **11** | Test empty inputs, malformed records, nulls, duplicates, late events | **PASSED** | Comprehensive 34-test suite covering edge cases in [`tests/`](tests/). |
| **12** | Measure execution time & document performance optimization | **PASSED** | Vectorized groupby optimization documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); feature computation completed in 0.188s in local test run. |
| **13** | Documentation: README, data dictionary, architecture, decisions, prompt history | **PASSED** | Complete documentation suite in [`docs/`](docs/). |

---

## 5. Project Directory Structure

```
transactional-ml-feature-pipeline/
│
├── data/                                # Local file-based data storage
│   ├── raw/                             # Raw generated input CSVs
│   │   ├── customers.csv
│   │   └── transactions.csv
│   ├── quarantine/                      # Quarantined bad records
│   │   └── quarantined_records.csv
│   ├── features/                        # Published feature datasets
│   │   ├── customer_features.csv
│   │   └── observation_targets.csv
│   └── pipeline.db                      # SQLite database (curated tables, features, audit logs)
│
├── src/                                 # Production pipeline source code
│   ├── __init__.py
│   ├── config.py                        # Pipeline configuration & constants
│   ├── generator.py                     # Synthetic data generator (seeded, configurable scale)
│   ├── schemas.py                       # Data contracts, schema definitions, rejection codes
│   ├── validator.py                     # Data quality validator & quarantine engine
│   ├── point_in_time.py                 # PIT temporal cutoff logic & observation alignment
│   ├── features.py                      # Vectorized feature engineering (7d, 30d, 90d windows)
│   ├── database.py                      # SQLite manager, ACID transactions, idempotency
│   ├── reconciliation.py                # Audit reporting, reconciliation assertions, metrics
│   └── pipeline.py                      # Main pipeline orchestrator CLI
│
├── tests/                               # Comprehensive automated test suite
│   ├── __init__.py
│   ├── conftest.py                      # Pytest fixtures and test database setup
│   ├── test_generator.py                # Reproducibility & seed verification
│   ├── test_validator.py                # Schema, nulls, duplicates, FK quarantine tests
│   ├── test_point_in_time.py            # Temporal cutoff, availability lag, late-arriving events
│   ├── test_features.py                 # Window calculation math correctness & feature consistency
│   ├── test_targets.py                  # Complete customer coverage, boundaries & target isolation
│   ├── test_idempotency.py              # Repeated execution stability & atomic rollback tests
│   ├── test_reconciliation.py           # Reconciliation balance checks (raw = valid + quarantined)
│   └── test_performance.py             # Benchmark 100k records execution time & optimization
│
├── docs/                                # Documentation and sample outputs
│   ├── DATA_DICTIONARY.md               # Detailed schema, fields, types, constraints
│   ├── ARCHITECTURE.md                  # In-depth architectural & design decisions
│   ├── AI_PROMPT_HISTORY.md             # Complete record of engineering prompts used
│   └── sample_reconciliation_report.txt # Sample generated execution report
│
├── README.md                            # Project overview, setup guide, execution instructions
├── requirements.txt                     # Dependencies
└── pyproject.toml                       # Build & pytest config
```

---

## 6. Installation & Quickstart

### Step 1: Clone and Set Up Virtual Environment

```bash
# Clone the repository (replace with your repository URL)
git clone <REPOSITORY_URL_PLACEHOLDER>
cd transactional-ml-feature-pipeline

# Create virtual environment
python -m venv .venv

# Activate virtual environment:
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the Pipeline

To execute the pipeline with a dataset of **100,000 transactions and 2,500 customers**:

```bash
python -m src.pipeline --generate --customers 2500 --transactions 100000 --seed 42
```

To run with a **small sample dataset** (e.g., 50 customers, 500 transactions):

```bash
python -m src.pipeline --generate --customers 50 --transactions 500 --seed 42
```

### Step 3: Run the Automated Test Suite

Run the full pytest suite (34 tests):

```bash
pytest -v
```

> **Verified Test Run**: All 34 tests passed in **1.82 seconds** on local test execution.

---

## 7. Sample Output & Reconciliation Report

When executing the pipeline, an audit report is emitted to the console and saved to [`docs/sample_reconciliation_report.txt`](docs/sample_reconciliation_report.txt). Below is the report from the latest verified run:

```text
================================================================================
           TRANSACTIONAL ML FEATURE PIPELINE - RECONCILIATION REPORT
================================================================================
Execution Metadata:
  - Run ID                    : RUN_20261029_092115_54cf85
  - Status                    : SUCCESS
  - Execution Started         : 2026-10-29T09:21:15Z
  - Execution Completed       : 2026-10-29T09:21:18Z
  - Overall Pipeline Duration : 2.883 seconds
  - Feature Computation Time  : 0.188 seconds
  - Processing Throughput     : ~34,686 records/second (overall pipeline)

Customer Reconciliation:
  - Raw Customer Records      : 2500
  - Valid Customer Records    : 2500
  - Quarantined Customers     : 0
  - Balance Invariant Check   : 2500 == 2500 + 0

Transaction Reconciliation:
  - Raw Transaction Records   : 100000
  - Valid Curated Records     : 98000
  - Quarantined Records       : 2000
  - Balance Invariant Check   : 100000 == 98000 + 2000
  - Reconciliation Result     : PASSED (BALANCED)

Quarantine Rejection Traceability Breakdown:
  - MISSING_REQUIRED_FIELD    :    334 records ( 0.33%)
  - INVALID_AMOUNT            :    334 records ( 0.33%)
  - INVALID_ENUM_VALUE        :    333 records ( 0.33%)
  - ORPHAN_FOREIGN_KEY        :    333 records ( 0.33%)
  - DUPLICATE_PRIMARY_KEY     :    333 records ( 0.33%)
  - NEGATIVE_TIMESTAMP_LAG    :    333 records ( 0.33%)

Feature & Target Engineering Output:
  - ML Feature Records        : 2500
  - Observation Target Records: 2500
  - Target Population Match   : PASSED (2500 == 2500)
  - Target Leakage Check      : PASSED (Forward target store isolated)
================================================================================
```

*Note: Execution durations represent local measured execution times on 100,000 transactions and 2,500 customers; performance may vary depending on host hardware, CPU frequency, and disk I/O.*

---

## 8. Key Engineering Safeguards & Optimizations

### 1. Dual-Timestamp Temporal Cutoff (Leakage Mitigation)
Every transaction record tracks two distinct timestamps:
* `event_timestamp`: The moment the financial transaction occurred.
* `available_timestamp`: The moment the record was committed into the analytical database ($T_{\text{available}} \ge T_{\text{event}}$).

When computing features as-of observation timestamp $T_{\text{obs}}$:
$$\text{Eligible} = (E.\text{event\_timestamp} \le T_{\text{obs}}) \;\land\; (E.\text{available\_timestamp} \le T_{\text{obs}})$$
This safeguard prevents ingestion lag leakage by ensuring that transactions delayed upstream (e.g., due to batch or network sync latency) are excluded if they were not yet queryable at $T_{\text{obs}}$.

### 2. Physical Separation of ML Targets
* Prediction target labels are defined over a forward-looking horizon $[T_{\text{obs}}, T_{\text{obs}} + 30\text{d})$ (inclusive start, exclusive end).
* Target generation is physically decoupled from feature engineering: target labels are computed in a separate routine and saved to [`data/features/observation_targets.csv`](data/features/observation_targets.csv), preventing target values from contaminating feature inputs.
* Every valid customer receives exactly one observation target, including customers with zero transactions (`target_label = 0`).

### 3. High-Performance Vectorized Aggregations
* Replaced iterative row-by-row scanning ($O(N \times M)$) with vectorized Pandas datetime filtering, interval slicing, and single-pass grouped aggregation (`groupby('customer_id')['amount'].agg(...)`).
* In verified test executions on 100,000 transactions and 2,500 customers:
  * **Feature computation duration**: **0.188 seconds** across 3 rolling windows (7d, 30d, 90d).
  * **Overall pipeline duration**: **2.883 seconds** (including data loading, validation, quarantine routing, SQLite persistence, feature engineering, target generation, and audit reporting).
* Execution times are documented to illustrate the performance benefits of vectorized array operations over naive iteration rather than as universal hardware-independent benchmarks.

### 4. Traceable Quarantine Routing & Mathematical Reconciliation
* Non-compliant records are routed to [`data/quarantine/quarantined_records.csv`](data/quarantine/quarantined_records.csv) with standard rejection error codes (`MISSING_REQUIRED_FIELD`, `INVALID_AMOUNT`, `INVALID_ENUM_VALUE`, `NEGATIVE_TIMESTAMP_LAG`, `ORPHAN_FOREIGN_KEY`, `DUPLICATE_PRIMARY_KEY`, `MALFORMED_TIMESTAMP`, `SCHEMA_MISMATCH`) and full JSON-serialized raw payloads.
* The reconciliation engine programmatically enforces mathematical balance:
  $$N_{\text{raw}} = N_{\text{valid}} + N_{\text{quarantined}}$$

---

## 9. Additional Documentation

* **[Data Dictionary](docs/DATA_DICTIONARY.md)**: Detailed schema specifications, data types, nullability, constraints, and rejection reason codes.
* **[Architecture & Design Decisions](docs/ARCHITECTURE.md)**: In-depth technical rationale, trade-offs, and educational fresher guide.
* **[AI Prompt History](docs/AI_PROMPT_HISTORY.md)**: Chronological record of prompts and design iterations.
