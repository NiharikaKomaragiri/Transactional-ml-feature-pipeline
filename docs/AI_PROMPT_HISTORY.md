# AI Prompt History & Engineering Log

This document records the AI-assisted development workflow, architectural decisions, prompt summaries, and verification steps utilized during the design, implementation, and refinement of the **Reproducible ML Feature Engineering Pipeline for Transactional Data**.

---

## 1. Transparency & Record-Keeping Methodology

To provide clear visibility for technical assessment review:
* **Prompt Records**: All prompts documented below are categorized and labeled. Where condensed or reconstructed from developer instructions and multi-step interactions, they are explicitly designated as **Reconstructed Prompt Summary**.
* **Scope of Assistance**: AI tools (Google Antigravity agentic coding environment using Gemini models) were utilized for architectural formulation, code generation, debugging platform-specific issues (e.g., Windows SQLite file locking), test suite creation, and documentation refinement.
* **Human-in-the-Loop Verification**: All AI-generated architectures, schemas, transformations, and fixes were validated through automated test execution (34 passing pytest tests), mathematical reconciliation balance checks, and inspection of generated datasets.

---

## 2. Chronological Prompt & Development History

### Phase 1: Project Setup, Requirements Analysis & Architectural Design

* **Prompt Category**: Reconstructed Prompt Summary
* **Developer Prompt**:
  > *"I am developing an individual technical assessment project for YottaFlex AI.  
  > Project title: Reproducible ML Feature Engineering Pipeline for Transactional Data.  
  > Business objective: Build a reproducible Python data engineering pipeline that processes transactional data, validates data quality, prevents target leakage, applies point-in-time feature calculations, and publishes reliable machine learning features.  
  > Tech stack: Python, Pandas, SQLite, pytest, CSV, Python logging.  
  > Requirements: (1) Reproducible synthetic data with fixed seed; (2) Small sample and >= 100k scale; (3) Schema, type, null, PK, duplicate, and FK validation; (4) Quarantine invalid records with traceable reasons; (5) Features over configurable windows; (6) Point-in-time correctness with event and available timestamps; (7) Separate future targets from features; (8) Idempotency; (9) Execution telemetry; (10) Reconciliation reports; (11) Edge case testing; (12) Performance optimization; (13) Documentation.  
  > Before writing any implementation: Propose architecture, file structure, data model, leakage risks, PIT rules, idempotency/recovery, and test plan mapped to requirements."*
* **AI Action & Design**:
  * Inspected local workspace and environment (Python 3.12, Pandas 3.1.0, pytest 9.2.0, built-in SQLite).
  * Formulated an Extract-Validate-Transform-Load (EVTL) modular architecture.
  * Formalized the dual-timestamp Point-in-Time eligibility criteria ($T_{\text{event}} \le T_{\text{obs}} \land T_{\text{available}} \le T_{\text{obs}}$) to guard against temporal lookahead and ingestion lag leakage.
  * Defined data contracts, quarantine schema with standardized rejection error codes, SQLite ACID transaction strategy, and mathematical reconciliation invariant ($N_{\text{raw}} = N_{\text{valid}} + N_{\text{quarantined}}$).
  * Created `implementation_plan.md` artifact.
* **Verification**: Developer reviewed and formally approved the implementation plan prior to any code authoring.

---

### Phase 2: Core Pipeline Implementation & Vectorized Optimization

* **Prompt Category**: Reconstructed Prompt Summary
* **Developer Prompt**: Developer instruction acknowledging review and approval of `implementation_plan.md`, directing commencement of pipeline implementation.
* **AI Action & Implementation**:
  * **Configuration & Schemas** (`src/config.py`, `src/schemas.py`): Defined `PipelineConfig` dataclass, standardized `RejectionReason` enums, and entity contracts for customers, transactions, features, and targets.
  * **Synthetic Data Generator** (`src/generator.py`): Implemented deterministic generation using `numpy.random.default_rng(seed)` with realistic log-normal amounts, Zipf-distributed transactions per customer, simulated ingestion lag, and configurable anomaly injection.
  * **Validator & Quarantine Engine** (`src/validator.py`): Implemented strict data-type checks, required field null checks, positive amount assertions ($> 0$), allowed enum verification, causality validation ($T_{\text{available}} \ge T_{\text{event}}$), FK referential checks against customer records, and PK uniqueness deduplication. Offending records are routed to `quarantine_records` with raw JSON payloads and error codes.
  * **Point-in-Time Engine** (`src/point_in_time.py`): Implemented dual-timestamp cutoff logic filtering out future events and delayed pipeline records as-of $T_{\text{obs}}$.
  * **Vectorized Feature Engineering** (`src/features.py`): Replaced iterative row-by-row loops with vectorized Pandas datetime masks and single-pass grouped aggregation (`groupby('customer_id')['amount'].agg(...)`) across 7-day, 30-day, and 90-day windows (count, sum, mean, max, std, recency, velocity).
  * **SQLite Database Manager** (`src/database.py`): Configured SQLite in Write-Ahead Logging (WAL) mode, added multi-column B-tree indexes, created table DDLs, and implemented idempotent upserts via `INSERT OR REPLACE INTO`.
  * **Reconciliation Engine** (`src/reconciliation.py`): Implemented balance verification ($N_{\text{raw}} == N_{\text{valid}} + N_{\text{quarantined}}$) and formatted executive audit reports.
  * **Pipeline Orchestrator** (`src/pipeline.py`): Built CLI-driven end-to-end pipeline runner coordinating all modules and logging run metadata to `pipeline_runs`.
* **Verification**: Ran initial pipeline on synthetic data; verified that curated tables, quarantine outputs, and feature CSVs were created.

---

### Phase 3: Automated Testing & Environment Debugging

* **Prompt Category**: Reconstructed Prompt Summary
* **Developer Instruction**:
  > *"Run the automated test suite and address any platform or library warnings/failures."*
* **AI Action & Debugging**:
  * Authored test modules: `test_generator.py`, `test_validator.py`, `test_point_in_time.py`, `test_features.py`, `test_idempotency.py`, `test_reconciliation.py`, and `test_performance.py` (26 initial tests).
  * *Issue Identified*: On Windows, SQLite database file locks held in temporary directories caused `PermissionError: [WinError 32]` during pytest teardown.
  * *Resolution*: Configured `tempfile.TemporaryDirectory(ignore_cleanup_errors=True)` in `tests/conftest.py` and ensured explicit `.close()` calls on SQLite connections in `tests/test_idempotency.py`.
  * *Pandas 3.1+ Warning*: Replaced deprecated `inplace=True` in `DataFrame.rename()` with explicit assignment in `src/features.py`, and supplied `format="ISO8601"` in `src/validator.py`.
* **Verification**: Executed `python -m pytest -v`, resulting in **all 26 initial tests passing in 1.74s**.

---

### Phase 4: Target Generation Population Coverage Refinement

* **Prompt Category**: Reconstructed Prompt Summary
* **Developer Prompt**:
  > *"Review my existing Transactional ML Feature Pipeline project and fix the target-generation issue described below. Modify the existing code directly rather than creating a new project.  
  > 1. Fix target generation so that every valid customer in the customer dataset receives exactly one observation target, including customers who have no transactions.  
  > 2. In src/generator.py, review generate_observation_targets(). Currently, it derives the customer population from transaction records. Update the function to use the complete customer dataset as the source of customer IDs.  
  > 3. Update all affected function calls in src/pipeline.py and any other files so the changes are consistent throughout the project. Preserve backward compatibility where practical.  
  > 4. Preserve existing target definition: Observation timestamp: 2026-06-01T00:00:00Z, Horizon: 30 days, Target label = 1 if spend > $1,500 OR >= 1 declined transaction in future horizon, else 0. Inclusive start, exclusive end.  
  > 5. Ensure future transactions are used only for generating target labels, never in historical features.  
  > 6. Ensure invalid, quarantined, duplicate, and orphan transactions do not incorrectly influence target labels.  
  > 7. Preserve point-in-time correctness for historical features.  
  > 8. Add or update automated tests to verify: customer coverage, 0-tx customers, threshold rules, declined transactions, horizon boundaries, orphan/invalid isolation, leakage absence, idempotency.  
  > 9. Run complete test suite and pipeline on 100k transactions; verify results."*
* **AI Action & Implementation**:
  * *Root Cause Analysis*: Diagnosed that `generate_observation_targets()` previously populated customer IDs via `set(transactions_df["customer_id"].dropna())`, omitting customers who had zero transactions or whose transactions were quarantined (yielding only 1,575 targets for 2,500 customers).
  * *Code Fix*:
    * Updated `generate_observation_targets()` in `src/generator.py` to accept `customers_df`, generating exactly one target observation per valid customer (`target_label = 0` for 0-transaction customers). Maintained backward compatibility if `customers_df` is omitted.
    * Enforced horizon boundaries $[T_{\text{obs}}, T_{\text{obs}} + 30\text{d})$ (inclusive start, exclusive end).
    * Sanitized input transactions to ignore orphan customer IDs, negative amounts, and duplicates.
    * Updated `src/pipeline.py` to pass `customers_df=valid_cust_df` and log `targets_count`.
    * Added `idx_targets_customer_obs` UNIQUE index in SQLite `DatabaseManager` ensuring uniqueness on `(customer_id, observation_timestamp)`.
    * Added `targets_generated_count` to `ReconciliationMetrics` and updated executive reporting in `src/reconciliation.py`.
    * Created dedicated test suite `tests/test_targets.py` with 8 comprehensive automated tests covering all required edge cases.
* **Verification**:
  * Executed `python -m pytest -v`: **All 34 tests passed in 1.82 seconds**.
  * Executed full pipeline on 100,000 transactions and 2,500 customers: Generated **2,500 observation targets (100% customer match)** with 0 duplicates, 0 nulls, and clean binary distribution (2,407 negative, 93 positive).

---

### Phase 5: Documentation Review & Repository Finalization

* **Prompt Category**: Reconstructed Prompt Summary
* **Developer Prompts**:
  1. *"Review and improve the existing README.md... Update sample execution results to reflect latest verified run... Clearly distinguish full pipeline duration from feature-computation duration... Replace unsupported absolute claims... Keep clone command as placeholder... Confirm no source code modified."*
  2. *"did .gitignore, or open the file and verify that it excludes at least: .venv/, __pycache__/, .pytest_cache/, \*.py[cod], .env"*
* **AI Action & Refinements**:
  * Updated `README.md` to reflect verified benchmark figures (overall pipeline duration: 2.883s, feature computation: 0.188s, 34 tests passing in 1.82s).
  * Nuanced performance claims by noting that execution durations reflect local test runs and depend on hardware.
  * Replaced absolute phrasing with technically precise descriptions of implemented safeguards (e.g., dual-timestamp filtering mitigating lookahead and ingestion lag leakage).
  * Provided explicit virtual-environment activation commands for PowerShell, Command Prompt, and Unix shells.
  * Inspected and confirmed that `.gitignore` contains exclusions for `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.py[cod]`, `.env`, logs, and temporary SQLite files.
* **Verification**:
  * Confirmed that zero source-code files in `src/` or `tests/` were altered during documentation updates.
  * Inspected `.gitignore` and confirmed presence of required exclusion patterns.

---

## 3. Summary of Project Verification Evidence

All pipeline behaviors and documentation claims are backed by executable code and verified test runs:

| Assessment Milestone | Verification Mechanism | Status / Observed Result |
|---|---|---|
| **Synthetic Reproducibility** | `tests/test_generator.py` | Identical outputs verified for fixed random seeds. |
| **Data Quality & Quarantine** | `tests/test_validator.py` | Standardized rejection error codes verified across validator unit tests; raw JSON payloads preserved. |
| **Point-in-Time Correctness** | `tests/test_point_in_time.py` | Lookahead and ingestion lag events successfully excluded. |
| **Rolling Window Aggregations** | `tests/test_features.py` | Exact numerical assertions for 7d, 30d, 90d metrics. |
| **Target Population Coverage** | `tests/test_targets.py` | Exactly 1 observation per customer; 0-tx customers receive label 0. |
| **Pipeline Idempotency & ACID** | `tests/test_idempotency.py` | Repeated runs produce identical counts; atomic rollback on simulated failure verified. |
| **Reconciliation Math** | `tests/test_reconciliation.py` | $N_{\text{raw}} \equiv N_{\text{valid}} + N_{\text{quarantined}}$ balance verified. |
| **100k Performance Optimization** | `tests/test_performance.py` | Vectorized aggregations completed in ~0.15s - 0.20s in local test runs (test asserts < 10.0s budget). |
| **Full Pytest Suite** | `python -m pytest -v` | **34 passed in 1.82s**. |
| **End-to-End Scale Execution** | `python -m src.pipeline` | 100k transactions, 2,500 customers processed in 2.883s overall. |
