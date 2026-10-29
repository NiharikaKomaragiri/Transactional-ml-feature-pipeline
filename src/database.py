"""SQLite database manager with ACID transactions, schema migration, and indexing."""

from contextlib import contextmanager
import logging
from pathlib import Path
import sqlite3
from typing import Generator, List, Optional
import pandas as pd

logger = logging.getLogger("transactional_pipeline.database")


class DatabaseManager:
    """Manages SQLite schema creation, atomic writes, and idempotency."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an ACID transactional connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")  # High concurrency & performance
        conn.execute("PRAGMA synchronous = NORMAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            logger.error("Transaction rolled back due to an exception.")
            raise
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Create tables and performance indexes if not already existing."""
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                signup_timestamp TEXT NOT NULL,
                country_code TEXT NOT NULL,
                kyc_status TEXT NOT NULL,
                account_tier TEXT NOT NULL,
                is_active INTEGER NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                event_timestamp TEXT NOT NULL,
                available_timestamp TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                channel TEXT NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_transactions_customer_event 
            ON transactions(customer_id, event_timestamp, available_timestamp);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_transactions_available 
            ON transactions(available_timestamp);
            """,
            """
            CREATE TABLE IF NOT EXISTS quarantine_records (
                quarantine_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                source_table TEXT NOT NULL,
                raw_record_id TEXT NOT NULL,
                rejection_reason TEXT NOT NULL,
                rejected_at TEXT NOT NULL,
                raw_payload TEXT NOT NULL
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_quarantine_run 
            ON quarantine_records(run_id);
            """,
            """
            CREATE TABLE IF NOT EXISTS features (
                customer_id TEXT NOT NULL,
                observation_timestamp TEXT NOT NULL,
                country_code TEXT,
                account_tier TEXT,
                is_active INTEGER,
                days_since_last_tx REAL,
                tx_count_7d INTEGER,
                tx_sum_amount_7d REAL,
                tx_avg_amount_7d REAL,
                tx_max_amount_7d REAL,
                tx_std_amount_7d REAL,
                tx_count_30d INTEGER,
                tx_sum_amount_30d REAL,
                tx_avg_amount_30d REAL,
                tx_max_amount_30d REAL,
                tx_std_amount_30d REAL,
                tx_count_90d INTEGER,
                tx_sum_amount_90d REAL,
                tx_avg_amount_90d REAL,
                tx_max_amount_90d REAL,
                tx_std_amount_90d REAL,
                tx_spend_velocity_7d_to_30d REAL,
                PRIMARY KEY (customer_id, observation_timestamp)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS observation_targets (
                observation_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                observation_timestamp TEXT NOT NULL,
                target_label INTEGER NOT NULL
            );
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_targets_customer_obs 
            ON observation_targets(customer_id, observation_timestamp);
            """,
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                duration_seconds REAL,
                input_records_count INTEGER,
                valid_records_count INTEGER,
                quarantined_records_count INTEGER,
                features_generated_count INTEGER,
                status TEXT NOT NULL,
                error_message TEXT
            );
            """,
        ]
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            for ddl in ddl_statements:
                cursor.execute(ddl)
            conn.commit()
        logger.info(f"Database schema initialized at {self.db_path}")

    def insert_or_replace_df(self, df: pd.DataFrame, table_name: str) -> int:
        """Insert or replace DataFrame rows into SQLite table idempotently."""
        if df.empty:
            return 0
        with self.get_connection() as conn:
            df.to_sql(name=table_name, con=conn, if_exists="append", index=False, method=self._sqlite_upsert_method)
        return len(df)

    @staticmethod
    def _sqlite_upsert_method(table, conn, keys, data_iter):
        """Custom SQL insert with OR REPLACE to guarantee idempotency on primary keys."""
        columns = ", ".join(keys)
        placeholders = ", ".join(["?"] * len(keys))
        sql = f"INSERT OR REPLACE INTO {table.name} ({columns}) VALUES ({placeholders})"
        conn.executemany(sql, data_iter)

    def log_run_start(self, run_id: str, started_at: str) -> None:
        """Record the beginning of a pipeline run."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pipeline_runs 
                (run_id, started_at, status)
                VALUES (?, ?, 'STARTED')
                """,
                (run_id, started_at),
            )

    def log_run_completion(
        self,
        run_id: str,
        completed_at: str,
        duration_seconds: float,
        input_count: int,
        valid_count: int,
        quarantine_count: int,
        features_count: int,
        status: str = "SUCCESS",
        error_message: Optional[str] = None,
    ) -> None:
        """Record completion or failure of a pipeline run."""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE pipeline_runs SET
                    completed_at = ?,
                    duration_seconds = ?,
                    input_records_count = ?,
                    valid_records_count = ?,
                    quarantined_records_count = ?,
                    features_generated_count = ?,
                    status = ?,
                    error_message = ?
                WHERE run_id = ?
                """,
                (
                    completed_at,
                    duration_seconds,
                    input_count,
                    valid_count,
                    quarantine_count,
                    features_count,
                    status,
                    error_message,
                    run_id,
                ),
            )
