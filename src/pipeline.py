"""Pipeline orchestrator for transactional ML feature engineering."""

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional, Tuple
import uuid
import pandas as pd

from src.config import PipelineConfig, setup_logging
from src.database import DatabaseManager
from src.features import FeatureEngineer
from src.generator import SyntheticDataGenerator, generate_and_save_data
from src.point_in_time import PointInTimeEngine
from src.reconciliation import ReconciliationEngine, ReconciliationMetrics
from src.validator import DataQualityValidator

logger = logging.getLogger("transactional_pipeline.orchestrator")


class FeaturePipeline:
    """End-to-end pipeline orchestrator for transactional feature engineering."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.config.ensure_directories()
        self.db_manager = DatabaseManager(self.config.db_path)

    def run(
        self,
        run_id: Optional[str] = None,
        customers_df: Optional[pd.DataFrame] = None,
        transactions_df: Optional[pd.DataFrame] = None,
    ) -> ReconciliationMetrics:
        """Execute the pipeline from ingestion to feature publication."""
        start_time = time.perf_counter()
        now_dt = datetime.now(timezone.utc)
        started_at = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        run_id = run_id or f"RUN_{now_dt.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        logger.info(f"Starting pipeline execution: {run_id}")
        self.db_manager.log_run_start(run_id, started_at)

        try:
            # 1. Load data if not supplied directly in-memory
            if customers_df is None or transactions_df is None:
                cust_path = self.config.raw_dir / "customers.csv"
                tx_path = self.config.raw_dir / "transactions.csv"
                if not cust_path.exists() or not tx_path.exists():
                    logger.info("Raw data files not found. Triggering synthetic generation...")
                    customers_df, transactions_df = generate_and_save_data(
                        raw_dir=self.config.raw_dir,
                        num_customers=self.config.num_customers,
                        num_transactions=self.config.num_transactions,
                        seed=self.config.seed,
                        inject_anomalies=True,
                        anomaly_rate=self.config.synthetic_noise_rate,
                    )
                else:
                    customers_df = pd.read_csv(cust_path)
                    transactions_df = pd.read_csv(tx_path)

            raw_tx_count = len(transactions_df)
            raw_cust_count = len(customers_df)
            logger.info(f"Loaded {raw_cust_count} raw customers and {raw_tx_count} raw transactions.")

            # 2. Data Quality & Schema Validation
            validator = DataQualityValidator(run_id=run_id)
            valid_cust_df, quarantine_cust_df = validator.validate_customers(customers_df)

            valid_customer_ids = set(valid_cust_df["customer_id"].astype(str))
            valid_tx_df, quarantine_tx_df = validator.validate_transactions(
                transactions_df, valid_customer_ids=valid_customer_ids
            )

            # 3. Store Quarantined Records
            all_quarantine_df = pd.concat(
                [quarantine_cust_df, quarantine_tx_df], ignore_index=True
            )
            if not all_quarantine_df.empty:
                self.db_manager.insert_or_replace_df(all_quarantine_df, "quarantine_records")
                self._save_df_atomic(
                    all_quarantine_df,
                    self.config.quarantine_dir / "quarantined_records.csv",
                    mode="a" if (self.config.quarantine_dir / "quarantined_records.csv").exists() else "w",
                )

            # 4. Ingest Curated Customers & Transactions into SQLite
            self.db_manager.insert_or_replace_df(valid_cust_df, "customers")
            self.db_manager.insert_or_replace_df(valid_tx_df, "transactions")

            # 5. Point-in-Time Feature Engineering
            engineer = FeatureEngineer(windows_days=self.config.feature_windows_days)
            features_df = engineer.compute_features(
                customers_df=valid_cust_df,
                transactions_df=valid_tx_df,
                observation_timestamp=self.config.observation_date,
            )

            # 6. Save Published Features Idempotently
            self.db_manager.insert_or_replace_df(features_df, "features")
            self._save_df_atomic(
                features_df,
                self.config.features_dir / "customer_features.csv",
            )

            # 7. Generate Isolated Ground Truth Targets
            obs_dt = PointInTimeEngine.parse_timestamp(self.config.observation_date)
            generator = SyntheticDataGenerator(seed=self.config.seed)
            targets_df = generator.generate_observation_targets(
                transactions_df=valid_tx_df,
                customers_df=valid_cust_df,
                observation_date=obs_dt,
                horizon_days=self.config.target_horizon_days,
            )
            self.db_manager.insert_or_replace_df(targets_df, "observation_targets")
            self._save_df_atomic(
                targets_df,
                self.config.features_dir / "observation_targets.csv",
            )

            # 8. Reconciliation and Audit Metrics
            elapsed = time.perf_counter() - start_time
            end_dt = datetime.now(timezone.utc)
            completed_at = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            metrics = ReconciliationEngine.generate_metrics(
                run_id=run_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=elapsed,
                raw_tx_count=raw_tx_count,
                valid_tx_count=len(valid_tx_df),
                quarantine_tx_df=quarantine_tx_df,
                raw_cust_count=raw_cust_count,
                valid_cust_count=len(valid_cust_df),
                quarantine_cust_df=quarantine_cust_df,
                features_count=len(features_df),
                targets_count=len(targets_df),
                status="SUCCESS",
            )

            # 9. Update Database Audit Run Record
            self.db_manager.log_run_completion(
                run_id=run_id,
                completed_at=completed_at,
                duration_seconds=metrics.duration_seconds,
                input_count=raw_tx_count,
                valid_count=len(valid_tx_df),
                quarantine_count=len(quarantine_tx_df),
                features_count=len(features_df),
                status="SUCCESS",
            )

            # 10. Format and save reconciliation report
            report_text = ReconciliationEngine.format_report_text(metrics)
            ReconciliationEngine.save_report(
                report_text, self.config.project_root / "docs" / "sample_reconciliation_report.txt"
            )
            print(report_text)
            return metrics

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            end_dt = datetime.now(timezone.utc)
            completed_at = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.exception(f"Pipeline run {run_id} failed: {e}")
            self.db_manager.log_run_completion(
                run_id=run_id,
                completed_at=completed_at,
                duration_seconds=round(elapsed, 3),
                input_count=0,
                valid_count=0,
                quarantine_count=0,
                features_count=0,
                status="FAILED",
                error_message=str(e),
            )
            raise

    @staticmethod
    def _save_df_atomic(df: pd.DataFrame, target_path: Path, mode: str = "w") -> None:
        """Atomically write DataFrame to file via temporary swap."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "a" and target_path.exists():
            df.to_csv(target_path, mode="a", header=False, index=False)
        else:
            tmp_path = target_path.with_suffix(".tmp")
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(target_path)


def main():
    """Command-line interface for the feature engineering pipeline."""
    parser = argparse.ArgumentParser(description="Transactional ML Feature Engineering Pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate fresh synthetic datasets")
    parser.add_argument("--customers", type=int, default=2500, help="Number of customers to generate")
    parser.add_argument("--transactions", type=int, default=100000, help="Number of transactions to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--observation-date", type=str, default="2026-06-01T00:00:00Z", help="Point-in-time observation cutoff")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run ID")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite database path")

    args = parser.parse_args()

    setup_logging(log_level=logging.INFO)
    config = PipelineConfig(
        seed=args.seed,
        num_customers=args.customers,
        num_transactions=args.transactions,
        observation_date=args.observation_date,
    )
    if args.db_path:
        config.db_path = Path(args.db_path)

    if args.generate:
        logger.info("Generating synthetic dataset from CLI flag...")
        generate_and_save_data(
            raw_dir=config.raw_dir,
            num_customers=config.num_customers,
            num_transactions=config.num_transactions,
            seed=config.seed,
            inject_anomalies=True,
            anomaly_rate=config.synthetic_noise_rate,
        )

    pipeline = FeaturePipeline(config=config)
    pipeline.run(run_id=args.run_id)


if __name__ == "__main__":
    main()
