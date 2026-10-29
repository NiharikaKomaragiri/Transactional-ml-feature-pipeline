"""Vectorized feature engineering across configurable time windows."""

from datetime import datetime, timezone
import logging
import time
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.point_in_time import PointInTimeEngine

logger = logging.getLogger("transactional_pipeline.features")


class FeatureEngineer:
    """Computes customer-level features over configurable rolling windows.
    
    Uses optimized vectorized aggregations to scale effortlessly to 100,000+ transactions.
    """

    def __init__(self, windows_days: Optional[List[int]] = None):
        self.windows_days = sorted(windows_days or [7, 30, 90])

    def compute_features(
        self,
        customers_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        observation_timestamp: Union[str, datetime],
    ) -> pd.DataFrame:
        """Compute customer point-in-time features vectorized as of observation_timestamp."""
        start_time = time.perf_counter()
        obs_dt = PointInTimeEngine.parse_timestamp(observation_timestamp)
        obs_pd = pd.Timestamp(obs_dt)
        obs_str = obs_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        max_window = max(self.windows_days)
        
        # 1. Apply Point-in-Time filter
        pit_tx = PointInTimeEngine.filter_transactions_as_of(
            transactions_df,
            observation_timestamp=obs_dt,
            max_lookback_days=max_window,
            only_completed=True,
        )

        # Base customer frame to ensure all customers are represented
        base_customers = customers_df[["customer_id", "country_code", "account_tier", "is_active"]].copy()
        base_customers["observation_timestamp"] = obs_str

        if pit_tx.empty:
            logger.warning("No transactions found eligible for feature calculation.")
            return self._empty_feature_frame(base_customers)

        # Prepare temporal deltas
        pit_tx["_ev_dt"] = pd.to_datetime(pit_tx["event_timestamp"], utc=True)
        pit_tx["_days_before_obs"] = (obs_pd - pit_tx["_ev_dt"]).dt.total_seconds() / 86400.0

        # Global recency (days since last transaction)
        recency_df = pit_tx.groupby("customer_id")["_days_before_obs"].min().reset_index()
        recency_df = recency_df.rename(columns={"_days_before_obs": "days_since_last_tx"})
        recency_df["days_since_last_tx"] = recency_df["days_since_last_tx"].round(2)

        # Merge recency into base
        features_df = pd.merge(base_customers, recency_df, on="customer_id", how="left")
        features_df["days_since_last_tx"] = features_df["days_since_last_tx"].fillna(999.0)

        # Calculate features per window using vectorized groupby
        for window in self.windows_days:
            w_mask = (pit_tx["_days_before_obs"] >= 0.0) & (pit_tx["_days_before_obs"] <= window)
            w_tx = pit_tx[w_mask]

            if w_tx.empty:
                features_df[f"tx_count_{window}d"] = 0
                features_df[f"tx_sum_amount_{window}d"] = 0.0
                features_df[f"tx_avg_amount_{window}d"] = 0.0
                features_df[f"tx_max_amount_{window}d"] = 0.0
                features_df[f"tx_std_amount_{window}d"] = 0.0
                continue

            # Grouped aggregation in a single vectorized pass
            agg = w_tx.groupby("customer_id")["amount"].agg(
                count="count",
                sum="sum",
                mean="mean",
                max="max",
                std="std",
            ).reset_index()

            agg = agg.rename(
                columns={
                    "count": f"tx_count_{window}d",
                    "sum": f"tx_sum_amount_{window}d",
                    "mean": f"tx_avg_amount_{window}d",
                    "max": f"tx_max_amount_{window}d",
                    "std": f"tx_std_amount_{window}d",
                }
            )

            # Round float values for clean storage and precision stability
            agg[f"tx_sum_amount_{window}d"] = agg[f"tx_sum_amount_{window}d"].round(2)
            agg[f"tx_avg_amount_{window}d"] = agg[f"tx_avg_amount_{window}d"].round(2)
            agg[f"tx_max_amount_{window}d"] = agg[f"tx_max_amount_{window}d"].round(2)
            agg[f"tx_std_amount_{window}d"] = agg[f"tx_std_amount_{window}d"].fillna(0.0).round(2)

            features_df = pd.merge(features_df, agg, on="customer_id", how="left")

            # Fill NA for customers with no transactions in this window
            features_df[f"tx_count_{window}d"] = features_df[f"tx_count_{window}d"].fillna(0).astype(int)
            features_df[f"tx_sum_amount_{window}d"] = features_df[f"tx_sum_amount_{window}d"].fillna(0.0)
            features_df[f"tx_avg_amount_{window}d"] = features_df[f"tx_avg_amount_{window}d"].fillna(0.0)
            features_df[f"tx_max_amount_{window}d"] = features_df[f"tx_max_amount_{window}d"].fillna(0.0)
            features_df[f"tx_std_amount_{window}d"] = features_df[f"tx_std_amount_{window}d"].fillna(0.0)

        # Cross-window velocity: 7d spend / (30d spend / 4.285)
        if 7 in self.windows_days and 30 in self.windows_days:
            # Expected weekly spend based on 30-day baseline
            expected_weekly = features_df["tx_sum_amount_30d"] / 4.2857
            # Avoid division by zero
            velocity = np.where(
                expected_weekly > 0.0,
                (features_df["tx_sum_amount_7d"] / expected_weekly).round(3),
                0.0,
            )
            features_df["tx_spend_velocity_7d_to_30d"] = velocity

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Vectorized feature computation completed in {elapsed:.3f}s "
            f"for {len(features_df)} customers ({len(self.windows_days)} windows)."
        )
        return features_df

    def _empty_feature_frame(self, base_customers: pd.DataFrame) -> pd.DataFrame:
        df = base_customers.copy()
        df["days_since_last_tx"] = 999.0
        for window in self.windows_days:
            df[f"tx_count_{window}d"] = 0
            df[f"tx_sum_amount_{window}d"] = 0.0
            df[f"tx_avg_amount_{window}d"] = 0.0
            df[f"tx_max_amount_{window}d"] = 0.0
            df[f"tx_std_amount_{window}d"] = 0.0
        if 7 in self.windows_days and 30 in self.windows_days:
            df["tx_spend_velocity_7d_to_30d"] = 0.0
        return df
