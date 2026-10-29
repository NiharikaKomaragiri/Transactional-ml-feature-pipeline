"""Point-in-Time (PIT) correctness engine and temporal cutoff filtering."""

from datetime import datetime, timedelta, timezone
import logging
from typing import Optional, Union
import pandas as pd

logger = logging.getLogger("transactional_pipeline.pit")


class PointInTimeEngine:
    """Enforces temporal isolation using dual-timestamp point-in-time rules."""

    @staticmethod
    def parse_timestamp(ts: Union[str, datetime]) -> datetime:
        """Parse any ISO string or datetime to UTC-aware datetime."""
        if isinstance(ts, str):
            dt = pd.to_datetime(ts, utc=True).to_pydatetime()
            return dt
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts.astimezone(timezone.utc)
        raise ValueError(f"Unsupported timestamp format: {ts}")

    @classmethod
    def filter_transactions_as_of(
        cls,
        transactions_df: pd.DataFrame,
        observation_timestamp: Union[str, datetime],
        max_lookback_days: Optional[int] = None,
        only_completed: bool = True,
    ) -> pd.DataFrame:
        """Filter transactions to only those visible and valid as of observation_timestamp.
        
        Point-in-time conditions:
          1. event_timestamp <= observation_timestamp (No future events)
          2. available_timestamp <= observation_timestamp (No pipeline lag / uningested events)
          3. event_timestamp >= observation_timestamp - max_lookback_days (Optional window bound)
          4. status == 'COMPLETED' (if only_completed is True)
        """
        if transactions_df.empty:
            return transactions_df.copy()

        obs_dt = cls.parse_timestamp(observation_timestamp)
        obs_pd = pd.Timestamp(obs_dt)

        df = transactions_df.copy()
        
        # Ensure datetime format for filtering
        df["_ev_dt"] = pd.to_datetime(df["event_timestamp"], utc=True)
        df["_av_dt"] = pd.to_datetime(df["available_timestamp"], utc=True)

        # Apply dual-timestamp PIT conditions
        pit_mask = (df["_ev_dt"] <= obs_pd) & (df["_av_dt"] <= obs_pd)

        if only_completed and "status" in df.columns:
            pit_mask = pit_mask & (df["status"] == "COMPLETED")

        if max_lookback_days is not None:
            earliest_allowed = obs_pd - pd.Timedelta(days=max_lookback_days)
            pit_mask = pit_mask & (df["_ev_dt"] >= earliest_allowed)

        filtered_df = df[pit_mask].drop(columns=["_ev_dt", "_av_dt"])

        logger.info(
            f"PIT Cutoff at {obs_dt.isoformat()}: "
            f"{len(filtered_df)} / {len(transactions_df)} transactions eligible."
        )
        return filtered_df
