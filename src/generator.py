"""Reproducible synthetic data generator for customers, transactions, and observation points."""

from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.schemas import CustomerSchema, TransactionSchema

logger = logging.getLogger("transactional_pipeline.generator")


class SyntheticDataGenerator:
    """Generates synthetic customer and transactional data with fixed random seeds."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_customers(
        self,
        num_customers: int = 2_500,
        start_date: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_date: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc),
    ) -> pd.DataFrame:
        """Generate reproducible customer profiles."""
        logger.info(f"Generating {num_customers} synthetic customer records (seed={self.seed})...")
        
        customer_ids = [f"CUST_{i:06d}" for i in range(1, num_customers + 1)]
        
        # Signup timestamps uniformly distributed
        total_seconds = int((end_date - start_date).total_seconds())
        random_seconds = self.rng.integers(0, total_seconds, size=num_customers)
        signups = [start_date + timedelta(seconds=int(s)) for s in random_seconds]
        signup_strs = [dt.strftime("%Y-%m-%dT%H:%M:%SZ") for dt in signups]
        
        countries = list(["US", "GB", "IN", "CA", "DE", "AU", "SG"])
        country_probs = [0.40, 0.15, 0.20, 0.10, 0.05, 0.05, 0.05]
        assigned_countries = self.rng.choice(countries, size=num_customers, p=country_probs)
        
        kyc_options = ["VERIFIED", "PENDING", "REJECTED"]
        kyc_probs = [0.85, 0.12, 0.03]
        assigned_kyc = self.rng.choice(kyc_options, size=num_customers, p=kyc_probs)
        
        tier_options = ["STANDARD", "PREMIUM", "VIP"]
        tier_probs = [0.70, 0.25, 0.05]
        assigned_tiers = self.rng.choice(tier_options, size=num_customers, p=tier_probs)
        
        is_active = self.rng.choice([True, False], size=num_customers, p=[0.92, 0.08])
        
        df = pd.DataFrame({
            "customer_id": customer_ids,
            "signup_timestamp": signup_strs,
            "country_code": assigned_countries,
            "kyc_status": assigned_kyc,
            "account_tier": assigned_tiers,
            "is_active": is_active,
        })
        return df

    def generate_transactions(
        self,
        customer_ids: List[str],
        num_transactions: int = 100_000,
        start_date: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_date: datetime = datetime(2026, 6, 30, tzinfo=timezone.utc),
        inject_anomalies: bool = False,
        anomaly_rate: float = 0.02,
    ) -> pd.DataFrame:
        """Generate reproducible transaction event stream with realistic ingestion lag."""
        logger.info(f"Generating {num_transactions} synthetic transactions (seed={self.seed})...")
        
        # Transaction IDs
        tx_ids = [f"TX_{i:09d}" for i in range(1, num_transactions + 1)]
        
        # Power-law/Zipf assignment of transactions to customers
        # (some customers transact much more frequently than others)
        num_cust = len(customer_ids)
        cust_indices = self.rng.zipf(a=1.5, size=num_transactions) - 1
        cust_indices = np.clip(cust_indices, 0, num_cust - 1)
        assigned_customers = [customer_ids[idx] for idx in cust_indices]
        
        # Event timestamps
        total_seconds = int((end_date - start_date).total_seconds())
        event_offsets = self.rng.integers(0, total_seconds, size=num_transactions)
        # Sort offsets roughly so transactions are roughly chronological
        event_offsets.sort()
        event_times = [start_date + timedelta(seconds=int(s)) for s in event_offsets]
        
        # Available timestamps: Event time + Ingestion Lag
        # Most lag is fast (seconds to minutes), some has network delay or batch lag (hours)
        lags_seconds = self.rng.exponential(scale=180.0, size=num_transactions)
        # Small portion of batch delayed transactions (up to 2 days lag)
        delayed_idx = self.rng.choice(num_transactions, size=int(num_transactions * 0.01), replace=False)
        lags_seconds[delayed_idx] += self.rng.uniform(3600 * 12, 3600 * 48, size=len(delayed_idx))
        
        available_times = [
            ev + timedelta(seconds=float(max(0.0, lag)))
            for ev, lag in zip(event_times, lags_seconds)
        ]
        
        # Amounts: Log-normal distribution (median ~$35, tail up to thousands)
        raw_amounts = self.rng.lognormal(mean=3.5, sigma=1.0, size=num_transactions)
        amounts = np.round(np.clip(raw_amounts, 1.0, 10000.0), 2)
        
        channels = ["MOBILE", "WEB", "POS", "ATM"]
        channel_probs = [0.55, 0.25, 0.15, 0.05]
        assigned_channels = self.rng.choice(channels, size=num_transactions, p=channel_probs)
        
        categories = ["RETAIL", "GROCERY", "TRAVEL", "ENTERTAINMENT", "UTILITIES", "TRANSFER"]
        category_probs = [0.30, 0.25, 0.10, 0.15, 0.10, 0.10]
        assigned_categories = self.rng.choice(categories, size=num_transactions, p=category_probs)
        
        statuses = ["COMPLETED", "PENDING", "DECLINED", "REVERSED"]
        status_probs = [0.93, 0.03, 0.03, 0.01]
        assigned_statuses = self.rng.choice(statuses, size=num_transactions, p=status_probs)
        
        df = pd.DataFrame({
            "transaction_id": tx_ids,
            "customer_id": assigned_customers,
            "event_timestamp": [dt.strftime("%Y-%m-%dT%H:%M:%SZ") for dt in event_times],
            "available_timestamp": [dt.strftime("%Y-%m-%dT%H:%M:%SZ") for dt in available_times],
            "amount": amounts,
            "currency": ["USD"] * num_transactions,
            "channel": assigned_channels,
            "category": assigned_categories,
            "status": assigned_statuses,
        })
        
        if inject_anomalies:
            df = self._inject_anomalies(df, anomaly_rate)
            
        return df

    def _inject_anomalies(self, df: pd.DataFrame, rate: float) -> pd.DataFrame:
        """Inject realistic data quality anomalies to test quarantine engine."""
        df = df.copy()
        n = len(df)
        num_anomalies = max(5, int(n * rate))
        
        logger.info(f"Injecting ~{num_anomalies} anomalies across transaction records...")
        anomaly_indices = self.rng.choice(n, size=num_anomalies, replace=False)
        
        # Partition anomaly indices into different categories
        splits = np.array_split(anomaly_indices, 6)
        
        # 1. Missing required field (null customer_id or amount)
        for idx in splits[0]:
            df.at[idx, "customer_id"] = None
            
        # 2. Invalid negative / zero amount
        for idx in splits[1]:
            df.at[idx, "amount"] = -25.50 if self.rng.random() > 0.5 else 0.00
            
        # 3. Negative timestamp lag (available_timestamp < event_timestamp)
        for idx in splits[2]:
            ev_str = df.at[idx, "event_timestamp"]
            ev_dt = datetime.strptime(ev_str, "%Y-%m-%dT%H:%M:%SZ")
            bad_avail = ev_dt - timedelta(hours=2)
            df.at[idx, "available_timestamp"] = bad_avail.strftime("%Y-%m-%dT%H:%M:%SZ")
            
        # 4. Orphan foreign key (customer does not exist)
        for idx in splits[3]:
            df.at[idx, "customer_id"] = "CUST_999999"
            
        # 5. Duplicate primary key
        for idx in splits[4]:
            donor_idx = max(0, idx - 1)
            df.at[idx, "transaction_id"] = df.at[donor_idx, "transaction_id"]
            
        # 6. Invalid enum value (e.g. invalid channel or status)
        for idx in splits[5]:
            df.at[idx, "channel"] = "BITCOIN_KIOSK"
            
        return df

    def generate_observation_targets(
        self,
        transactions_df: pd.DataFrame,
        customers_df: Optional[Union[pd.DataFrame, List[str], Set[str], pd.Series]] = None,
        observation_date: Union[datetime, str] = datetime(2026, 6, 1, tzinfo=timezone.utc),
        horizon_days: int = 30,
        customer_ids: Optional[Union[List[str], Set[str], pd.Series]] = None,
    ) -> pd.DataFrame:
        """Generate separate ground truth target dataset for ML observation points.
        
        Ensures every valid customer in the customer dataset receives exactly one
        observation target, even if they have zero transactions.
        
        Target Definition:
          - Observation timestamp: observation_date
          - Horizon: [observation_date, observation_date + horizon_days) (inclusive start, exclusive end)
          - Label = 1 if customer total spending in future horizon > $1,500 OR customer has >= 1 declined transaction in future horizon.
          - Otherwise, Label = 0.
        """
        # Handle backward compatibility if observation_date was passed positionally as 2nd arg
        if isinstance(customers_df, (datetime, pd.Timestamp)):
            observation_date = customers_df
            customers_df = None
        elif isinstance(customers_df, str) and ("T" in customers_df or ("-" in customers_df and not str(customers_df).startswith("CUST"))):
            observation_date = customers_df
            customers_df = None

        # Parse observation timestamp to UTC datetime
        if isinstance(observation_date, str):
            obs_dt = pd.to_datetime(observation_date, utc=True).to_pydatetime()
        elif isinstance(observation_date, datetime):
            obs_dt = observation_date if observation_date.tzinfo else observation_date.replace(tzinfo=timezone.utc)
        else:
            obs_dt = pd.to_datetime(str(observation_date), utc=True).to_pydatetime()

        obs_str = obs_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        horizon_end = obs_dt + timedelta(days=horizon_days)

        logger.info(f"Generating observation targets as-of {obs_str} (horizon={horizon_days}d)...")

        # 1. Determine complete valid customer population
        if customers_df is not None:
            if isinstance(customers_df, pd.DataFrame):
                if "customer_id" in customers_df.columns:
                    all_customers = sorted(list(set(customers_df["customer_id"].dropna().astype(str))))
                else:
                    raise ValueError("customers_df must contain a 'customer_id' column.")
            elif isinstance(customers_df, (list, set, tuple, pd.Series)):
                all_customers = sorted(list(set(str(c) for c in customers_df if pd.notna(c))))
            else:
                raise ValueError(f"Unsupported format for customers_df: {type(customers_df)}")
        elif customer_ids is not None:
            all_customers = sorted(list(set(str(c) for c in customer_ids if pd.notna(c))))
        else:
            # Fallback for backward compatibility when only transactions_df is provided
            if not transactions_df.empty and "customer_id" in transactions_df.columns:
                all_customers = sorted(list(set(transactions_df["customer_id"].dropna().astype(str))))
            else:
                all_customers = []

        if not all_customers:
            logger.warning("No customers found for observation target generation.")
            return pd.DataFrame(columns=["observation_id", "customer_id", "observation_timestamp", "target_label"])

        # 2. Filter & sanitize future transactions
        cust_spending = pd.Series(dtype=float)
        declined_custs = set()

        if not transactions_df.empty and "customer_id" in transactions_df.columns and "event_timestamp" in transactions_df.columns:
            valid_cust_set = set(all_customers)
            # Filter only transactions belonging to valid customers (ignores orphans)
            tx_clean = transactions_df[
                transactions_df["customer_id"].notna() &
                transactions_df["customer_id"].astype(str).isin(valid_cust_set) &
                transactions_df["event_timestamp"].notna()
            ].copy()

            # Deduplicate transaction_id if present
            if "transaction_id" in tx_clean.columns:
                tx_clean = tx_clean.drop_duplicates(subset=["transaction_id"], keep="first")

            # Parse event timestamps
            tx_clean["event_dt"] = pd.to_datetime(tx_clean["event_timestamp"], errors="coerce", utc=True)
            tx_clean = tx_clean.dropna(subset=["event_dt"])

            # Filter strictly within future horizon [obs_dt, horizon_end) (inclusive start, exclusive end)
            horizon_mask = (tx_clean["event_dt"] >= obs_dt) & (tx_clean["event_dt"] < horizon_end)
            horizon_tx = tx_clean[horizon_mask].copy()

            if not horizon_tx.empty:
                # Valid spending: positive numeric amounts
                numeric_amounts = pd.to_numeric(horizon_tx["amount"], errors="coerce").fillna(0.0)
                horizon_tx["clean_amount"] = numeric_amounts.clip(lower=0.0)

                cust_spending = horizon_tx.groupby("customer_id")["clean_amount"].sum()

                # Declined transactions in future horizon
                if "status" in horizon_tx.columns:
                    is_declined = horizon_tx["status"].astype(str).str.strip().str.upper() == "DECLINED"
                    declined_custs = set(horizon_tx[is_declined]["customer_id"].astype(str))

        # 3. Build target observations for EVERY customer in all_customers
        records = []
        for i, cid in enumerate(all_customers, 1):
            spent = float(cust_spending.get(cid, 0.0))
            has_declined = cid in declined_custs
            # Label = 1 if spending > $1,500 OR has at least one declined transaction in horizon
            label = 1 if (spent > 1500.0 or has_declined) else 0
            records.append({
                "observation_id": f"OBS_{i:06d}",
                "customer_id": cid,
                "observation_timestamp": obs_str,
                "target_label": int(label),
            })

        logger.info(
            f"Generated {len(records)} observation targets ({sum(r['target_label'] for r in records)} positive labels)."
        )
        return pd.DataFrame(records, columns=["observation_id", "customer_id", "observation_timestamp", "target_label"])


def generate_and_save_data(
    raw_dir: Path,
    num_customers: int = 2_500,
    num_transactions: int = 100_000,
    seed: int = 42,
    inject_anomalies: bool = True,
    anomaly_rate: float = 0.02,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and write raw datasets to CSVs atomically."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    generator = SyntheticDataGenerator(seed=seed)
    
    customers_df = generator.generate_customers(num_customers=num_customers)
    transactions_df = generator.generate_transactions(
        customer_ids=customers_df["customer_id"].tolist(),
        num_transactions=num_transactions,
        inject_anomalies=inject_anomalies,
        anomaly_rate=anomaly_rate,
    )
    
    customers_path = raw_dir / "customers.csv"
    transactions_path = raw_dir / "transactions.csv"
    
    customers_df.to_csv(customers_path, index=False)
    transactions_df.to_csv(transactions_path, index=False)
    logger.info(f"Saved {len(customers_df)} customers to {customers_path}")
    logger.info(f"Saved {len(transactions_df)} transactions to {transactions_path}")
    
    return customers_df, transactions_df
