"""Pipeline configuration, constants, and logging setup."""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
QUARANTINE_DIR = DATA_DIR / "quarantine"
FEATURES_DIR = DATA_DIR / "features"
LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_DB_PATH = DATA_DIR / "pipeline.db"


@dataclass
class PipelineConfig:
    """Runtime configuration for the transactional feature pipeline."""
    seed: int = 42
    num_customers: int = 2_500
    num_transactions: int = 100_000
    feature_windows_days: List[int] = field(default_factory=lambda: [7, 30, 90])
    observation_date: str = "2026-06-01T00:00:00Z"
    target_horizon_days: int = 30
    synthetic_noise_rate: float = 0.02
    
    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    quarantine_dir: Path = QUARANTINE_DIR
    features_dir: Path = FEATURES_DIR
    logs_dir: Path = LOGS_DIR
    db_path: Path = DEFAULT_DB_PATH
    
    def ensure_directories(self) -> None:
        """Create required storage directories if they do not exist."""
        for directory in [self.data_dir, self.raw_dir, self.quarantine_dir, self.features_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)


def setup_logging(log_level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    """Set up structured logging to stdout and optionally a log file."""
    logger = logging.getLogger("transactional_pipeline")
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ"
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger
