from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig

logger = logging.getLogger(__name__)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with a consistent logging pattern."""
    logger.info("Reading file: %s", path)
    return pd.read_csv(path)


def load_raw_tables(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the primary Stripe exports used across the pipeline."""
    checkout = read_csv(config.checkout_path)
    payment_links = read_csv(config.payment_links_path)
    payments = read_csv(config.payments_path)
    return checkout, payment_links, payments


def load_unified_payments(config: ProjectConfig) -> pd.DataFrame:
    """Load the unified payments export used for enrichment and correlation analysis."""
    return read_csv(config.payments_path)
