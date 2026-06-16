from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Resolve project root robustly (works no matter where you run Python from)
# Project structure assumed:
#   <PROJECT_ROOT>/
#     src/air_quality_loader.py
#     data/air_quality/AirQualityUCI.csv
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class UciAirQualityConfig:
    """
    Configuration for loading the UCI Air Quality dataset.
    """
    # Path to the UCI CSV file (semicolon-separated, decimal comma)
    csv_path: Path = PROJECT_ROOT / "data" / "air_quality" / "AirQualityUCI.csv"

    # UCI uses -200 to indicate missing values in many sensor columns
    missing_sentinel: float = -200.0

    # Whether to interpolate missing values after replacing sentinel
    interpolate: bool = True

    # Interpolation method for numeric columns
    interpolation_method: str = "time"

    # If set, resample to this pandas frequency (e.g., "1H"). Dataset is already hourly,
    # but this is useful if you later merge with other time series.
    resample_freq: Optional[str] = "1H"

    # Columns used to build the timestamp
    date_col: str = "Date"
    time_col: str = "Time"


def load_uci_air_quality(cfg: UciAirQualityConfig = UciAirQualityConfig()) -> pd.DataFrame:
    """
    Load and clean the UCI Air Quality dataset.

    - Parses Date + Time into a DateTimeIndex
    - Handles semicolon separator and decimal commas
    - Replaces -200 sentinel with NaN
    - Drops empty trailing column if present
    - Optionally resamples and interpolates missing values

    Returns:
        pandas DataFrame indexed by datetime, with numeric columns as float.
    """
    # Make sure the expected file exists (use resolved path for clarity in the error)
    if not cfg.csv_path.exists():
        raise FileNotFoundError(
            f"UCI AirQuality file not found at: {cfg.csv_path.resolve()}\n"
            f"Expected at: {PROJECT_ROOT / 'data' / 'air_quality' / 'AirQualityUCI.csv'}\n"
            f"Fix: place the file there, rename it to AirQualityUCI.csv, or update UciAirQualityConfig.csv_path."
        )

    # UCI file is semicolon-separated and typically uses decimal commas.
    # Read as strings first, then convert numeric columns carefully.
    df = pd.read_csv(cfg.csv_path, sep=";", dtype=str)

    # Some versions have a trailing empty column due to final semicolon.
    # Detect and drop completely empty columns.
    empty_cols = [
        c for c in df.columns
        if df[c].isna().all() or (df[c].astype(str).str.strip() == "").all()
    ]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    # Build datetime index from Date + Time.
    # UCI commonly stores Date as 'DD/MM/YYYY' and Time as 'HH.MM.SS'
    if cfg.date_col not in df.columns or cfg.time_col not in df.columns:
        raise ValueError(
            f"Expected columns '{cfg.date_col}' and '{cfg.time_col}' not found.\n"
            f"Columns present: {list(df.columns)}"
        )

    # Normalize time format: "HH.MM.SS" -> "HH:MM:SS"
    time_norm = df[cfg.time_col].astype(str).str.replace(".", ":", regex=False)
    dt_str = df[cfg.date_col].astype(str).str.strip() + " " + time_norm.str.strip()

    # Parse datetime
    dt = pd.to_datetime(
        dt_str,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
        dayfirst=True
    )

    # Remove date/time cols, set datetime index
    df = df.drop(columns=[cfg.date_col, cfg.time_col])
    df.insert(0, "datetime", dt)
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    # Convert numeric columns:
    # Replace decimal comma with dot, then to numeric.
    for col in df.columns:
        s = df[col].astype(str).str.strip()
        s = s.str.replace(",", ".", regex=False)  # "2,3" -> "2.3"
        df[col] = pd.to_numeric(s, errors="coerce")

    # Replace sentinel values (-200) with NaN
    df = df.replace(cfg.missing_sentinel, np.nan)

    # Ensure regular hourly index if requested
    if cfg.resample_freq:
        df = df.resample(cfg.resample_freq).mean()

    # Interpolate missing values over time for numeric columns
    if cfg.interpolate:
        df = df.interpolate(method=cfg.interpolation_method, limit_direction="both")

    return df


def select_aq_columns(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Convenience helper to keep only a subset of columns if desired.
    If columns=None, returns df unchanged.
    """
    if columns is None:
        return df
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Requested columns not in dataframe: {missing}. "
            f"Available: {list(df.columns)}"
        )
    return df[list(columns)].copy()
