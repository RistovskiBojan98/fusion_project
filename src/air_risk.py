"""
air_risk.py

Compute an outdoor air-quality risk score from the UCI Air Quality dataset.

Goal:
- Turn raw ambient pollutant signals into a single interpretable risk value in [0, 1]
- Keep it simple, explainable, and robust for an Ambient Intelligence project

Notes:
- The UCI Air Quality dataset contains "ground truth" columns like CO(GT), NO2(GT), NOx(GT)
  and multiple sensor signals PT08.S* that are less directly interpretable as pollutant levels.
- This module defaults to using the GT columns when available.
- Risk thresholds here are "reasonable engineering" defaults (not medical advice).
  You can tune them later in your report/experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AirRiskConfig:
    """
    Configuration for air risk scoring.

    - thresholds_*: values at which each pollutant is considered "high" (maps near 1.0)
    - weights_*: how much each pollutant contributes to overall risk
    """
    # Which columns to use if present
    preferred_cols: Tuple[str, ...] = ("CO(GT)", "NO2(GT)", "NOx(GT)")

    # Thresholds for normalization (value / threshold -> clipped to [0,1])
    # These are practical defaults; adjust as you like.
    co_high_mg_m3: float = 10.0     # CO (mg/m^3)
    no2_high_ug_m3: float = 200.0   # NO2 (µg/m^3)
    nox_high_ppb: float = 300.0     # NOx (ppb)

    # Weights must sum to 1.0 (we'll normalize if they don't).
    w_co: float = 0.2
    w_no2: float = 0.4
    w_nox: float = 0.4

    # Optional weather modifier: heat/humidity can increase physiological strain.
    use_weather_modifier: bool = True
    temp_col: str = "T"
    rh_col: str = "RH"
    # Conditions where strain becomes relevant; mild effect only.
    temp_high_c: float = 30.0
    rh_high_pct: float = 70.0
    weather_boost_max: float = 0.10  # at most +0.10 risk

    # Output column name
    risk_col: str = "air_risk_score"


def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)


def _safe_norm(series: pd.Series, denom: float) -> pd.Series:
    """
    Normalize series by denom and clip to [0,1]. Handles missing/NaN gracefully.
    """
    if denom <= 0:
        raise ValueError("Normalization denominator must be > 0.")
    return _clip01(series / denom)


def available_risk_inputs(df_aq: pd.DataFrame) -> Dict[str, bool]:
    """
    Report which expected columns are present in the AQ dataframe.
    """
    expected = ["CO(GT)", "NO2(GT)", "NOx(GT)", "T", "RH"]
    return {c: (c in df_aq.columns) for c in expected}


def compute_air_risk_components(
    df_aq: pd.DataFrame,
    cfg: AirRiskConfig = AirRiskConfig(),
    columns_override: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compute per-pollutant normalized components (each in [0,1]) and optionally a weather modifier.

    Returns a dataframe with component columns:
      - risk_CO
      - risk_NO2
      - risk_NOx
      - risk_weather (optional)
    """
    cols = list(columns_override) if columns_override is not None else list(cfg.preferred_cols)

    missing = [c for c in cols if c not in df_aq.columns]
    if missing:
        raise ValueError(
            f"Missing required pollutant columns: {missing}. "
            f"Available columns: {list(df_aq.columns)}"
        )

    components = pd.DataFrame(index=df_aq.index)

    # Pollutant components
    if "CO(GT)" in cols:
        components["risk_CO"] = _safe_norm(df_aq["CO(GT)"], cfg.co_high_mg_m3)
    if "NO2(GT)" in cols:
        components["risk_NO2"] = _safe_norm(df_aq["NO2(GT)"], cfg.no2_high_ug_m3)
    if "NOx(GT)" in cols:
        components["risk_NOx"] = _safe_norm(df_aq["NOx(GT)"], cfg.nox_high_ppb)

    # Optional mild weather modifier: only boosts risk slightly under hot & humid conditions.
    if cfg.use_weather_modifier and (cfg.temp_col in df_aq.columns) and (cfg.rh_col in df_aq.columns):
        temp_excess = _clip01((df_aq[cfg.temp_col] - cfg.temp_high_c) / max(cfg.temp_high_c, 1e-6))
        rh_excess = _clip01((df_aq[cfg.rh_col] - cfg.rh_high_pct) / max(cfg.rh_high_pct, 1e-6))
        # Combine (AND-ish) by multiplying, then scale to max boost.
        components["risk_weather"] = (temp_excess * rh_excess) * cfg.weather_boost_max
    else:
        components["risk_weather"] = 0.0

    return components


def compute_air_risk_score(
    df_aq: pd.DataFrame,
    cfg: AirRiskConfig = AirRiskConfig(),
    columns_override: Optional[Sequence[str]] = None,
) -> pd.Series:
    """
    Compute a single air risk score in [0,1] for each timestamp.

    Steps:
    1) compute normalized pollutant components in [0,1]
    2) weighted sum of pollutant risks
    3) add small weather modifier (optional)
    4) clip to [0,1]

    Returns:
        pd.Series named cfg.risk_col
    """
    comps = compute_air_risk_components(df_aq, cfg=cfg, columns_override=columns_override)

    # Build weights based on which components exist
    w = {}
    if "risk_CO" in comps.columns:
        w["risk_CO"] = cfg.w_co
    if "risk_NO2" in comps.columns:
        w["risk_NO2"] = cfg.w_no2
    if "risk_NOx" in comps.columns:
        w["risk_NOx"] = cfg.w_nox

    if not w:
        raise ValueError("No pollutant risk components were computed; cannot build risk score.")

    # Normalize weights to sum to 1.0, just in case
    w_sum = sum(w.values())
    if w_sum <= 0:
        raise ValueError("Weights must sum to > 0.")
    for k in w:
        w[k] = w[k] / w_sum

    base_risk = sum(comps[k] * w[k] for k in w.keys())
    risk = _clip01(base_risk + comps.get("risk_weather", 0.0))

    risk.name = cfg.risk_col
    return risk


def attach_air_risk(
    df_aq: pd.DataFrame,
    cfg: AirRiskConfig = AirRiskConfig(),
    columns_override: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return a copy of df_aq with:
      - cfg.risk_col (air_risk_score)
      - component columns (risk_CO, risk_NO2, risk_NOx, risk_weather)

    Useful for debugging and explainability.
    """
    out = df_aq.copy()
    comps = compute_air_risk_components(out, cfg=cfg, columns_override=columns_override)
    out = out.join(comps)
    out[cfg.risk_col] = compute_air_risk_score(out, cfg=cfg, columns_override=columns_override)
    return out
