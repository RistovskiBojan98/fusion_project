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
- Risk thresholds here are engineering defaults for scenario-based experiments.
  They are not medical advice or regulatory thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class AirRiskConfig:
    """
    Configuration for air-risk scoring.

    - thresholds_*: values at which each pollutant is considered "high" and maps near 1.0
    - weights_*: how much each pollutant contributes to the overall pollutant risk

    Default pollutant weights:
    - CO  = 0.20
    - NO2 = 0.40
    - NOx = 0.40

    These defaults give more weight to NO2 and NOx because they are useful
    traffic-related outdoor pollution signals for a respiratory/physiological
    strain scenario. CO is still included, but with a smaller default
    contribution. The weights are configurable so the project can report
    sensitivity analyses instead of claiming that one hand-tuned setting is
    universally correct.
    """

    preferred_cols: Tuple[str, ...] = ("CO(GT)", "NO2(GT)", "NOx(GT)")

    # Thresholds for normalization: value / threshold -> clipped to [0, 1].
    co_high_mg_m3: float = 10.0
    no2_high_ug_m3: float = 200.0
    nox_high_ppb: float = 300.0

    # Pollutant weights. They are normalized over available pollutant components.
    w_co: float = 0.2
    w_no2: float = 0.4
    w_nox: float = 0.4

    # Optional weather modifier: heat/humidity can increase physiological strain.
    use_weather_modifier: bool = True
    temp_col: str = "T"
    rh_col: str = "RH"
    temp_high_c: float = 30.0
    rh_high_pct: float = 70.0
    weather_boost_max: float = 0.10

    risk_col: str = "air_risk_score"

    def pollutant_weight_map(self) -> Dict[str, float]:
        return {
            "risk_CO": self.w_co,
            "risk_NO2": self.w_no2,
            "risk_NOx": self.w_nox,
        }


def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)


def _safe_norm(series: pd.Series, denom: float) -> pd.Series:
    """
    Normalize series by denom and clip to [0, 1]. Handles missing values gracefully.
    """
    if denom <= 0:
        raise ValueError("Normalization denominator must be > 0.")
    return _clip01(series / denom)


def available_risk_inputs(df_aq: pd.DataFrame) -> Dict[str, bool]:
    """
    Report which expected columns are present in the air-quality dataframe.
    """
    expected = ["CO(GT)", "NO2(GT)", "NOx(GT)", "T", "RH"]
    return {c: (c in df_aq.columns) for c in expected}


def normalized_pollutant_weights(cfg: AirRiskConfig, component_columns: Sequence[str]) -> Dict[str, float]:
    """
    Return normalized weights for the pollutant components that are available.

    If a pollutant is missing or excluded, the remaining pollutant weights are
    re-normalized to sum to 1.0. The weather modifier is additive and is not part
    of the pollutant weight sum.
    """
    raw = cfg.pollutant_weight_map()
    weights = {col: raw[col] for col in component_columns if col in raw}
    if not weights:
        raise ValueError("No pollutant risk components were computed; cannot build risk score.")

    w_sum = sum(weights.values())
    if w_sum <= 0:
        raise ValueError(
            "Pollutant weights must sum to > 0. "
            f"Received CO={cfg.w_co}, NO2={cfg.w_no2}, NOx={cfg.w_nox}."
        )

    return {col: value / w_sum for col, value in weights.items()}


def compute_air_risk_components(
    df_aq: pd.DataFrame,
    cfg: AirRiskConfig = AirRiskConfig(),
    columns_override: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compute per-pollutant normalized components and an optional weather modifier.

    Returns columns:
    - risk_CO
    - risk_NO2
    - risk_NOx
    - risk_weather
    """
    cols = list(columns_override) if columns_override is not None else list(cfg.preferred_cols)

    missing = [c for c in cols if c not in df_aq.columns]
    if missing:
        raise ValueError(
            f"Missing required pollutant columns: {missing}. "
            f"Available columns: {list(df_aq.columns)}"
        )

    components = pd.DataFrame(index=df_aq.index)

    if "CO(GT)" in cols:
        components["risk_CO"] = _safe_norm(df_aq["CO(GT)"], cfg.co_high_mg_m3)
    if "NO2(GT)" in cols:
        components["risk_NO2"] = _safe_norm(df_aq["NO2(GT)"], cfg.no2_high_ug_m3)
    if "NOx(GT)" in cols:
        components["risk_NOx"] = _safe_norm(df_aq["NOx(GT)"], cfg.nox_high_ppb)

    if cfg.use_weather_modifier and (cfg.temp_col in df_aq.columns) and (cfg.rh_col in df_aq.columns):
        temp_excess = _clip01((df_aq[cfg.temp_col] - cfg.temp_high_c) / max(cfg.temp_high_c, 1e-6))
        rh_excess = _clip01((df_aq[cfg.rh_col] - cfg.rh_high_pct) / max(cfg.rh_high_pct, 1e-6))
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
    Compute a single air-risk score in [0, 1] for each timestamp.
    """
    comps = compute_air_risk_components(df_aq, cfg=cfg, columns_override=columns_override)
    weights = normalized_pollutant_weights(cfg, comps.columns)
    base_risk = sum(comps[col] * weights[col] for col in weights)
    risk = _clip01(base_risk + comps.get("risk_weather", 0.0))
    risk.name = cfg.risk_col
    return risk


def attach_air_risk(
    df_aq: pd.DataFrame,
    cfg: AirRiskConfig = AirRiskConfig(),
    columns_override: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return a copy of df_aq with component columns and cfg.risk_col attached.
    """
    out = df_aq.copy()
    comps = compute_air_risk_components(out, cfg=cfg, columns_override=columns_override)
    out = out.join(comps)
    out[cfg.risk_col] = compute_air_risk_score(out, cfg=cfg, columns_override=columns_override)
    return out
