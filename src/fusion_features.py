from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FusionFeatureConfig:
    trend_k: int = 3
    include_interactions: bool = True


def compute_trend(series: np.ndarray, k: int) -> np.ndarray:
    """
    Trend feature per time step:
      trend[t] = series[t] - mean(series[t-k : t]) for t>=k
      else 0
    """
    s = np.asarray(series, dtype=float)
    out = np.zeros_like(s, dtype=float)
    if k <= 1:
        return out
    for t in range(len(s)):
        if t >= k:
            prev = s[t - k : t]
            out[t] = s[t] - float(np.mean(prev))
        else:
            out[t] = 0.0
    return out


def build_fusion_features(
    df: pd.DataFrame,
    cfg: FusionFeatureConfig = FusionFeatureConfig(),
) -> pd.DataFrame:
    """
    Build a feature table for fusion learning.
    Requires columns: p_stress, air_risk
    Optionally uses subject_id for grouping (not as a feature).
    """
    if "p_stress" not in df.columns or "air_risk" not in df.columns:
        raise ValueError("df must contain columns: p_stress, air_risk")

    p = df["p_stress"].astype(float).to_numpy()
    a = df["air_risk"].astype(float).to_numpy()

    trend = compute_trend(p, cfg.trend_k)

    X = pd.DataFrame(
        {
            "p_stress": p,
            "air_risk": a,
            "p_trend": trend,
        },
        index=df.index,
    )

    if cfg.include_interactions:
        X["p_x_air"] = X["p_stress"] * X["air_risk"]
        X["p_plus_air"] = X["p_stress"] + X["air_risk"]

    return X
