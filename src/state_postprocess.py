from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpecialStateConfig:
    # thresholds for special states (use calibrated p_stress + smoothed signals)
    stress_med: float
    stress_high: float
    air_high: float = 0.65

    # rising stress
    trend_k: int = 3
    rising_trend_delta: float = 0.10  # p_now - mean(prev k)
    rising_min_p: float | None = None  # if None -> use stress_med


def compute_trend(p: np.ndarray, k: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p, dtype=float)
    for i in range(len(p)):
        if i >= k:
            out[i] = p[i] - float(np.mean(p[i - k : i]))
        else:
            out[i] = 0.0
    return out


def enforce_special_states(
    status: list[str],
    p_stress_s: np.ndarray,
    air_risk_s: np.ndarray,
    cfg: SpecialStateConfig,
) -> list[str]:
    """
    Enforce two special statuses AFTER model prediction:
      1) high_risk_combined has top priority
      2) rising_stress is applied when trend is high but absolute stress isn't "high" yet

    This guarantees the states exist in outputs (when conditions occur).
    """
    status_out = list(status)
    p = np.asarray(p_stress_s, dtype=float)
    a = np.asarray(air_risk_s, dtype=float)

    trend = compute_trend(p, cfg.trend_k)
    rising_min_p = cfg.rising_min_p if cfg.rising_min_p is not None else cfg.stress_med

    for i in range(len(status_out)):
        # Priority 1: combined high risk
        if (p[i] >= cfg.stress_high) and (a[i] >= cfg.air_high):
            status_out[i] = "high_risk_combined"
            continue

        # Priority 2: rising stress (trend-based early warning)
        # Only if not already high phys or combined
        if (trend[i] >= cfg.rising_trend_delta) and (p[i] >= rising_min_p) and (p[i] < cfg.stress_high):
            status_out[i] = "rising_stress"
            continue

    return status_out
