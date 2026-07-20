from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class FusionThresholds:
    # Physiology probability thresholds (calibrated).
    stress_med: float = 0.50
    stress_high: float = 0.70

    # Air-risk thresholds.
    air_med: float = 0.40
    air_high: float = 0.65

    # Trend threshold: if stress probability rises by at least this amount over K windows.
    stress_trend_delta_high: float = 0.10
    trend_k: int = 3


def level_from_thresholds(x: float, med: float, high: float) -> RiskLevel:
    if x >= high:
        return RiskLevel.HIGH
    if x >= med:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def stress_trend(p_stress_series: np.ndarray, k: int) -> float:
    """
    Compute a simple trend: last value minus mean of previous k-1 values.
    If the series is too short, return 0.
    """
    if p_stress_series.size < k:
        return 0.0
    last = float(p_stress_series[-1])
    prev = p_stress_series[-k:-1]
    if prev.size == 0:
        return 0.0
    return last - float(np.mean(prev))


def decide_action(
    p_stress: float,
    air_risk: float,
    p_stress_history: np.ndarray | None = None,
    th: FusionThresholds = FusionThresholds(),
) -> Dict[str, str]:
    """
    Explainable fusion: combine physiology and air context into an action recommendation.

    Returns a dict with:
    - status: overall state label
    - action: suggested behavior
    - rationale: human-readable explanation
    - stress_level, air_level: discretized levels
    """
    stress_level = level_from_thresholds(p_stress, th.stress_med, th.stress_high)
    air_level = level_from_thresholds(air_risk, th.air_med, th.air_high)

    trend = 0.0
    rising = False
    if p_stress_history is not None and p_stress_history.size >= th.trend_k:
        trend = stress_trend(p_stress_history, th.trend_k)
        rising = trend >= th.stress_trend_delta_high

    # Core ambient-intelligence logic: context changes the response.
    # It is deliberately rule-based so the final decision can be explained.
    if stress_level == RiskLevel.HIGH and air_level == RiskLevel.HIGH:
        return {
            "status": "high_risk_combined",
            "action": "Stop/avoid outdoor exertion; move indoors; rest; consider mask if you must go out.",
            "rationale": "High physiological stress detected while outdoor air risk is high; combined load increases risk of deterioration.",
            "stress_level": stress_level.value,
            "air_level": air_level.value,
        }

    if stress_level == RiskLevel.HIGH and air_level in (RiskLevel.LOW, RiskLevel.MEDIUM):
        return {
            "status": "high_physiological_risk",
            "action": "Reduce intensity; take a break; hydrate; do 2-3 minutes of slow breathing.",
            "rationale": "High physiological stress detected but air quality is not the main driver; prioritize recovery and breathing regulation.",
            "stress_level": stress_level.value,
            "air_level": air_level.value,
        }

    if stress_level == RiskLevel.MEDIUM and air_level == RiskLevel.HIGH:
        return {
            "status": "elevated_context_risk",
            "action": "Avoid running/HIIT outdoors; choose low intensity or indoor activity.",
            "rationale": "Moderate physiological strain plus high air risk suggests exposure-driven risk; context-aware adjustment reduces load.",
            "stress_level": stress_level.value,
            "air_level": air_level.value,
        }

    if stress_level in (RiskLevel.LOW, RiskLevel.MEDIUM) and air_level == RiskLevel.HIGH:
        return {
            "status": "high_air_risk",
            "action": "Limit outdoor exposure; consider mask; keep intensity low; prefer indoors.",
            "rationale": "Air risk is high even without strong physiological stress; ambient context warrants protective behavior.",
            "stress_level": stress_level.value,
            "air_level": air_level.value,
        }

    if rising and stress_level != RiskLevel.HIGH:
        return {
            "status": "rising_stress",
            "action": "Your stress is trending up; slow down and take a short recovery break.",
            "rationale": f"Stress probability is rising ({trend:.2f}) even if not yet high; early intervention prevents deterioration.",
            "stress_level": stress_level.value,
            "air_level": air_level.value,
        }

    return {
        "status": "normal",
        "action": "No action needed. Maintain current pace and check in periodically.",
        "rationale": "Physiological stress and air context are not elevated.",
        "stress_level": stress_level.value,
        "air_level": air_level.value,
    }
