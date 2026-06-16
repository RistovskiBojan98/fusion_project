from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass(frozen=True)
class SmoothingConfig:
    ema_alpha: float = 0.30
    confirm_k: int = 3
    min_dwell: int = 5

    # per-state entry confirmation requirements
    entry_confirm: dict[str, int] = None  # set default in __post_init__
    # per-state minimum dwell once you enter it
    state_min_dwell: dict[str, int] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "entry_confirm",
            self.entry_confirm
            or {
                "high_risk_combined": 1,  # instant (safety)
                "rising_stress": 2,       # needs 2 consecutive windows
            },
        )
        object.__setattr__(
            self,
            "state_min_dwell",
            self.state_min_dwell
            or {
                "high_risk_combined": 2,
                "rising_stress": 4,       # hold it for a bit (prevents flip-flop)
            },
        )



def ema_smooth(values: List[float], alpha: float) -> List[float]:
    """
    Exponential moving average smoothing.
    alpha in (0,1]: higher reacts faster, lower smooths more.
    """
    if not values:
        return []
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out


def hysteresis_smooth_status(statuses: List[str], cfg: SmoothingConfig = SmoothingConfig()) -> List[str]:
    if not statuses:
        return []

    current = statuses[0]
    dwell = 1
    candidate: Optional[str] = None
    candidate_count = 0
    out = [current]

    for s in statuses[1:]:
        if s == current:
            dwell += 1
            candidate = None
            candidate_count = 0
            out.append(current)
            continue

        # determine dwell rule for CURRENT state
        current_min_dwell = cfg.state_min_dwell.get(current, cfg.min_dwell)
        if dwell < current_min_dwell:
            dwell += 1
            out.append(current)
            continue

        # determine confirm rule for TARGET state
        confirm_k = cfg.entry_confirm.get(s, cfg.confirm_k)

        if candidate is None or s != candidate:
            candidate = s
            candidate_count = 1
        else:
            candidate_count += 1

        if candidate_count >= confirm_k:
            current = candidate
            dwell = 1
            candidate = None
            candidate_count = 0
            out.append(current)
        else:
            dwell += 1
            out.append(current)

    return out

