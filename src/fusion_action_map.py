from __future__ import annotations

from typing import Dict


STATUS_TO_ACTION: Dict[str, Dict[str, str]] = {
    "high_risk_combined": {
        "action": "Stop/avoid outdoor exertion; move indoors; rest; consider mask if you must go out.",
        "rationale": "High physiological stress while outdoor air risk is high; combined load increases deterioration risk.",
    },
    "high_physiological_risk": {
        "action": "Reduce intensity; take a break; hydrate; do 2-3 minutes of slow breathing.",
        "rationale": "Physiological stress is high; prioritize recovery and breathing regulation.",
    },
    "elevated_context_risk": {
        "action": "Avoid running/HIIT outdoors; choose low intensity or indoor activity.",
        "rationale": "Moderate physiological strain plus high air risk suggests exposure-driven risk; adjust behavior.",
    },
    "high_air_risk": {
        "action": "Limit outdoor exposure; consider mask; keep intensity low; prefer indoors.",
        "rationale": "Air risk is high even without strong physiological stress; context warrants protective behavior.",
    },
    "rising_stress": {
        "action": "Your stress is trending up; slow down and take a short recovery break.",
        "rationale": "Stress likelihood is rising; early intervention helps prevent deterioration.",
    },
    "normal": {
        "action": "No action needed. Maintain current pace and check in periodically.",
        "rationale": "Physiological stress and air context are not elevated.",
    },
}


def attach_actions(df, status_col: str = "status") -> None:
    """
    Add df['action'] and df['rationale'] in-place based on status.
    Unknown statuses fall back to normal.
    """
    actions = []
    rationales = []
    for status in df[status_col].astype(str).tolist():
        meta = STATUS_TO_ACTION.get(status, STATUS_TO_ACTION["normal"])
        actions.append(meta["action"])
        rationales.append(meta["rationale"])
    df["action"] = actions
    df["rationale"] = rationales
