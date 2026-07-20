import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.fusion_action_map import attach_actions  # noqa: E402
from src.fusion_features import compute_trend  # noqa: E402


SEVERITY = {
    "normal": 0,
    "rising_stress": 1,
    "high_air_risk": 2,
    "elevated_context_risk": 2,
    "high_physiological_risk": 3,
    "high_risk_combined": 4,
}


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {source}: {missing}. "
            "Run: py scripts/run_fusion_demo.py"
        )


def status_distribution_json(statuses: pd.Series) -> str:
    return json.dumps(statuses.astype(str).value_counts().sort_index().to_dict(), sort_keys=True)


def action_distribution_json(actions: pd.Series) -> str:
    return json.dumps(actions.astype(str).value_counts().sort_index().to_dict(), sort_keys=True)


def physiology_only_status(df: pd.DataFrame, stress_med: float, stress_high: float) -> list[str]:
    p = df["p_stress"].astype(float).to_numpy()
    trend = compute_trend(p, k=3)
    out = []
    for prob, delta in zip(p, trend):
        if prob >= stress_high:
            out.append("high_physiological_risk")
        elif prob >= stress_med and delta >= 0.10:
            out.append("rising_stress")
        else:
            out.append("normal")
    return out


def air_only_status(df: pd.DataFrame, air_high: float) -> list[str]:
    air = df["air_risk"].astype(float).to_numpy()
    return ["high_air_risk" if value >= air_high else "normal" for value in air]


def weighted_score_status(
    df: pd.DataFrame,
    stress_high: float,
    air_high: float,
    p_weight: float = 0.5,
    air_weight: float = 0.5,
) -> list[str]:
    p = df["p_stress"].astype(float).to_numpy()
    air = df["air_risk"].astype(float).to_numpy()
    score = (p_weight * p) + (air_weight * air)
    out = []

    for prob, air_value, combined in zip(p, air, score):
        if combined >= 0.65:
            if prob >= stress_high and air_value >= air_high:
                out.append("high_risk_combined")
            elif prob >= air_value:
                out.append("high_physiological_risk")
            else:
                out.append("high_air_risk")
        elif combined >= 0.45:
            out.append("elevated_context_risk")
        else:
            out.append("normal")
    return out


def frame_strategy(df_base: pd.DataFrame, method_name: str, statuses: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df_base.index)
    out["method_name"] = method_name
    out["status"] = statuses
    attach_actions(out, status_col="status")
    return out


def stability(statuses: pd.Series) -> tuple[int, float]:
    if len(statuses) <= 1:
        return 0, 1.0
    values = statuses.astype(str).to_list()
    changes = sum(values[i] != values[i - 1] for i in range(1, len(values)))
    return changes, 1.0 - (changes / (len(values) - 1))


def summarize_strategy(
    method_name: str,
    strategy_df: pd.DataFrame,
    physio_actions: pd.Series,
    physio_statuses: pd.Series,
) -> dict:
    statuses = strategy_df["status"].astype(str)
    actions = strategy_df["action"].astype(str)
    rationales = strategy_df["rationale"].astype(str)
    n = len(strategy_df)
    severities = statuses.map(SEVERITY).fillna(0).astype(float)
    non_normal = statuses != "normal"
    high_risk = severities >= 2
    status_changes, stability_pct = stability(statuses)
    action_changes, action_stability_pct = stability(actions)

    action_diff = int((actions.reset_index(drop=True) != physio_actions.reset_index(drop=True)).sum())
    status_diff = int((statuses.reset_index(drop=True) != physio_statuses.reset_index(drop=True)).sum())
    rationale_present = rationales.str.strip().ne("").sum()

    return {
        "method_name": method_name,
        "evaluation_frame": "scenario_based_not_causal_validation",
        "n_windows": int(n),
        "non_normal_alert_count": int(non_normal.sum()),
        "non_normal_alert_pct": float(non_normal.mean() * 100.0),
        "high_risk_alert_count": int(high_risk.sum()),
        "high_risk_alert_pct": float(high_risk.mean() * 100.0),
        "action_changes_compared_with_physio_only": action_diff,
        "action_changes_compared_with_physio_only_pct": float(action_diff / max(n, 1) * 100.0),
        "action_differences_vs_physio_only": action_diff,
        "action_differences_vs_physio_only_pct": float(action_diff / max(n, 1) * 100.0),
        "status_differences_vs_physio_only": status_diff,
        "status_differences_vs_physio_only_pct": float(status_diff / max(n, 1) * 100.0),
        "status_change_count": int(status_changes),
        "decision_stability_pct": float(stability_pct * 100.0),
        "action_change_count": int(action_changes),
        "action_stability_pct": float(action_stability_pct * 100.0),
        "average_severity_score": float(severities.mean()),
        "decisions_with_rationale_pct": float(rationale_present / max(n, 1) * 100.0),
        "status_distribution": status_distribution_json(statuses),
        "action_distribution": action_distribution_json(actions),
    }


def main() -> None:
    in_path = PROJECT_ROOT / "outputs" / "fusion_demo_results_strict.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing fusion demo output: {in_path}\n"
            "Run: py scripts/run_fusion_demo.py"
        )

    df = pd.read_csv(in_path)
    require_columns(df, ["p_stress", "air_risk", "status", "action", "rationale"], in_path)

    bundle_path = PROJECT_ROOT / "outputs" / "models" / "physio_rf_strict_calibrated_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing physiology bundle: {bundle_path}\n"
            "Run: py scripts/train_physio_model_strict.py"
        )
    bundle = joblib.load(bundle_path)

    stress_med = float(bundle.get("threshold_f1", 0.24))
    stress_high = float(min(0.95, stress_med + 0.15))
    air_high = 0.65

    strategies = {}
    strategies["physiology_only"] = frame_strategy(
        df,
        "physiology_only",
        physiology_only_status(df, stress_med=stress_med, stress_high=stress_high),
    )
    strategies["air_risk_only"] = frame_strategy(
        df,
        "air_risk_only",
        air_only_status(df, air_high=air_high),
    )
    strategies["weighted_score"] = frame_strategy(
        df,
        "weighted_score",
        weighted_score_status(df, stress_high=stress_high, air_high=air_high),
    )

    rule_based = df[["status", "action", "rationale"]].copy()
    rule_based["method_name"] = "rule_based_fusion_policy"
    strategies["rule_based_fusion_policy"] = rule_based

    physio_actions = strategies["physiology_only"]["action"]
    physio_statuses = strategies["physiology_only"]["status"]
    rows = [
        summarize_strategy(name, strategy, physio_actions, physio_statuses)
        for name, strategy in strategies.items()
    ]

    out_path = PROJECT_ROOT / "outputs" / "fusion_comparison_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved scenario-based fusion comparison metrics to:", out_path)
    print(pd.DataFrame(rows)[["method_name", "high_risk_alert_count", "decision_stability_pct", "average_severity_score"]])


if __name__ == "__main__":
    main()
