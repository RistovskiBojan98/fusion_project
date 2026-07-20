import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.air_quality_loader import load_uci_air_quality  # noqa: E402
from src.air_risk import AirRiskConfig, compute_air_risk_score  # noqa: E402
from src.fusion_policy import FusionThresholds, decide_action, level_from_thresholds  # noqa: E402


WEIGHT_CONFIGS = {
    "default_weights": AirRiskConfig(w_co=0.20, w_no2=0.40, w_nox=0.40),
    "equal_weights": AirRiskConfig(w_co=0.33, w_no2=0.33, w_nox=0.34),
    "no2_focused": AirRiskConfig(w_co=0.20, w_no2=0.50, w_nox=0.30),
    "nox_focused": AirRiskConfig(w_co=0.20, w_no2=0.30, w_nox=0.50),
    "co_focused": AirRiskConfig(w_co=0.50, w_no2=0.25, w_nox=0.25),
}


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {source}: {missing}. "
            "Run: py scripts/run_fusion_demo.py"
        )


def align_air_risk(df_aq: pd.DataFrame, cfg: AirRiskConfig, n_windows: int) -> np.ndarray:
    risk = compute_air_risk_score(df_aq, cfg=cfg)
    df_air = pd.DataFrame({"air_risk": risk}).dropna()
    if df_air.empty:
        raise RuntimeError("Air-risk series is empty after cleaning; cannot run sensitivity analysis.")
    values = df_air["air_risk"].to_numpy()
    return values[np.arange(n_windows) % len(values)]


def category_labels(air_values: np.ndarray, th: FusionThresholds) -> list[str]:
    return [
        level_from_thresholds(float(value), th.air_med, th.air_high).value
        for value in air_values
    ]


def run_policy(p_stress: np.ndarray, air_values: np.ndarray, th: FusionThresholds) -> pd.DataFrame:
    statuses, actions, rationales = [], [], []
    p_hist = []

    for p, air in zip(p_stress, air_values):
        res = decide_action(
            p_stress=float(p),
            air_risk=float(air),
            p_stress_history=np.asarray(p_hist, dtype=float),
            th=th,
        )
        statuses.append(res["status"])
        actions.append(res["action"])
        rationales.append(res["rationale"])

        p_hist.append(float(p))
        if len(p_hist) > 20:
            p_hist = p_hist[-20:]

    return pd.DataFrame({"status": statuses, "action": actions, "rationale": rationales})


def distribution_json(values: list[str] | pd.Series) -> str:
    return json.dumps(pd.Series(values).astype(str).value_counts().sort_index().to_dict(), sort_keys=True)


def main() -> None:
    fusion_path = PROJECT_ROOT / "outputs" / "fusion_demo_results_strict.csv"
    if not fusion_path.exists():
        raise FileNotFoundError(
            f"Missing fusion demo output: {fusion_path}\n"
            "Run: py scripts/run_fusion_demo.py"
        )
    df_fusion = pd.read_csv(fusion_path)
    require_columns(df_fusion, ["p_stress"], fusion_path)

    bundle_path = PROJECT_ROOT / "outputs" / "models" / "physio_rf_strict_calibrated_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing physiology bundle: {bundle_path}\n"
            "Run: py scripts/train_physio_model_strict.py"
        )
    bundle = joblib.load(bundle_path)

    t_med = float(bundle.get("threshold_f1", 0.24))
    t_high = float(min(0.95, t_med + 0.15))
    th = FusionThresholds(
        stress_med=t_med,
        stress_high=t_high,
        air_med=0.40,
        air_high=0.65,
        stress_trend_delta_high=0.10,
        trend_k=3,
    )

    df_aq = load_uci_air_quality()
    p_stress = df_fusion["p_stress"].astype(float).to_numpy()
    n = len(p_stress)

    results = {}
    for config_name, cfg in WEIGHT_CONFIGS.items():
        air_values = align_air_risk(df_aq, cfg, n_windows=n)
        categories = category_labels(air_values, th)
        policy_df = run_policy(p_stress, air_values, th)
        results[config_name] = {
            "cfg": cfg,
            "air_values": air_values,
            "categories": categories,
            "policy": policy_df,
        }

    default = results["default_weights"]
    default_categories = pd.Series(default["categories"]).astype(str)
    default_statuses = default["policy"]["status"].astype(str)
    default_actions = default["policy"]["action"].astype(str)

    rows = []
    for config_name, payload in results.items():
        cfg = payload["cfg"]
        air_values = payload["air_values"]
        categories = pd.Series(payload["categories"]).astype(str)
        policy = payload["policy"]
        statuses = policy["status"].astype(str)
        actions = policy["action"].astype(str)

        category_changes = int((categories != default_categories).sum())
        status_changes = int((statuses != default_statuses).sum())
        action_changes = int((actions != default_actions).sum())
        high_risk = statuses != "normal"

        rows.append(
            {
                "config_name": config_name,
                "w_co": cfg.w_co,
                "w_no2": cfg.w_no2,
                "w_nox": cfg.w_nox,
                "n_windows": int(n),
                "mean_air_risk": float(np.mean(air_values)),
                "min_air_risk": float(np.min(air_values)),
                "max_air_risk": float(np.max(air_values)),
                "air_risk_category_changes_vs_default": category_changes,
                "air_risk_category_change_pct_vs_default": float(category_changes / max(n, 1) * 100.0),
                "final_status_changes_vs_default": status_changes,
                "final_status_change_pct_vs_default": float(status_changes / max(n, 1) * 100.0),
                "action_changes_vs_default": action_changes,
                "action_change_pct_vs_default": float(action_changes / max(n, 1) * 100.0),
                "non_normal_alert_count": int(high_risk.sum()),
                "non_normal_alert_pct": float(high_risk.mean() * 100.0),
                "air_category_distribution": distribution_json(categories),
                "final_status_distribution": distribution_json(statuses),
            }
        )

    out_path = PROJECT_ROOT / "outputs" / "air_weight_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved air-weight sensitivity analysis to:", out_path)
    print(pd.DataFrame(rows)[["config_name", "air_risk_category_changes_vs_default", "final_status_changes_vs_default"]])


if __name__ == "__main__":
    main()
