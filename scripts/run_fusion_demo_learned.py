import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import matplotlib.pyplot as plt  # noqa: E402
except ImportError:
    plt = None

from src.air_quality_loader import load_uci_air_quality  # noqa: E402
from src.air_risk import AirRiskConfig, compute_air_risk_score  # noqa: E402
from src.fusion_action_map import attach_actions  # noqa: E402
from src.fusion_features import FusionFeatureConfig, build_fusion_features  # noqa: E402
from src.physio_infer import predict_p_stress_calibrated  # noqa: E402
from src.smoothing import SmoothingConfig, ema_smooth, hysteresis_smooth_status  # noqa: E402
from src.state_postprocess import SpecialStateConfig, enforce_special_states  # noqa: E402


def main() -> None:
    feat_path = PROJECT_ROOT / "outputs" / "wesad_window_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Missing WESAD features CSV: {feat_path}\n"
            "Run: py scripts/build_wesad_features.py"
        )
    df_feat = pd.read_csv(feat_path)

    phys_path = PROJECT_ROOT / "outputs" / "models" / "physio_rf_strict_calibrated_bundle.joblib"
    if not phys_path.exists():
        raise FileNotFoundError(
            f"Missing strict physio bundle: {phys_path}\n"
            "Run: py scripts/train_physio_model_strict.py"
        )
    phys_bundle = joblib.load(phys_path)
    df_feat["p_stress"] = predict_p_stress_calibrated(phys_bundle, df_feat)

    df_aq = load_uci_air_quality()
    air_risk = compute_air_risk_score(df_aq, cfg=AirRiskConfig())
    df_air = pd.DataFrame({"air_risk": air_risk}).dropna()
    if df_air.empty:
        raise RuntimeError("Air-risk series is empty after cleaning.")
    air_vals = df_air["air_risk"].to_numpy()
    df_feat["air_risk"] = air_vals[np.arange(len(df_feat)) % len(air_vals)]

    fusion_path = PROJECT_ROOT / "outputs" / "models" / "fusion_decision_tree.joblib"
    if not fusion_path.exists():
        raise FileNotFoundError(
            f"Missing fusion tree: {fusion_path}\n"
            "Run: py scripts/train_fusion_model.py"
        )
    fusion_bundle = joblib.load(fusion_path)
    tree = fusion_bundle["model"]
    feat_cols = fusion_bundle["feature_columns"]

    smooth_cfg = SmoothingConfig(
        ema_alpha=0.25,
        confirm_k=3,
        min_dwell=5,
    )

    df_feat["p_stress_s"] = ema_smooth(df_feat["p_stress"].astype(float).tolist(), smooth_cfg.ema_alpha)
    df_feat["air_risk_s"] = ema_smooth(df_feat["air_risk"].astype(float).tolist(), smooth_cfg.ema_alpha)

    tmp = df_feat.copy()
    tmp["p_stress"] = tmp["p_stress_s"]
    tmp["air_risk"] = tmp["air_risk_s"]

    X = build_fusion_features(tmp, FusionFeatureConfig(trend_k=3, include_interactions=True))
    X = X[feat_cols]

    t_med = float(phys_bundle.get("threshold_f1", 0.24))
    t_high = float(min(0.95, t_med + 0.15))

    df_feat["status_raw"] = tree.predict(X)
    special_cfg = SpecialStateConfig(
        stress_med=t_med,
        stress_high=t_high,
        air_high=0.65,
        trend_k=3,
        rising_trend_delta=0.10,
    )

    df_feat["status_enforced"] = enforce_special_states(
        status=df_feat["status_raw"].astype(str).tolist(),
        p_stress_s=df_feat["p_stress_s"].astype(float).to_numpy(),
        air_risk_s=df_feat["air_risk_s"].astype(float).to_numpy(),
        cfg=special_cfg,
    )
    df_feat["status"] = hysteresis_smooth_status(
        df_feat["status_enforced"].astype(str).tolist(),
        smooth_cfg,
    )

    attach_actions(df_feat, status_col="status")

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "fusion_demo_results_learned_smoothed.csv"
    df_feat.to_csv(out_csv, index=False)
    print("Saved learned+smoothed fusion results:", out_csv)

    status_counts = df_feat["status"].value_counts()
    print("\nStatus distribution (learned+smoothed):")
    print(status_counts)

    raw = df_feat["status_raw"].astype(str).to_list()
    smoothed = df_feat["status"].astype(str).to_list()
    raw_flips = sum(raw[i] != raw[i - 1] for i in range(1, len(raw)))
    smoothed_flips = sum(smoothed[i] != smoothed[i - 1] for i in range(1, len(smoothed)))
    print(f"\nFlip count: raw={raw_flips} vs smoothed={smoothed_flips} (lower is better)")

    if plt is None:
        print("\nmatplotlib is not installed; skipped learned fusion plots.")
    else:
        fig_dir = PROJECT_ROOT / "outputs" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 4))
        status_counts.plot(kind="bar")
        plt.title("Fusion status distribution (learned + hysteresis)")
        plt.xlabel("status")
        plt.ylabel("count")
        plt.tight_layout()
        fig1 = fig_dir / "fusion_status_counts_learned_smoothed.png"
        plt.savefig(fig1, dpi=150)
        plt.close()
        print("Saved plot:", fig1)

        n = min(200, len(df_feat))
        plt.figure(figsize=(12, 4))
        plt.plot(df_feat["p_stress"].iloc[:n].to_numpy(), label="p_stress (cal)")
        plt.plot(df_feat["p_stress_s"].iloc[:n].to_numpy(), label="p_stress (EMA)")
        plt.plot(df_feat["air_risk"].iloc[:n].to_numpy(), label="air_risk")
        plt.plot(df_feat["air_risk_s"].iloc[:n].to_numpy(), label="air_risk (EMA)")
        plt.title("Signals (raw vs smoothed)")
        plt.xlabel("Window index")
        plt.ylabel("Value (0-1)")
        plt.legend()
        plt.tight_layout()
        fig2 = fig_dir / "fusion_signals_learned_smoothed.png"
        plt.savefig(fig2, dpi=150)
        plt.close()
        print("Saved plot:", fig2)

    print("\nSmoothing config:", smooth_cfg)


if __name__ == "__main__":
    main()
