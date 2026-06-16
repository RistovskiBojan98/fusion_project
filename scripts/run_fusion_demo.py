import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.air_quality_loader import load_uci_air_quality  # noqa: E402
from src.air_risk import AirRiskConfig, compute_air_risk_score  # noqa: E402
from src.fusion_policy import FusionThresholds, decide_action  # noqa: E402
from src.physio_infer import predict_p_stress_calibrated  # noqa: E402


def main() -> None:
    # Load WESAD features
    feat_path = PROJECT_ROOT / "outputs" / "wesad_window_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("Missing WESAD features CSV. Run build_wesad_features.py first.")
    df_feat = pd.read_csv(feat_path)

    # Load strict calibrated physiology bundle
    bundle_path = PROJECT_ROOT / "outputs" / "models" / "physio_rf_strict_calibrated_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            "Missing strict calibrated bundle. Run: py scripts/train_physio_model_strict.py"
        )
    bundle = joblib.load(bundle_path)

    # Compute calibrated stress probabilities
    p_stress = predict_p_stress_calibrated(bundle, df_feat)
    df_feat["p_stress"] = p_stress

    # Load air quality and compute air risk score
    df_aq = load_uci_air_quality()
    air_cfg = AirRiskConfig()
    air_risk = compute_air_risk_score(df_aq, cfg=air_cfg)
    df_air = pd.DataFrame({"air_risk": air_risk}).dropna()

    if df_air.empty:
        raise RuntimeError("Air risk series is empty after cleaning.")

    # Scenario alignment (synthetic): cycle through air risk values
    air_vals = df_air["air_risk"].to_numpy()
    df_feat["air_risk"] = air_vals[np.arange(len(df_feat)) % len(air_vals)]

    # Use tuned threshold as "medium", and set "high" a bit higher for conservative alerts.
    # This is realistic: one threshold is optimized, another is a stricter alarm level.
    t_med = float(bundle["threshold_f1"])
    t_high = float(min(0.95, t_med + 0.15))

    th = FusionThresholds(
        stress_med=t_med,
        stress_high=t_high,
        air_med=0.40,
        air_high=0.65,
        stress_trend_delta_high=0.10,
        trend_k=3,
    )

    statuses, actions, rationales, stress_levels, air_levels = [], [], [], [], []
    p_hist = []

    for i in range(len(df_feat)):
        p = float(df_feat.loc[i, "p_stress"])
        a = float(df_feat.loc[i, "air_risk"])

        p_hist_arr = np.asarray(p_hist, dtype=float)
        res = decide_action(p_stress=p, air_risk=a, p_stress_history=p_hist_arr, th=th)

        statuses.append(res["status"])
        actions.append(res["action"])
        rationales.append(res["rationale"])
        stress_levels.append(res["stress_level"])
        air_levels.append(res["air_level"])

        p_hist.append(p)
        if len(p_hist) > 20:
            p_hist = p_hist[-20:]

    df_out = df_feat.copy()
    df_out["status"] = statuses
    df_out["action"] = actions
    df_out["rationale"] = rationales
    df_out["stress_level"] = stress_levels
    df_out["air_level"] = air_levels

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "fusion_demo_results_strict.csv"
    df_out.to_csv(out_csv, index=False)
    print("✅ Saved fusion demo results:", out_csv)

    # Plots
    n = min(200, len(df_out))
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(df_out["p_stress"].iloc[:n].to_numpy(), label="p_stress_calibrated")
    plt.plot(df_out["air_risk"].iloc[:n].to_numpy(), label="air_risk")
    plt.axhline(t_med, linestyle="--", label=f"stress_med (tuned)={t_med:.2f}")
    plt.axhline(t_high, linestyle="--", label=f"stress_high={t_high:.2f}")
    plt.title("Fusion demo (strict): calibrated stress + air risk")
    plt.xlabel("Window index")
    plt.ylabel("Value (0-1)")
    plt.legend()
    plt.tight_layout()
    fig1 = fig_dir / "fusion_demo_signals_strict.png"
    plt.savefig(fig1, dpi=150)
    plt.close()
    print("✅ Saved plot:", fig1)

    status_counts = df_out["status"].value_counts()
    print("\nStatus distribution:")
    print(status_counts)

    plt.figure(figsize=(10, 4))
    status_counts.plot(kind="bar")
    plt.title("Fusion status distribution (strict)")
    plt.xlabel("status")
    plt.ylabel("count")
    plt.tight_layout()
    fig2 = fig_dir / "fusion_demo_status_counts_strict.png"
    plt.savefig(fig2, dpi=150)
    plt.close()
    print("✅ Saved plot:", fig2)

    print("\nUsed thresholds:")
    print(f"  stress_med (tuned): {t_med:.3f}")
    print(f"  stress_high:        {t_high:.3f}")
    print("  air_med:            0.400")
    print("  air_high:           0.650")


if __name__ == "__main__":
    main()
