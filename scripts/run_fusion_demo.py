import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import joblib  # noqa: E402

try:
    import matplotlib.pyplot as plt  # noqa: E402
except ImportError:
    plt = None

from src.air_quality_loader import load_uci_air_quality  # noqa: E402
from src.air_risk import AirRiskConfig, compute_air_risk_score  # noqa: E402
from src.fusion_policy import FusionThresholds, decide_action  # noqa: E402
from src.physio_infer import predict_p_stress_calibrated  # noqa: E402


def main() -> None:
    feat_path = PROJECT_ROOT / "outputs" / "wesad_window_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Missing WESAD features CSV: {feat_path}\n"
            "Run: py scripts/build_wesad_features.py"
        )
    df_feat = pd.read_csv(feat_path)

    bundle_path = PROJECT_ROOT / "outputs" / "models" / "physio_rf_strict_calibrated_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing strict calibrated bundle: {bundle_path}\n"
            "Run: py scripts/train_physio_model_strict.py"
        )
    bundle = joblib.load(bundle_path)

    df_feat["p_stress"] = predict_p_stress_calibrated(bundle, df_feat)

    df_aq = load_uci_air_quality()
    air_cfg = AirRiskConfig()
    air_risk = compute_air_risk_score(df_aq, cfg=air_cfg)
    df_air = pd.DataFrame({"air_risk": air_risk}).dropna()
    if df_air.empty:
        raise RuntimeError("Air risk series is empty after cleaning.")

    # Scenario alignment: WESAD and UCI Air Quality are not synchronized, so the
    # demo cycles air-risk values across WESAD windows as contextual augmentation.
    air_vals = df_air["air_risk"].to_numpy()
    df_feat["air_risk"] = air_vals[np.arange(len(df_feat)) % len(air_vals)]

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

        res = decide_action(
            p_stress=p,
            air_risk=a,
            p_stress_history=np.asarray(p_hist, dtype=float),
            th=th,
        )

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
    print("Saved fusion demo results:", out_csv)

    status_counts = df_out["status"].value_counts()
    print("\nStatus distribution:")
    print(status_counts)

    if plt is None:
        print("\nmatplotlib is not installed; skipped fusion demo plots.")
    else:
        fig_dir = PROJECT_ROOT / "outputs" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        n = min(200, len(df_out))

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
        print("Saved plot:", fig1)

        plt.figure(figsize=(10, 4))
        status_counts.plot(kind="bar")
        plt.title("Fusion status distribution (strict)")
        plt.xlabel("status")
        plt.ylabel("count")
        plt.tight_layout()
        fig2 = fig_dir / "fusion_demo_status_counts_strict.png"
        plt.savefig(fig2, dpi=150)
        plt.close()
        print("Saved plot:", fig2)

    print("\nUsed thresholds:")
    print(f"  stress_med (tuned): {t_med:.3f}")
    print(f"  stress_high:        {t_high:.3f}")
    print("  air_med:            0.400")
    print("  air_high:           0.650")
    print("\nEvaluation frame: scenario-based contextual augmentation, not causal validation.")


if __name__ == "__main__":
    main()
