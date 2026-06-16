# scripts/test_air_risk.py
#
# Test script for:
# - loading the UCI Air Quality dataset
# - computing an air risk score
# - printing sanity stats
# - saving a few quick plots to outputs/figures/
#
# Run from project root:
#   py scripts/test_air_risk.py
# (Works even if run from scripts/ due to PROJECT_ROOT injection below.)

import sys
from pathlib import Path

# Ensure project root is on sys.path so "src" imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.air_quality_loader import load_uci_air_quality  # noqa: E402
from src.air_risk import AirRiskConfig, attach_air_risk, available_risk_inputs  # noqa: E402


def _ensure_outputs_dirs() -> Path:
    out_dir = PROJECT_ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    out_dir = _ensure_outputs_dirs()

    print("=== Loading UCI Air Quality ===")
    df_aq = load_uci_air_quality()
    print("Loaded:", df_aq.shape)
    print("Date range:", df_aq.index.min(), "→", df_aq.index.max())
    print("\nAvailable expected columns:")
    print(available_risk_inputs(df_aq))

    print("\n=== Computing air risk ===")
    cfg = AirRiskConfig()
    df = attach_air_risk(df_aq, cfg=cfg)

    # Basic stats
    risk = df[cfg.risk_col]
    print("\nAir risk score summary:")
    print(risk.describe())

    # Confirm component columns exist
    component_cols = [c for c in ["risk_CO", "risk_NO2", "risk_NOx", "risk_weather"] if c in df.columns]
    print("\nComponent columns present:", component_cols)

    # Save a time-series plot (use a subset so it renders quickly)
    print("\n=== Plotting & saving figures ===")
    plt.figure(figsize=(12, 4))
    df[cfg.risk_col].iloc[: 24 * 30].plot()  # first ~30 days (hourly)
    plt.title("Air Risk Score (first ~30 days)")
    plt.xlabel("Time")
    plt.ylabel("Risk (0-1)")
    plt.tight_layout()
    fig_path = out_dir / "air_risk_timeseries_first_30_days.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print("Saved:", fig_path)

    # Histogram of risk
    plt.figure(figsize=(8, 4))
    risk.plot(kind="hist", bins=30)
    plt.title("Distribution of Air Risk Score")
    plt.xlabel("Risk (0-1)")
    plt.ylabel("Count")
    plt.tight_layout()
    fig_path = out_dir / "air_risk_hist.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print("Saved:", fig_path)

    # Correlation sanity check: risk vs pollutants (if available)
    # We'll compute correlations with the "GT" pollutant columns when present.
    candidate_pollutants = [c for c in ["CO(GT)", "NO2(GT)", "NOx(GT)"] if c in df.columns]
    if candidate_pollutants:
        corr = df[[cfg.risk_col] + candidate_pollutants].corr(numeric_only=True)[cfg.risk_col].drop(cfg.risk_col)
        print("\nCorrelation with risk (sanity check):")
        print(corr.sort_values(ascending=False))

        plt.figure(figsize=(8, 4))
        corr.sort_values(ascending=False).plot(kind="bar")
        plt.title("Correlation of Pollutants with Air Risk Score")
        plt.xlabel("Pollutant")
        plt.ylabel("Correlation")
        plt.tight_layout()
        fig_path = out_dir / "air_risk_pollutant_correlations.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print("Saved:", fig_path)
    else:
        print("\nNo GT pollutant columns found for correlation plot.")

    # Optional: save a tiny CSV sample with risk columns for inspection
    sample_path = PROJECT_ROOT / "outputs" / "air_quality_with_risk_sample.csv"
    df[[cfg.risk_col] + component_cols].head(200).to_csv(sample_path, index=True)
    print("Saved sample CSV:", sample_path)


if __name__ == "__main__":
    main()
