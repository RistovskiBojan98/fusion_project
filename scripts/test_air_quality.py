import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.air_quality_loader import load_uci_air_quality


def main():
    df_aq = load_uci_air_quality()

    print("Shape:", df_aq.shape)
    print("\nColumns:")
    print(df_aq.columns.tolist())

    print("\nDate range:")
    print(df_aq.index.min(), "→", df_aq.index.max())

    print("\nMissing values (top 10):")
    print(df_aq.isna().sum().sort_values(ascending=False).head(10))

    print("\nHead:")
    print(df_aq.head())


if __name__ == "__main__":
    main()
