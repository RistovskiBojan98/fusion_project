import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.wesad_loader import WesadConfig, list_subject_ids, load_subject_raw, get_chest_signals  # noqa: E402
from src.physio_features import WindowConfig, extract_windowed_features  # noqa: E402


def main():
    wesad_cfg = WesadConfig()
    win_cfg = WindowConfig(window_seconds=60.0, overlap=0.5)

    subjects = list_subject_ids(wesad_cfg)
    print("Subjects:", subjects)

    all_frames = []

    for sid in subjects:
        print(f"\n=== Processing {sid} ===")
        raw = load_subject_raw(sid, wesad_cfg)
        ecg, resp, labels, fs = get_chest_signals(raw)

        # WESAD chest ECG/RESP share same fs (your output confirms 700 Hz)
        df_feat = extract_windowed_features(
            ecg=ecg,
            resp=resp,
            labels=labels,
            fs=float(fs["ecg"]),
            win_cfg=win_cfg,
        )
        df_feat.insert(0, "subject_id", sid)

        print(f"{sid}: windows={len(df_feat)}  stress={int((df_feat['label_stress']==1).sum())}  non_stress={int((df_feat['label_stress']==0).sum())}")
        all_frames.append(df_feat)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.concat(all_frames, ignore_index=True)
    out_path = out_dir / "wesad_window_features.csv"
    df_all.to_csv(out_path, index=False)

    print("\n✅ Saved features to:", out_path)
    print("Shape:", df_all.shape)
    print("\nLabel distribution:")
    print(df_all["label_stress"].value_counts())


if __name__ == "__main__":
    main()
