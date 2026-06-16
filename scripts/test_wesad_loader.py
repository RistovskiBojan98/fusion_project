import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.wesad_loader import (
    WesadConfig,
    list_subject_ids,
    load_subject_raw,
    get_chest_signals,
    wesad_label_map,
    labels_to_binary_stress,
)


def main():
    cfg = WesadConfig()

    subjects = list_subject_ids(cfg)
    print("Found subjects:", subjects[:10], f"(total={len(subjects)})")
    if not subjects:
        raise RuntimeError(
            f"No subject folders found in {cfg.root_dir.resolve()}. "
            f"Expected e.g. data/wesad/S2/S2.pkl"
        )

    subject_id = subjects[0]
    print("\nLoading subject:", subject_id)

    raw = load_subject_raw(subject_id, cfg)
    ecg, resp, labels, fs = get_chest_signals(raw)

    print("\nSignals loaded:")
    print("ECG shape:", ecg.shape, "fs:", fs["ecg"])
    print("RESP shape:", resp.shape, "fs:", fs["resp"])
    print("Labels shape:", labels.shape)

    # Basic label distribution
    lm = wesad_label_map()
    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution (raw):")
    for u, c in zip(unique, counts):
        print(f"  {int(u)} ({lm.get(int(u), 'unknown')}): {int(c)}")

    y_bin = labels_to_binary_stress(labels)
    u2, c2 = np.unique(y_bin, return_counts=True)
    print("\nBinary label distribution (-1 undefined, 0 non-stress, 1 stress):")
    for u, c in zip(u2, c2):
        print(f"  {int(u)}: {int(c)}")

    print("\n✅ WESAD loader works. Next we can start windowing + feature extraction.")


if __name__ == "__main__":
    main()
