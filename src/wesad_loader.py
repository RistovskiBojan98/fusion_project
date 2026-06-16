from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WesadConfig:
    """
    Minimal configuration for loading WESAD.

    Assumptions (typical WESAD structure):
      data/wesad/
        S2/S2.pkl
        S3/S3.pkl
        ...

    We will primarily use:
      - chest ECG (for HR/HRV later)
      - chest respiration (RESP) (for RR later)
      - labels (baseline/stress/amusement/meditation/etc.)
    """
    root_dir: Path = PROJECT_ROOT / "data" / "wesad"
    subject_prefix: str = "S"
    pkl_suffix: str = ".pkl"

    # If True, returns only ECG/RESP/label for speed & simplicity
    minimal_signals_only: bool = True


def list_subject_ids(cfg: WesadConfig = WesadConfig()) -> List[str]:
    """
    Return available subject IDs like ["S2", "S3", ...] based on folders found in cfg.root_dir.
    """
    if not cfg.root_dir.exists():
        raise FileNotFoundError(f"WESAD root_dir not found: {cfg.root_dir.resolve()}")

    subjects = []
    for p in cfg.root_dir.iterdir():
        if p.is_dir() and p.name.startswith(cfg.subject_prefix):
            subjects.append(p.name)

    subjects.sort(key=lambda s: int(s.replace(cfg.subject_prefix, "")) if s.replace(cfg.subject_prefix, "").isdigit() else s)
    return subjects


def _subject_pkl_path(subject_id: str, cfg: WesadConfig) -> Path:
    """
    WESAD usually has <root>/<subject>/<subject>.pkl
    """
    return cfg.root_dir / subject_id / f"{subject_id}{cfg.pkl_suffix}"


def load_subject_raw(subject_id: str, cfg: WesadConfig = WesadConfig()) -> Dict:
    """
    Load a subject's raw pickle dict.
    """
    pkl_path = _subject_pkl_path(subject_id, cfg)
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Subject pickle not found: {pkl_path.resolve()}\n"
            f"Expected structure: data/wesad/<S#>/<S#>.pkl"
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")  # latin1 is common for older pickles
    return data


def get_chest_signals(
    subject_data: Dict,
    minimal_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Extract chest ECG, chest respiration, and labels as numpy arrays, plus sampling rates.

    Returns:
      ecg: 1D np.ndarray
      resp: 1D np.ndarray
      labels: 1D np.ndarray (same sample rate as chest signals)
      fs: dict with sampling rates (e.g., {"ecg": 700, "resp": 700})
    """
    # WESAD format is typically:
    # data['signal']['chest']['ECG'], data['signal']['chest']['Resp']
    # data['label'] for class labels
    try:
        chest = subject_data["signal"]["chest"]
    except Exception as e:
        raise KeyError("Could not find subject_data['signal']['chest']. Dataset structure differs.") from e

    # ECG key can be 'ECG' and respiration key can be 'Resp' or 'RESP'
    ecg_key = "ECG" if "ECG" in chest else None
    if ecg_key is None:
        raise KeyError(f"Chest ECG not found. Available chest keys: {list(chest.keys())}")

    resp_key = None
    for k in ["Resp", "RESP", "resp", "Respiration"]:
        if k in chest:
            resp_key = k
            break
    if resp_key is None:
        raise KeyError(f"Chest respiration not found. Available chest keys: {list(chest.keys())}")

    ecg = np.asarray(chest[ecg_key]).squeeze()
    resp = np.asarray(chest[resp_key]).squeeze()

    labels = np.asarray(subject_data.get("label", None))
    if labels is None or len(labels) == 0:
        raise KeyError("Labels not found at subject_data['label'].")

    # Sampling rates: WESAD often provides subject_data['subject']['sampling_rate']
    # If not present, use defaults (WESAD chest is commonly 700 Hz).
    fs = {}
    sr = None
    # Try multiple known locations
    for path_try in [
        ("subject", "sampling_rate"),
        ("subject", "sampling_rates"),
        ("sampling_rate",),
        ("sampling_rates",),
    ]:
        cur = subject_data
        ok = True
        for key in path_try:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            sr = cur
            break

    # Interpret sr if possible
    # Sometimes sr is a dict like {'chest': {'ECG': 700, 'Resp': 700}, 'wrist': {...}}
    if isinstance(sr, dict):
        # Try to find chest ECG/RESP rates
        try:
            chest_sr = sr.get("chest", sr.get("Chest", None))
            if isinstance(chest_sr, dict):
                # if dict keyed by signal name
                fs["ecg"] = float(chest_sr.get(ecg_key, 700))
                fs["resp"] = float(chest_sr.get(resp_key, 700))
            else:
                # unknown structure
                fs["ecg"] = 700.0
                fs["resp"] = 700.0
        except Exception:
            fs["ecg"] = 700.0
            fs["resp"] = 700.0
    else:
        # If no sr found, use default
        fs["ecg"] = 700.0
        fs["resp"] = 700.0

    # Sanity: labels should be aligned with chest sample count (usually same length)
    # If labels differ, we still return but warn via exception message would be too harsh;
    # leave alignment to downstream processing.
    return ecg, resp, labels, fs


def wesad_label_map() -> Dict[int, str]:
    """
    Common WESAD label mapping (as used in many public implementations).
    Exact meaning can vary slightly; we will mainly use baseline vs stress later.

    Typical:
      0 = undefined / not used
      1 = baseline
      2 = stress
      3 = amusement
      4 = meditation
    """
    return {0: "undefined", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def labels_to_binary_stress(labels: np.ndarray) -> np.ndarray:
    """
    Convert multiclass labels to binary:
      1 for stress, 0 for non-stress (baseline + amusement + meditation)
    Unknown/undefined (0) is set to -1 (so you can filter it out).
    """
    y = np.full_like(labels, fill_value=-1, dtype=int)
    y[labels == 2] = 1
    y[(labels == 1) | (labels == 3) | (labels == 4)] = 0
    return y


def window_signal(
    x: np.ndarray,
    fs: float,
    window_seconds: float = 60.0,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Turn a 1D array into overlapping windows (N_windows, window_samples).
    """
    if x.ndim != 1:
        x = x.squeeze()
    win = int(round(window_seconds * fs))
    if win <= 0:
        raise ValueError("window_seconds*fs must be > 0")
    step = int(round(win * (1.0 - overlap)))
    step = max(step, 1)

    windows = []
    for start in range(0, len(x) - win + 1, step):
        windows.append(x[start : start + win])
    if not windows:
        return np.empty((0, win))
    return np.stack(windows, axis=0)


def window_labels_majority(
    labels: np.ndarray,
    fs: float,
    window_seconds: float = 60.0,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Window labels aligned to signal sampling and take majority label per window.
    """
    win = int(round(window_seconds * fs))
    step = int(round(win * (1.0 - overlap)))
    step = max(step, 1)

    y_win = []
    for start in range(0, len(labels) - win + 1, step):
        chunk = labels[start : start + win]
        # majority vote ignoring -1 if present
        vals, counts = np.unique(chunk, return_counts=True)
        # pick the most frequent
        y_win.append(vals[np.argmax(counts)])
    return np.asarray(y_win)
