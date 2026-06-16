from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    window_seconds: float = 60.0
    overlap: float = 0.5


def _window_start_indices(n: int, win: int, step: int) -> np.ndarray:
    if n < win:
        return np.array([], dtype=int)
    return np.arange(0, n - win + 1, step, dtype=int)


def window_majority_label(
    labels: np.ndarray,
    fs: float,
    window_seconds: float,
    overlap: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute majority label per window.

    Returns:
      starts: start indices (samples)
      y_win: majority label per window (same length as starts)
    """
    win = int(round(window_seconds * fs))
    step = int(round(win * (1.0 - overlap)))
    step = max(step, 1)

    starts = _window_start_indices(len(labels), win, step)
    y_win = np.empty(len(starts), dtype=int)

    for i, s in enumerate(starts):
        chunk = labels[s : s + win]
        vals, counts = np.unique(chunk, return_counts=True)
        y_win[i] = int(vals[np.argmax(counts)])
    return starts, y_win


def labels_to_binary_stress_windowed(y_win: np.ndarray) -> np.ndarray:
    """
    Map window-level labels to:
      1 = stress (label 2)
      0 = non-stress (labels 1,3,4)
     -1 = ignore (label 0 and anything else like 6,7)
    """
    out = np.full_like(y_win, fill_value=-1, dtype=int)
    out[y_win == 2] = 1
    out[(y_win == 1) | (y_win == 3) | (y_win == 4)] = 0
    return out


def _rr_intervals_from_peaks(peaks_idx: np.ndarray, fs: float) -> np.ndarray:
    """
    Convert R-peak indices to RR intervals in milliseconds.
    """
    if peaks_idx.size < 2:
        return np.array([], dtype=float)
    rr_s = np.diff(peaks_idx) / fs
    rr_ms = rr_s * 1000.0
    return rr_ms


def hrv_rmssd(rr_ms: np.ndarray) -> float:
    if rr_ms.size < 3:
        return np.nan
    diff = np.diff(rr_ms)
    return float(np.sqrt(np.mean(diff * diff)))


def hrv_sdnn(rr_ms: np.ndarray) -> float:
    if rr_ms.size < 2:
        return np.nan
    return float(np.std(rr_ms, ddof=1))


def _simple_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return x - mu
    return (x - mu) / sd


def ecg_features_from_window(ecg_win: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract ECG-based features from a window:
      - mean HR (bpm) from RR intervals
      - HR std (bpm)
      - RMSSD (ms)
      - SDNN (ms)

    R-peak detection: simple robust approach using a derivative-energy heuristic.
    Not as perfect as full ECG toolkits, but fast and works reasonably for WESAD-like clean chest ECG.
    """
    x = _simple_zscore(ecg_win)

    # Basic "energy" signal
    dx = np.diff(x, prepend=x[0])
    energy = dx * dx

    # Threshold on energy (adaptive)
    thr = np.nanpercentile(energy, 95)
    cand = np.where(energy > thr)[0]

    # If too few candidates, fail gracefully
    if cand.size < 10:
        return {
            "hr_mean_bpm": np.nan,
            "hr_std_bpm": np.nan,
            "hrv_rmssd_ms": np.nan,
            "hrv_sdnn_ms": np.nan,
            "rpeaks_count": 0.0,
        }

    # Enforce refractory period (~250ms)
    refractory = int(0.25 * fs)
    peaks = []
    last = -10**9
    for idx in cand:
        if idx - last >= refractory:
            peaks.append(idx)
            last = idx
    peaks_idx = np.asarray(peaks, dtype=int)

    rr_ms = _rr_intervals_from_peaks(peaks_idx, fs)
    if rr_ms.size < 2:
        hr_mean = np.nan
        hr_std = np.nan
    else:
        hr = 60000.0 / rr_ms  # bpm
        hr_mean = float(np.nanmean(hr))
        hr_std = float(np.nanstd(hr))

    return {
        "hr_mean_bpm": hr_mean,
        "hr_std_bpm": hr_std,
        "hrv_rmssd_ms": hrv_rmssd(rr_ms),
        "hrv_sdnn_ms": hrv_sdnn(rr_ms),
        "rpeaks_count": float(peaks_idx.size),
    }


def resp_features_from_window(resp_win: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract respiration-based features:
      - estimated breathing rate (breaths per minute)
      - respiration variability proxy (std of signal)

    Uses a simple peak-counting approach after smoothing.
    """
    x = np.asarray(resp_win, dtype=float)
    x = _simple_zscore(x)

    # Smooth with moving average (~0.5s)
    k = int(max(1, round(0.5 * fs)))
    kernel = np.ones(k) / k
    xs = np.convolve(x, kernel, mode="same")

    # Peak detection using simple local maxima
    # Refractory ~1.0s (breathing rate won't exceed ~60 bpm realistically)
    refractory = int(1.0 * fs)
    peaks = []
    for i in range(1, len(xs) - 1):
        if xs[i] > xs[i - 1] and xs[i] > xs[i + 1]:
            peaks.append(i)
    peaks = np.asarray(peaks, dtype=int)

    # Keep only prominent peaks using adaptive threshold
    if peaks.size > 0:
        amp_thr = np.nanpercentile(xs[peaks], 70)
        peaks = peaks[xs[peaks] >= amp_thr]

    # Enforce refractory
    final = []
    last = -10**9
    for idx in peaks:
        if idx - last >= refractory:
            final.append(idx)
            last = idx
    peaks_idx = np.asarray(final, dtype=int)

    # Breathing rate = peaks per minute in this window
    dur_s = len(resp_win) / fs
    if dur_s <= 0:
        rr_bpm = np.nan
    else:
        rr_bpm = float((peaks_idx.size / dur_s) * 60.0)

    return {
        "resp_rate_bpm": rr_bpm,
        "resp_std": float(np.nanstd(xs)),
        "resp_peaks_count": float(peaks_idx.size),
    }


def extract_windowed_features(
    ecg: np.ndarray,
    resp: np.ndarray,
    labels: np.ndarray,
    fs: float,
    win_cfg: WindowConfig = WindowConfig(),
) -> pd.DataFrame:
    """
    Produce one row per window with:
      - window start/end timestamps in samples
      - ECG features (HR/HRV)
      - Respiration features (RR proxy)
      - window label (binary stress with -1 ignore)
    """
    starts, y_major = window_majority_label(
        labels=labels,
        fs=fs,
        window_seconds=win_cfg.window_seconds,
        overlap=win_cfg.overlap,
    )
    y_bin = labels_to_binary_stress_windowed(y_major)

    win = int(round(win_cfg.window_seconds * fs))

    rows = []
    for s, y in zip(starts, y_bin):
        if y == -1:
            continue  # ignore undefined/unknown windows

        ecg_win = ecg[s : s + win]
        resp_win = resp[s : s + win]

        f_ecg = ecg_features_from_window(ecg_win, fs)
        f_rsp = resp_features_from_window(resp_win, fs)

        row = {
            "start_sample": int(s),
            "end_sample": int(s + win),
            "label_stress": int(y),
            "window_seconds": float(win_cfg.window_seconds),
            "fs_hz": float(fs),
            **f_ecg,
            **f_rsp,
        }
        rows.append(row)

    return pd.DataFrame(rows)
