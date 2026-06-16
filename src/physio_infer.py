from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def predict_p_stress_calibrated(bundle: Dict[str, Any], df_feat: pd.DataFrame) -> np.ndarray:
    """
    Compute calibrated stress probabilities using the saved strict bundle.

    bundle contains:
      - imputer, znorm, classifier
      - calibrator (isotonic or platt)
      - feature_cols
    """
    feature_cols = bundle["feature_cols"]
    if "subject_id" not in df_feat.columns:
        raise ValueError("df_feat must contain 'subject_id' column.")

    # 1) Impute numeric features
    imp = bundle["imputer"]
    X_num = imp.transform(df_feat[feature_cols])
    X_imp = pd.DataFrame(X_num, columns=feature_cols, index=df_feat.index)
    X_imp.insert(0, "subject_id", df_feat["subject_id"].astype(str).values)

    # 2) Subject z-norm
    zn = bundle["znorm"]
    Z = zn.transform(X_imp)

    # 3) Base classifier probability
    clf = bundle["classifier"]
    p_base = clf.predict_proba(Z)[:, 1]

    # 4) Calibration mapping
    cal_type = bundle.get("calibrator_type", "isotonic")
    cal = bundle["calibrator"]

    if cal_type == "isotonic":
        p_cal = cal.transform(p_base)
    elif cal_type == "platt":
        p = np.clip(p_base, 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        p_cal = cal.predict_proba(logit)[:, 1]
    else:
        raise ValueError(f"Unknown calibrator_type: {cal_type}")

    return np.asarray(p_cal, dtype=float)
