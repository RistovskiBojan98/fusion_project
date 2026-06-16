from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SubjectZNorm(BaseEstimator, TransformerMixin):
    """
    Z-normalize numeric features per subject using TRAINING data only.

    Fit:
      compute mean/std per subject_id for each feature column
      compute global mean/std as fallback

    Transform:
      for each row, use its subject's mean/std
      if subject unseen, use global stats
    """

    def __init__(self, subject_col: str = "subject_id", feature_cols: Optional[list[str]] = None, eps: float = 1e-8):
        self.subject_col = subject_col
        self.feature_cols = feature_cols
        self.eps = eps

        self._subj_mean = None
        self._subj_std = None
        self._global_mean = None
        self._global_std = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.feature_cols is None:
            raise ValueError("SubjectZNorm requires feature_cols.")

        self._global_mean = X[self.feature_cols].mean()
        self._global_std = X[self.feature_cols].std(ddof=0).replace(0.0, np.nan)

        grp = X.groupby(self.subject_col, sort=False)
        self._subj_mean = grp[self.feature_cols].mean()
        self._subj_std = grp[self.feature_cols].std(ddof=0).replace(0.0, np.nan)
        return self

    def transform(self, X: pd.DataFrame):
        if self._subj_mean is None:
            raise RuntimeError("SubjectZNorm not fit.")

        subjects = X[self.subject_col].astype(str).to_numpy()
        vals = X[self.feature_cols].to_numpy(dtype=float)

        out = np.empty_like(vals, dtype=float)

        gmu = self._global_mean.to_numpy(dtype=float)
        gsd = self._global_std.to_numpy(dtype=float)

        for i, sid in enumerate(subjects):
            if sid in self._subj_mean.index:
                mu = self._subj_mean.loc[sid].to_numpy(dtype=float)
                sd = self._subj_std.loc[sid].to_numpy(dtype=float)
            else:
                mu, sd = gmu, gsd

            z = (vals[i] - mu) / (sd + self.eps)
            bad = ~np.isfinite(z)
            if bad.any():
                z2 = (vals[i] - gmu) / (gsd + self.eps)
                z[bad] = z2[bad]
            out[i] = z

        return out
