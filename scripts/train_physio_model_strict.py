import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.subject_norm import SubjectZNorm


def make_feature_list(df: pd.DataFrame) -> list[str]:
    drop_cols = {
        "subject_id",
        "label_stress",
        "start_sample",
        "end_sample",
        "window_seconds",
        "fs_hz",
    }
    return [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]


def fit_isotonic_calibrator(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    return iso


def fit_platt_calibrator(p: np.ndarray, y: np.ndarray) -> LogisticRegression:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(logit, y)
    return lr


def apply_platt(cal: LogisticRegression, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return cal.predict_proba(logit)[:, 1]


def pick_threshold(
    y_true: np.ndarray,
    p: np.ndarray,
    objective: str = "f1",
    fp_cost: float = 1.0,
    fn_cost: float = 2.0,
) -> dict:
    thresholds = np.linspace(0.05, 0.95, 91)
    best = None

    for t in thresholds:
        y_pred = (p >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        cost = fp_cost * fp + fn_cost * fn
        score = f1 if objective == "f1" else -cost

        cand = {
            "threshold": float(t),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "tn": int(tn),
            "cost": float(cost),
            "objective": objective,
        }

        if best is None:
            best = cand
        else:
            best_score = best["f1"] if objective == "f1" else -best["cost"]
            if score > best_score:
                best = cand

    return best


def strict_oof_probs(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    base_model: RandomForestClassifier,
    n_splits: int = 5,
) -> dict:
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    p_oof = np.full(len(y), np.nan, dtype=float)
    folds = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        y_tr, y_te = y[tr], y[te]
        g_te = groups[te]

        # 1) Impute numeric features only (fit on train only)
        imp = SimpleImputer(strategy="median")
        Xtr_num = imp.fit_transform(X_tr[feature_cols])
        Xte_num = imp.transform(X_te[feature_cols])

        Xtr_imp = pd.DataFrame(Xtr_num, columns=feature_cols, index=X_tr.index)
        Xtr_imp.insert(0, "subject_id", X_tr["subject_id"].astype(str).values)

        Xte_imp = pd.DataFrame(Xte_num, columns=feature_cols, index=X_te.index)
        Xte_imp.insert(0, "subject_id", X_te["subject_id"].astype(str).values)

        # 2) Subject-wise z-norm (fit on train only)
        zn = SubjectZNorm(subject_col="subject_id", feature_cols=feature_cols)
        Ztr = zn.fit(Xtr_imp).transform(Xtr_imp)
        Zte = zn.transform(Xte_imp)

        # 3) Fit model + predict
        clf = RandomForestClassifier(**base_model.get_params())
        clf.fit(Ztr, y_tr)
        p = clf.predict_proba(Zte)[:, 1]
        p_oof[te] = p

        auc = roc_auc_score(y_te, p) if len(np.unique(y_te)) > 1 else float("nan")
        brier = brier_score_loss(y_te, p)
        folds.append(
            {
                "fold": fold,
                "n_test": int(len(te)),
                "auc": float(auc),
                "brier": float(brier),
                "test_subjects": sorted(list(set(g_te.astype(str)))),
            }
        )
        print(f"[OOF base] Fold {fold} | n={len(te)} | AUC={auc:.3f} | Brier={brier:.4f}")

    if np.isnan(p_oof).any():
        raise RuntimeError("Some OOF probabilities are NaN; check CV splitting.")

    overall_auc = roc_auc_score(y, p_oof) if len(np.unique(y)) > 1 else float("nan")
    overall_brier = brier_score_loss(y, p_oof)

    return {"p_oof": p_oof, "overall_auc": float(overall_auc), "overall_brier": float(overall_brier), "folds": folds}


def train_final_components(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    base_model: RandomForestClassifier,
):
    # Fit imputer on all numeric features
    imp = SimpleImputer(strategy="median")
    X_num = imp.fit_transform(X[feature_cols])
    X_imp = pd.DataFrame(X_num, columns=feature_cols, index=X.index)
    X_imp.insert(0, "subject_id", X["subject_id"].astype(str).values)

    # Fit subject z-norm on all data
    zn = SubjectZNorm(subject_col="subject_id", feature_cols=feature_cols)
    Z = zn.fit(X_imp).transform(X_imp)

    # Fit classifier
    clf = RandomForestClassifier(**base_model.get_params())
    clf.fit(Z, y)

    return imp, zn, clf


def main() -> None:
    feat_path = PROJECT_ROOT / "outputs" / "wesad_window_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("Missing features CSV. Run: py scripts/build_wesad_features.py")

    df = pd.read_csv(feat_path)
    feature_cols = make_feature_list(df)

    X = df[["subject_id"] + feature_cols].copy()
    y = df["label_stress"].astype(int).to_numpy()
    groups = df["subject_id"].astype(str).to_numpy()

    print("Loaded:", df.shape)
    print("Features:", feature_cols)
    print("Label counts:\n", df["label_stress"].value_counts())

    base_rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\n=== Step A: Strict OOF probabilities (GroupKFold) ===")
    oof = strict_oof_probs(X, y, groups, feature_cols, base_rf, n_splits=5)
    print(f"\nBase OOF: AUC={oof['overall_auc']:.3f} | Brier={oof['overall_brier']:.4f}")

    p_oof = oof["p_oof"]

    print("\n=== Step B: Fit calibrator on OOF predictions ===")
    calibrator_type = "isotonic"  # or "platt"
    if calibrator_type == "isotonic":
        cal = fit_isotonic_calibrator(p_oof, y)
        p_cal = cal.transform(p_oof)
    else:
        cal = fit_platt_calibrator(p_oof, y)
        p_cal = apply_platt(cal, p_oof)

    cal_brier = brier_score_loss(y, p_cal)
    cal_auc = roc_auc_score(y, p_cal) if len(np.unique(y)) > 1 else float("nan")
    print(f"Calibrated OOF: AUC={cal_auc:.3f} | Brier={cal_brier:.4f} (lower is better)")

    print("\n=== Step C: Threshold tuning on calibrated OOF probabilities ===")
    best_f1 = pick_threshold(y, p_cal, objective="f1")
    print(
        f"Best-F1 threshold={best_f1['threshold']:.2f} | "
        f"F1={best_f1['f1']:.3f} | Prec={best_f1['precision']:.3f} | "
        f"Rec={best_f1['recall']:.3f} | Acc={best_f1['accuracy']:.3f} | "
        f"CM={best_f1['confusion_matrix']}"
    )

    best_cost = pick_threshold(y, p_cal, objective="cost", fp_cost=1.0, fn_cost=2.0)
    print(
        f"Best-COST threshold={best_cost['threshold']:.2f} | "
        f"cost={best_cost['cost']:.1f} (fn cost=2x) | "
        f"F1={best_cost['f1']:.3f} | Prec={best_cost['precision']:.3f} | "
        f"Rec={best_cost['recall']:.3f} | CM={best_cost['confusion_matrix']}"
    )

    print("\n=== Step D: Train final components on ALL data ===")
    imp, zn, clf = train_final_components(X, y, feature_cols, base_rf)

    out_models = PROJECT_ROOT / "outputs" / "models"
    out_models.mkdir(parents=True, exist_ok=True)

    bundle = {
        "imputer": imp,
        "znorm": zn,
        "classifier": clf,
        "calibrator_type": calibrator_type,
        "calibrator": cal,
        "feature_cols": feature_cols,
        "threshold_f1": best_f1["threshold"],
        "threshold_cost": best_cost["threshold"],
        "metrics": {
            "base_oof_auc": oof["overall_auc"],
            "base_oof_brier": oof["overall_brier"],
            "cal_oof_auc": float(cal_auc),
            "cal_oof_brier": float(cal_brier),
            "best_f1": best_f1,
            "best_cost": best_cost,
            "folds": oof["folds"],
        },
        "notes": {
            "subject_normalization": "per-subject z-score using training-subject stats",
            "calibration_fit": "pooled out-of-fold probabilities (GroupKFold)",
        },
    }

    out_path = out_models / "physio_rf_strict_calibrated_bundle.joblib"
    joblib.dump(bundle, out_path)
    print("\n✅ Saved bundle to:", out_path)

    out_metrics = PROJECT_ROOT / "outputs" / "physio_model_metrics_strict.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(bundle["metrics"], f, indent=2)
    print("✅ Saved strict metrics to:", out_metrics)


if __name__ == "__main__":
    main()
