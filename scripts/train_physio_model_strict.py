import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold  # noqa: E402

from src.subject_norm import SubjectZNorm  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    bundle_filename: str
    factory: Callable[[np.ndarray], Any]


METRIC_COLS = [
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
    "accuracy",
    "brier",
]


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


def safe_curve_metric(metric_name: str, y_true: np.ndarray, p: np.ndarray, scope: str) -> float:
    if len(np.unique(y_true)) < 2:
        warnings.warn(
            f"{metric_name} is undefined for {scope}: only one class is present. Returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")

    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, p))
    if metric_name == "pr_auc":
        return float(average_precision_score(y_true, p))
    raise ValueError(f"Unsupported curve metric: {metric_name}")


def confusion_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def balanced_accuracy_from_counts(counts: dict) -> float:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    recalls = []
    if tp + fn > 0:
        recalls.append(tp / (tp + fn))
    if tn + fp > 0:
        recalls.append(tn / (tn + fp))
    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def binary_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float, scope: str) -> dict:
    y_pred = (p >= threshold).astype(int)
    counts = confusion_values(y_true, y_pred)

    metrics = {
        "balanced_accuracy": balanced_accuracy_from_counts(counts),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "pr_auc": safe_curve_metric("pr_auc", y_true, p, scope),
        "roc_auc": safe_curve_metric("roc_auc", y_true, p, scope),
        "brier": float(brier_score_loss(y_true, p)) if len(y_true) else float("nan"),
    }
    metrics.update(counts)
    return metrics


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
        metrics = binary_metrics(y_true, p, float(t), f"threshold search ({objective})")
        cost = fp_cost * metrics["fp"] + fn_cost * metrics["fn"]
        score = metrics["f1"] if objective == "f1" else -cost

        cand = {
            "threshold": float(t),
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "confusion_matrix": [
                [metrics["tn"], metrics["fp"]],
                [metrics["fn"], metrics["tp"]],
            ],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "cost": float(cost),
            "objective": objective,
        }

        if best is None:
            best = cand
            continue

        best_score = best["f1"] if objective == "f1" else -best["cost"]
        if score > best_score:
            best = cand

    return best


def predict_positive_proba(clf: Any, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(X)
    classes = list(getattr(clf, "classes_", []))
    if 1 not in classes:
        raise RuntimeError(f"Classifier was not trained with positive class 1. Classes: {classes}")
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def class_counts(y: np.ndarray) -> dict:
    return {
        "non_stress": int(np.sum(y == 0)),
        "stress": int(np.sum(y == 1)),
    }


def fit_fold_components(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: np.ndarray,
    feature_cols: list[str],
    spec: ModelSpec,
) -> tuple[np.ndarray, Any, SubjectZNorm, Any]:
    imp = SimpleImputer(strategy="median")
    Xtr_num = imp.fit_transform(X_tr[feature_cols])
    Xte_num = imp.transform(X_te[feature_cols])

    Xtr_imp = pd.DataFrame(Xtr_num, columns=feature_cols, index=X_tr.index)
    Xtr_imp.insert(0, "subject_id", X_tr["subject_id"].astype(str).values)

    Xte_imp = pd.DataFrame(Xte_num, columns=feature_cols, index=X_te.index)
    Xte_imp.insert(0, "subject_id", X_te["subject_id"].astype(str).values)

    zn = SubjectZNorm(subject_col="subject_id", feature_cols=feature_cols)
    Ztr = zn.fit(Xtr_imp).transform(Xtr_imp)
    Zte = zn.transform(Xte_imp)

    clf = spec.factory(y_tr)
    clf.fit(Ztr, y_tr)
    p = predict_positive_proba(clf, Zte)
    return p, imp, zn, clf


def strict_oof_base_probs(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    spec: ModelSpec,
    n_splits: int = 5,
) -> dict:
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    p_oof = np.full(len(y), np.nan, dtype=float)
    fold_details = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        y_tr, y_te = y[tr], y[te]
        p = fit_fold_components(X_tr, X_te, y_tr, feature_cols, spec)[0]
        p_oof[te] = p

        train_counts = class_counts(y_tr)
        test_counts = class_counts(y_te)
        auc = safe_curve_metric("roc_auc", y_te, p, f"{spec.name} fold {fold}")
        pr_auc = safe_curve_metric("pr_auc", y_te, p, f"{spec.name} fold {fold}")
        brier = float(brier_score_loss(y_te, p))

        detail = {
            "fold": int(fold),
            "train_idx": tr,
            "test_idx": te,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "train_non_stress": train_counts["non_stress"],
            "train_stress": train_counts["stress"],
            "test_non_stress": test_counts["non_stress"],
            "test_stress": test_counts["stress"],
            "test_subjects": sorted(set(groups[te].astype(str))),
            "auc": auc,
            "pr_auc": pr_auc,
            "brier": brier,
        }
        fold_details.append(detail)

        print(
            f"[{spec.display_name}] Fold {fold} | n_test={len(te)} | "
            f"ROC-AUC={auc:.3f} | PR-AUC={pr_auc:.3f} | Brier={brier:.4f}"
        )

    if np.isnan(p_oof).any():
        raise RuntimeError(f"Some OOF probabilities are NaN for {spec.name}; check CV splitting.")

    return {"p_oof_base": p_oof, "fold_details": fold_details}


def train_final_components(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    spec: ModelSpec,
):
    imp = SimpleImputer(strategy="median")
    X_num = imp.fit_transform(X[feature_cols])
    X_imp = pd.DataFrame(X_num, columns=feature_cols, index=X.index)
    X_imp.insert(0, "subject_id", X["subject_id"].astype(str).values)

    zn = SubjectZNorm(subject_col="subject_id", feature_cols=feature_cols)
    Z = zn.fit(X_imp).transform(X_imp)

    clf = spec.factory(y)
    clf.fit(Z, y)
    return imp, zn, clf


def build_fold_rows(
    spec: ModelSpec,
    fold_details: list[dict],
    y: np.ndarray,
    p_cal: np.ndarray,
    threshold: float,
) -> list[dict]:
    rows = []
    for detail in fold_details:
        te = detail["test_idx"]
        scope = f"{spec.name} fold {detail['fold']} calibrated"
        metrics = binary_metrics(y[te], p_cal[te], threshold, scope)
        row = {
            "fold_index": detail["fold"],
            "test_subject_ids": ";".join(detail["test_subjects"]),
            "model_name": spec.name,
            "threshold_used": float(threshold),
            "n_train_samples": detail["n_train"],
            "n_test_samples": detail["n_test"],
            "train_non_stress": detail["train_non_stress"],
            "train_stress": detail["train_stress"],
            "test_non_stress": detail["test_non_stress"],
            "test_stress": detail["test_stress"],
            "train_class_distribution": (
                f"non_stress={detail['train_non_stress']};stress={detail['train_stress']}"
            ),
            "test_class_distribution": (
                f"non_stress={detail['test_non_stress']};stress={detail['test_stress']}"
            ),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def build_subject_rows(
    spec: ModelSpec,
    subject_ids: np.ndarray,
    y: np.ndarray,
    p_cal: np.ndarray,
    threshold: float,
) -> list[dict]:
    rows = []
    y_pred = (p_cal >= threshold).astype(int)
    for subject_id in sorted(set(subject_ids.astype(str))):
        mask = subject_ids.astype(str) == subject_id
        y_s = y[mask]
        p_s = p_cal[mask]
        pred_s = y_pred[mask]
        scope = f"{spec.name} subject {subject_id}"
        metrics = binary_metrics(y_s, p_s, threshold, scope)
        row = {
            "subject_id": subject_id,
            "model_name": spec.name,
            "threshold_used": float(threshold),
            "n_windows": int(mask.sum()),
            "n_stress": int(np.sum(y_s == 1)),
            "n_non_stress": int(np.sum(y_s == 0)),
            "predicted_stress_count": int(np.sum(pred_s == 1)),
            "mean_predicted_stress_probability": float(np.mean(p_s)),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def summarize_metric_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    summary = {}
    df = pd.DataFrame(rows)
    for col in METRIC_COLS:
        vals = pd.to_numeric(df[col], errors="coerce")
        summary[f"{col}_mean"] = float(vals.mean(skipna=True))
        summary[f"{col}_std"] = float(vals.std(skipna=True, ddof=1))
    return summary


def old_style_fold_summary(fold_details: list[dict]) -> list[dict]:
    return [
        {
            "fold": int(d["fold"]),
            "n_test": int(d["n_test"]),
            "auc": float(d["auc"]),
            "pr_auc": float(d["pr_auc"]),
            "brier": float(d["brier"]),
            "test_subjects": d["test_subjects"],
        }
        for d in fold_details
    ]


def evaluate_model(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    spec: ModelSpec,
) -> dict:
    print(f"\n=== Strict GroupKFold evaluation: {spec.display_name} ===")
    oof = strict_oof_base_probs(X, y, groups, feature_cols, spec, n_splits=5)
    p_base = oof["p_oof_base"]

    base_oof_auc = safe_curve_metric("roc_auc", y, p_base, f"{spec.name} base OOF")
    base_oof_pr_auc = safe_curve_metric("pr_auc", y, p_base, f"{spec.name} base OOF")
    base_oof_brier = float(brier_score_loss(y, p_base))

    calibrator_type = "isotonic"
    cal = fit_isotonic_calibrator(p_base, y)
    p_cal = cal.transform(p_base)

    cal_oof_auc = safe_curve_metric("roc_auc", y, p_cal, f"{spec.name} calibrated OOF")
    cal_oof_pr_auc = safe_curve_metric("pr_auc", y, p_cal, f"{spec.name} calibrated OOF")
    cal_oof_brier = float(brier_score_loss(y, p_cal))

    best_f1 = pick_threshold(y, p_cal, objective="f1")
    best_cost = pick_threshold(y, p_cal, objective="cost", fp_cost=1.0, fn_cost=2.0)
    threshold = float(best_f1["threshold"])

    fold_rows = build_fold_rows(spec, oof["fold_details"], y, p_cal, threshold)
    subject_rows = build_subject_rows(spec, groups, y, p_cal, threshold)
    overall_at_f1 = binary_metrics(y, p_cal, threshold, f"{spec.name} calibrated OOF overall")

    metrics = {
        "model_name": spec.name,
        "base_oof_auc": base_oof_auc,
        "base_oof_pr_auc": base_oof_pr_auc,
        "base_oof_brier": base_oof_brier,
        "cal_oof_auc": cal_oof_auc,
        "cal_oof_pr_auc": cal_oof_pr_auc,
        "cal_oof_brier": cal_oof_brier,
        "best_f1": best_f1,
        "best_cost": best_cost,
        "overall_at_best_f1_threshold": overall_at_f1,
        "fold_metric_summary": summarize_metric_rows(fold_rows),
        "subject_metric_summary": summarize_metric_rows(subject_rows),
        "folds": old_style_fold_summary(oof["fold_details"]),
    }

    print(
        f"[{spec.display_name}] Calibrated OOF: ROC-AUC={cal_oof_auc:.3f} | "
        f"PR-AUC={cal_oof_pr_auc:.3f} | Brier={cal_oof_brier:.4f}"
    )
    print(
        f"[{spec.display_name}] Best-F1 threshold={threshold:.2f} | "
        f"F1={best_f1['f1']:.3f} | Precision={best_f1['precision']:.3f} | "
        f"Recall={best_f1['recall']:.3f} | Balanced Acc={best_f1['balanced_accuracy']:.3f}"
    )

    return {
        "spec": spec,
        "p_base": p_base,
        "p_cal": p_cal,
        "calibrator_type": calibrator_type,
        "calibrator": cal,
        "metrics": metrics,
        "fold_rows": fold_rows,
        "subject_rows": subject_rows,
    }


def optional_model_specs() -> tuple[list[ModelSpec], list[dict]]:
    specs = []
    skipped = []

    def rf_factory(_y_train: np.ndarray) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    specs.append(
        ModelSpec(
            name="random_forest",
            display_name="RandomForest",
            bundle_filename="physio_rf_strict_calibrated_bundle.joblib",
            factory=rf_factory,
        )
    )

    try:
        from xgboost import XGBClassifier  # type: ignore

        def xgb_factory(y_train: np.ndarray) -> Any:
            pos = max(int(np.sum(y_train == 1)), 1)
            neg = max(int(np.sum(y_train == 0)), 1)
            return XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=neg / pos,
                random_state=42,
                n_jobs=-1,
            )

        specs.append(
            ModelSpec(
                name="xgboost",
                display_name="XGBoost",
                bundle_filename="physio_xgboost_strict_calibrated_bundle.joblib",
                factory=xgb_factory,
            )
        )
    except ImportError as exc:
        skipped.append({"model_name": "xgboost", "reason": f"xgboost is not installed ({exc})."})

    try:
        from lightgbm import LGBMClassifier  # type: ignore

        def lgbm_factory(_y_train: np.ndarray) -> Any:
            return LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        specs.append(
            ModelSpec(
                name="lightgbm",
                display_name="LightGBM",
                bundle_filename="physio_lightgbm_strict_calibrated_bundle.joblib",
                factory=lgbm_factory,
            )
        )
    except ImportError as exc:
        skipped.append({"model_name": "lightgbm", "reason": f"lightgbm is not installed ({exc})."})

    return specs, skipped


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return None if not np.isfinite(val) else val
    return value


def main() -> None:
    feat_path = PROJECT_ROOT / "outputs" / "wesad_window_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Missing features CSV: {feat_path}\n"
            "Run: py scripts/build_wesad_features.py"
        )

    df = pd.read_csv(feat_path)
    feature_cols = make_feature_list(df)

    X = df[["subject_id"] + feature_cols].copy()
    y = df["label_stress"].astype(int).to_numpy()
    groups = df["subject_id"].astype(str).to_numpy()

    print("Loaded:", df.shape)
    print("Features:", feature_cols)
    print("Label counts:\n", df["label_stress"].value_counts())

    model_specs, skipped_models = optional_model_specs()
    for skipped in skipped_models:
        print(f"Skipping {skipped['model_name']}: {skipped['reason']}")

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_models = out_dir / "models"
    out_models.mkdir(parents=True, exist_ok=True)

    all_fold_rows = []
    all_subject_rows = []
    model_metrics = {}
    primary_result = None

    for spec in model_specs:
        try:
            result = evaluate_model(X, y, groups, feature_cols, spec)
        except Exception as exc:
            if spec.name == "random_forest":
                raise
            skipped_models.append({"model_name": spec.name, "reason": f"training failed: {exc}"})
            print(f"Skipping {spec.display_name}: training failed with error: {exc}")
            continue

        all_fold_rows.extend(result["fold_rows"])
        all_subject_rows.extend(result["subject_rows"])
        model_metrics[spec.name] = result["metrics"]

        imp, zn, clf = train_final_components(X, y, feature_cols, spec)
        bundle = {
            "model_name": spec.name,
            "imputer": imp,
            "znorm": zn,
            "classifier": clf,
            "calibrator_type": result["calibrator_type"],
            "calibrator": result["calibrator"],
            "feature_cols": feature_cols,
            "threshold_f1": result["metrics"]["best_f1"]["threshold"],
            "threshold_cost": result["metrics"]["best_cost"]["threshold"],
            "metrics": result["metrics"],
            "notes": {
                "subject_normalization": "per-subject z-score using training-subject stats",
                "calibration_fit": "pooled out-of-fold probabilities from GroupKFold",
                "evaluation_frame": "strict leave-subjects-out GroupKFold by subject_id",
            },
        }

        out_path = out_models / spec.bundle_filename
        joblib.dump(bundle, out_path)
        print("Saved calibrated bundle to:", out_path)

        if spec.name == "random_forest":
            primary_result = result

    if primary_result is None:
        raise RuntimeError("RandomForest evaluation did not complete; no compatible primary bundle was produced.")

    fold_metrics_path = out_dir / "physio_groupkfold_fold_metrics.csv"
    pd.DataFrame(all_fold_rows).to_csv(fold_metrics_path, index=False)
    print("Saved fold-wise GroupKFold metrics to:", fold_metrics_path)

    subject_metrics_path = out_dir / "physio_subject_level_metrics.csv"
    pd.DataFrame(all_subject_rows).to_csv(subject_metrics_path, index=False)
    print("Saved subject-level OOF metrics to:", subject_metrics_path)

    primary_metrics = primary_result["metrics"].copy()
    strict_metrics = {
        **primary_metrics,
        "models": model_metrics,
        "skipped_models": skipped_models,
        "new_outputs": {
            "fold_metrics_csv": str(fold_metrics_path.relative_to(PROJECT_ROOT)),
            "subject_metrics_csv": str(subject_metrics_path.relative_to(PROJECT_ROOT)),
        },
    }

    out_metrics = out_dir / "physio_model_metrics_strict.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(json_safe(strict_metrics), f, indent=2, allow_nan=False)
    print("Saved strict metrics to:", out_metrics)


if __name__ == "__main__":
    main()
