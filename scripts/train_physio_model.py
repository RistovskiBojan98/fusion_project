import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402


def make_feature_list(df: pd.DataFrame) -> list[str]:
    """
    Select the numeric feature columns for modeling.
    Exclude IDs, labels, and window metadata.
    """
    drop_cols = {
        "subject_id",
        "label_stress",
        "start_sample",
        "end_sample",
        "window_seconds",
        "fs_hz",
    }
    numeric_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


def eval_model_cv(model_name: str, pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> dict:
    """
    GroupKFold CV evaluation (leave-subjects-out style).
    Returns averaged metrics and stores per-fold metrics too.
    """
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    fold_metrics = []

    y_true_all = []
    y_prob_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)

        # Some models/pipelines might not support predict_proba; ours do.
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fold_res = {
            "fold": fold,
            "n_test": int(len(test_idx)),
            "auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else float("nan"),
            "f1": float(f1_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        fold_metrics.append(fold_res)

        y_true_all.append(y_test)
        y_prob_all.append(y_prob)
        y_pred_all.append(y_pred)

        print(f"[{model_name}] Fold {fold} | n={fold_res['n_test']} | AUC={fold_res['auc']:.3f} | F1={fold_res['f1']:.3f} | Acc={fold_res['accuracy']:.3f}")

    y_true_all = np.concatenate(y_true_all)
    y_prob_all = np.concatenate(y_prob_all)
    y_pred_all = np.concatenate(y_pred_all)

    overall = {
        "model": model_name,
        "overall_auc": float(roc_auc_score(y_true_all, y_prob_all)) if len(np.unique(y_true_all)) > 1 else float("nan"),
        "overall_f1": float(f1_score(y_true_all, y_pred_all)),
        "overall_accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "overall_confusion_matrix": confusion_matrix(y_true_all, y_pred_all).tolist(),
        "folds": fold_metrics,
    }
    return overall


def main() -> None:
    in_path = PROJECT_ROOT / "outputs" / "wesad_window_features.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing features file: {in_path}\n"
            "Run: py scripts/build_wesad_features.py first."
        )

    df = pd.read_csv(in_path)
    if "subject_id" not in df.columns or "label_stress" not in df.columns:
        raise ValueError("Expected columns subject_id and label_stress in the features CSV.")

    # Labels + groups
    y = df["label_stress"].astype(int).to_numpy()
    groups = df["subject_id"].astype(str).to_numpy()

    # Features
    feature_cols = make_feature_list(df)
    X = df[feature_cols].copy()

    print("Loaded features:", df.shape)
    print("Using feature columns:", feature_cols)
    print("\nLabel distribution:")
    print(df["label_stress"].value_counts())

    # Common preprocessing: impute + scale (scale helps logistic regression)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)],
        remainder="drop",
    )

    # Model 1: Logistic Regression (baseline, interpretable)
    logreg = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    pipe_logreg = Pipeline(steps=[("pre", preprocessor), ("clf", logreg)])

    # Model 2: Random Forest (often strong on engineered features)
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    # RF doesn't need scaling, but we'll keep the same preprocessor (imputer+scaler) for simplicity.
    # Alternatively you can remove StandardScaler for RF later.
    pipe_rf = Pipeline(steps=[("pre", preprocessor), ("clf", rf)])

    print("\n=== Cross-validated evaluation (GroupKFold by subject) ===")
    res_logreg = eval_model_cv("logreg", pipe_logreg, X, y, groups)
    res_rf = eval_model_cv("random_forest", pipe_rf, X, y, groups)

    print("\n=== Overall Results ===")
    for res in [res_logreg, res_rf]:
        print(
            f"{res['model']}: AUC={res['overall_auc']:.3f}, "
            f"F1={res['overall_f1']:.3f}, Acc={res['overall_accuracy']:.3f}, "
            f"CM={res['overall_confusion_matrix']}"
        )

    # Train a final model on all data (choose the better one; default RF)
    best_pipe = pipe_rf
    best_name = "random_forest"

    out_models = PROJECT_ROOT / "outputs" / "models"
    out_models.mkdir(parents=True, exist_ok=True)

    model_path = out_models / f"physio_{best_name}.joblib"
    best_pipe.fit(X, y)
    joblib.dump(
        {
            "pipeline": best_pipe,
            "feature_cols": feature_cols,
        },
        model_path,
    )
    print("\n✅ Saved trained model to:", model_path)

    # Save metrics
    out_metrics = PROJECT_ROOT / "outputs" / "physio_model_metrics.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump({"logreg": res_logreg, "random_forest": res_rf}, f, indent=2)
    print("✅ Saved metrics to:", out_metrics)

    # Save feature importances for RF (interpretability)
    # Important: feature importances correspond to feature_cols (after preprocessor).
    # Because we used ColumnTransformer+Pipeline, we can access the RF via named step.
    try:
        rf_model = best_pipe.named_steps["clf"]
        importances = getattr(rf_model, "feature_importances_", None)
        if importances is not None:
            imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
            imp_path = PROJECT_ROOT / "outputs" / "rf_feature_importances.csv"
            imp_df.to_csv(imp_path, index=False)
            print("✅ Saved RF feature importances to:", imp_path)
            print("\nTop 10 features:")
            print(imp_df.head(10).to_string(index=False))
    except Exception as e:
        print("Could not save feature importances:", e)


if __name__ == "__main__":
    main()
