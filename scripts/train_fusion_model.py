import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json  # noqa: E402
import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

from src.fusion_features import FusionFeatureConfig, build_fusion_features  # noqa: E402


def main() -> None:
    in_path = PROJECT_ROOT / "outputs" / "fusion_demo_results_strict.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing {in_path}. Run the strict fusion demo first (run_fusion_demo.py)."
        )

    df = pd.read_csv(in_path)

    if "status" not in df.columns:
        raise ValueError("Expected a 'status' column in fusion_demo_results_strict.csv")

    # Features for learned fusion
    X = build_fusion_features(df, FusionFeatureConfig(trend_k=3, include_interactions=True))
    y = df["status"].astype(str)

    # Group by subject (prevents leaking subject-specific physiology patterns)
    groups = df["subject_id"].astype(str) if "subject_id" in df.columns else None

    # Explainable model
    clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )


    # Evaluate with GroupKFold if we have subjects
    if groups is not None:
        gkf = GroupKFold(n_splits=min(5, groups.nunique()))
        y_true_all, y_pred_all = [], []

        for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
            clf.fit(X.iloc[tr], y.iloc[tr])
            pred = clf.predict(X.iloc[te])
            acc = accuracy_score(y.iloc[te], pred)
            print(f"Fold {fold} accuracy: {acc:.3f}")
            y_true_all.extend(y.iloc[te].tolist())
            y_pred_all.extend(pred.tolist())

        print("\nOverall accuracy:", accuracy_score(y_true_all, y_pred_all))
        print("\nConfusion matrix:\n", confusion_matrix(y_true_all, y_pred_all))
        print("\nClassification report:\n", classification_report(y_true_all, y_pred_all))
    else:
        clf.fit(X, y)

    # Fit final on all data
    clf.fit(X, y)

    # Export readable rules
    rules_txt = export_text(clf, feature_names=list(X.columns))

    out_dir = PROJECT_ROOT / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": clf,
        "feature_columns": list(X.columns),
        "feature_cfg": {"trend_k": 3, "include_interactions": True},
        "rules_text": rules_txt,
        "classes": sorted(y.unique().tolist()),
    }

    out_path = out_dir / "fusion_decision_tree.joblib"
    joblib.dump(bundle, out_path)
    print("\nSaved learned fusion model:", out_path)

    rules_path = PROJECT_ROOT / "outputs" / "fusion_tree_rules.txt"
    rules_path.write_text(rules_txt, encoding="utf-8")
    print("Saved tree rules:", rules_path)

    meta_path = PROJECT_ROOT / "outputs" / "fusion_tree_meta.json"
    meta_path.write_text(json.dumps({"classes": bundle["classes"]}, indent=2), encoding="utf-8")
    print("Saved meta:", meta_path)


if __name__ == "__main__":
    main()
