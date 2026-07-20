# Ambient Intelligence for Health Deterioration Detection

This repository contains a university research prototype for context-aware stress
and deterioration warning. It combines:

- WESAD physiological data: chest ECG, respiration, and stress labels
- UCI Air Quality data: outdoor pollutant and weather measurements

The goal is not to build a medical device. The goal is to show a transparent
ambient-intelligence pipeline where body signals and environmental context are
processed separately, then fused into explainable warning states.

## Pipeline

```text
WESAD ECG/RESP
  -> 60-second windows
  -> HR, HRV, respiration features
  -> subject-grouped stress models
  -> calibrated p_stress

UCI Air Quality
  -> cleaned hourly pollutant data
  -> configurable CO/NO2/NOx risk weights
  -> air_risk score

p_stress + air_risk + trend
  -> baseline comparisons and rule-based fusion
  -> status, action, rationale
```

The WESAD and UCI Air Quality datasets were not collected from the same people
or at the same time. Therefore, the fusion step is a scenario-based contextual
evaluation, not causal health validation. Air quality is used to test how an
ambient context signal could modify decisions from a wearable stress model.

## Main Components

### 1. Physiological Stress Model

`scripts/build_wesad_features.py` loads WESAD subject files and extracts window
features from ECG and respiration. The resulting feature table is saved as:

```text
outputs/wesad_window_features.csv
```

`scripts/train_physio_model_strict.py` trains subject-aware models with
`GroupKFold`, so windows from the same subject do not appear in both train and
test folds. It keeps RandomForest as the main compatible model and also tries:

- XGBoost, if `xgboost` is installed
- LightGBM, if `lightgbm` is installed

If those optional libraries are missing, the script prints a clear skip message
and continues with RandomForest.

The strict workflow includes imputation, subject-wise normalization, out-of-fold
prediction, isotonic calibration, threshold tuning, fold-wise reporting, and
subject-level reporting.

### 2. Environmental Air-Risk Model

`src/air_quality_loader.py` loads the UCI Air Quality CSV, handles decimal
commas, missing values, datetime parsing, resampling, and interpolation.

`src/air_risk.py` converts pollutant measurements into a score in `[0, 1]`.
Default pollutant weights are:

- CO: `0.20`
- NO2: `0.40`
- NOx: `0.40`

NO2 and NOx are weighted more heavily by default because they are useful
traffic-related outdoor pollution signals for this scenario. The weights are
configurable, and `scripts/run_air_weight_sensitivity.py` compares several
weight settings against the default.

### 3. Fusion Decision Layer

`scripts/run_fusion_demo.py` combines calibrated stress probability,
scenario-aligned air risk, and stress trend. The rule-based policy in
`src/fusion_policy.py` outputs:

| State | Meaning |
| --- | --- |
| `normal` | Physiology and air context are not elevated. |
| `high_air_risk` | Air quality is risky even if stress is not high. |
| `high_physiological_risk` | Physiological stress is high. |
| `elevated_context_risk` | Moderate stress plus high air risk. |
| `rising_stress` | Stress probability is increasing. |
| `high_risk_combined` | Physiology and air quality are both high risk. |

Each state receives an action and rationale. `scripts/run_fusion_comparison.py`
adds non-causal scenario metrics comparing:

- physiology-only decisions
- air-risk-only decisions
- a simple weighted-score baseline
- the current rule-based fusion policy

## Key Outputs

Physiology:

```text
outputs/physio_model_metrics_strict.json
outputs/physio_groupkfold_fold_metrics.csv
outputs/physio_subject_level_metrics.csv
outputs/models/physio_rf_strict_calibrated_bundle.joblib
outputs/models/physio_xgboost_strict_calibrated_bundle.joblib
outputs/models/physio_lightgbm_strict_calibrated_bundle.joblib
```

Fusion and air-risk:

```text
outputs/fusion_demo_results_strict.csv
outputs/fusion_comparison_metrics.csv
outputs/air_weight_sensitivity.csv
outputs/fusion_demo_results_learned_smoothed.csv
outputs/models/fusion_decision_tree.joblib
outputs/figures/
```

The new physiology CSVs report balanced accuracy, precision, recall, F1, PR-AUC,
ROC-AUC when available, threshold used, confusion matrix values, sample counts,
class distributions, fold test subjects, and subject-level out-of-fold behavior.
PR-AUC and ROC-AUC are written as missing values when a fold or subject contains
only one class.

The fusion comparison file reports scenario-level behavior, including high-risk
alert counts, decision stability, action differences compared with
physiology-only decisions, average severity, status/action distributions, and
whether decisions include rationales.

## Results Snapshot

Earlier RandomForest baseline physiology results:

| Model | AUC | F1 | Accuracy |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.855 | 0.581 | 0.777 |
| Random forest | 0.883 | 0.620 | 0.817 |

Strict calibrated physiology model comparison:

| Model | ROC-AUC | PR-AUC | Brier | Threshold | F1 | Precision | Recall | Balanced Acc. | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.837 | 0.687 | 0.107 | 0.24 | 0.621 | 0.697 | 0.560 | 0.745 | 0.849 |
| XGBoost | 0.843 | 0.684 | 0.105 | 0.23 | 0.598 | 0.526 | 0.693 | 0.758 | 0.794 |
| LightGBM | 0.833 | 0.693 | 0.107 | 0.30 | 0.605 | 0.744 | 0.509 | 0.730 | 0.853 |

Confusion values at each model's best-F1 threshold:

| Model | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: |
| RandomForest | 186 | 81 | 1086 | 146 |
| XGBoost | 230 | 207 | 960 | 102 |
| LightGBM | 169 | 58 | 1109 | 163 |

XGBoost has the highest ROC-AUC and recall, so it catches more stress windows
but creates more false positives. LightGBM is more conservative, with the best
precision and accuracy but lower recall.

Rule-based strict fusion output:

| Status | Count |
| --- | ---: |
| `normal` | 983 |
| `high_physiological_risk` | 288 |
| `high_air_risk` | 129 |
| `rising_stress` | 64 |
| `high_risk_combined` | 33 |
| `elevated_context_risk` | 2 |

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Feature extraction:

```bash
py scripts/build_wesad_features.py
```

Strict physiology training with fold-wise and subject-level metrics:

```bash
py scripts/train_physio_model_strict.py
```

Rule-based fusion demo:

```bash
py scripts/run_fusion_demo.py
```

Fusion comparison:

```bash
py scripts/run_fusion_comparison.py
```

Air-weight sensitivity analysis:

```bash
py scripts/run_air_weight_sensitivity.py
```

Optional learned fusion workflow:

```bash
py scripts/train_fusion_model.py
py scripts/run_fusion_demo_learned.py
```

## Limitations

- WESAD and UCI Air Quality are not synchronized; fusion is scenario-based.
- WESAD stress labels come from controlled experimental conditions.
- ECG and respiration feature extraction is intentionally lightweight.
- Air-risk weights and thresholds are engineering assumptions, not clinical or
  regulatory standards.
- The system has not been validated on real synchronized wearable and
  environmental recordings.
- Recommendations are illustrative and should not be treated as medical advice.

## Summary

The project demonstrates how a wearable stress detector, an environmental
air-risk model, and an explainable fusion policy can be combined into a compact
ambient-intelligence prototype. The added evaluations make the model behavior
clearer by reporting fold-level, subject-level, baseline-comparison, and
air-weight sensitivity results.
