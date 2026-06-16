# Ambient Intelligence for Health Deterioration Detection

This repository contains a university research prototype for detecting possible
health deterioration by combining two signal sources:

- physiological signals from the WESAD wearable stress dataset
- environmental signals from the UCI Air Quality dataset

The project is not a medical device. It is an engineering prototype that shows
how wearable physiology and ambient air quality can be processed, fused, and
turned into interpretable warning states.

## Project Goal

The central question is:

> Can wearable physiology and air-quality context be combined into an
> interpretable early-warning state?

The motivation is that a person's risk state may depend on both body signals and
the surrounding environment. For example, moderate physiological stress may be
more concerning when outdoor air quality is poor than when the environment is
safe.

The system therefore builds a full pipeline:

1. Load physiological data from WESAD.
2. Extract ECG, heart-rate-variability, and respiration features.
3. Train stress-detection models with subject-separated validation.
4. Load and clean air-quality measurements.
5. Convert pollutants into a normalized air-risk score.
6. Fuse stress probability and air risk into warning states.
7. Smooth the final state stream to reduce noisy alert changes.

## Datasets

### WESAD

WESAD is used for physiological modeling. This project uses the chest ECG,
chest respiration, and WESAD labels.

Expected structure:

```text
data/wesad/
  S2/S2.pkl
  S3/S3.pkl
  ...
```

The local copy may be named `data/WESAD` on Windows. On a case-sensitive system,
rename it to `data/wesad` or adjust `WesadConfig.root_dir` in
`src/wesad_loader.py`.

Labels are converted into a binary stress task:

- label `2` -> stress
- labels `1`, `3`, and `4` -> non-stress
- undefined labels are ignored

### UCI Air Quality

The UCI Air Quality dataset is used as environmental context.

Expected file:

```text
data/air_quality/AirQualityUCI.csv
```

The loader handles the dataset's semicolon-separated format, decimal commas,
missing values marked as `-200`, datetime parsing, hourly resampling, and
interpolation.

The air-risk score uses these pollutant columns when available:

- `CO(GT)`
- `NO2(GT)`
- `NOx(GT)`

Temperature and relative humidity can add a small weather modifier.

## Repository Structure

```text
src/
  wesad_loader.py          Load WESAD subject files.
  physio_features.py       Build physiological window features.
  subject_norm.py          Subject-wise normalization.
  physio_infer.py          Stress-probability inference helper.
  air_quality_loader.py    Load and clean UCI Air Quality data.
  air_risk.py              Convert pollutants into a 0-1 risk score.
  fusion_policy.py         Rule-based physiology plus air-risk fusion.
  fusion_features.py       Feature builder for learned fusion.
  fusion_action_map.py     Map states to actions and rationales.
  smoothing.py             EMA smoothing and hysteresis.
  state_postprocess.py     Stabilize special warning states.

scripts/
  build_wesad_features.py          Build WESAD feature CSV.
  train_physio_model.py            Train baseline stress models.
  train_physio_model_strict.py     Train calibrated random-forest bundle.
  test_air_quality.py              Check air-quality loading.
  test_air_risk.py                 Build air-risk diagnostics.
  run_fusion_demo.py               Run rule-based fusion demo.
  train_fusion_model.py            Train explainable fusion tree.
  run_fusion_demo_learned.py       Run learned and smoothed fusion demo.

outputs/
  models/                          Trained model artifacts.
  figures/                         Diagnostic plots.
  wesad_window_features.csv
  physio_model_metrics.json
  physio_model_metrics_strict.json
  fusion_demo_results_strict.csv
  fusion_demo_results_learned_smoothed.csv
  fusion_tree_rules.txt
```

## Method

### Physiological Modeling

The WESAD recordings are split into 60-second windows with 50 percent overlap.
For each window, the project extracts features such as:

- mean heart rate
- heart-rate variability using RMSSD and SDNN
- ECG peak count
- respiration rate
- respiration variability

The resulting feature table contains 1,499 windows from 15 WESAD subjects:

- 1,167 non-stress windows
- 332 stress windows

Because physiological signals are person-specific, the models are evaluated with
`GroupKFold`, using `subject_id` as the group. This prevents windows from the
same person appearing in both training and test folds.

Two training stages are included:

- a baseline logistic regression and random forest
- a stricter calibrated random-forest bundle for downstream fusion

### Air-Risk Modeling

The air-quality module converts pollutants into normalized risk components:

- `risk_CO`
- `risk_NO2`
- `risk_NOx`
- `risk_weather`
- `air_risk_score`

The default pollutant weights are:

- CO: `0.2`
- NO2: `0.4`
- NOx: `0.4`

The score is clipped to the range `[0, 1]`, where higher values mean higher
environmental risk.

### Fusion

The fusion layer combines:

- calibrated physiological stress probability, `p_stress`
- environmental score, `air_risk`
- recent trend in stress probability

It outputs one of these interpretable states:

| State | Meaning |
| --- | --- |
| `normal` | Physiology and air context are not elevated. |
| `high_air_risk` | Air quality is risky even if stress is not high. |
| `high_physiological_risk` | Physiological stress is high. |
| `elevated_context_risk` | Moderate stress plus high air risk. |
| `rising_stress` | Stress probability is increasing. |
| `high_risk_combined` | Physiology and air quality are both high risk. |

Each state is mapped to a recommended action and a short rationale. A shallow
decision tree is also trained to learn the rule-based fusion policy while
remaining easy to inspect in `outputs/fusion_tree_rules.txt`.

The learned demo then applies exponential moving average smoothing, hysteresis,
and minimum dwell times so the final alert stream does not change too rapidly.

## Results

Baseline physiological stress detection:

| Model | AUC | F1 | Accuracy |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.855 | 0.581 | 0.777 |
| Random forest | 0.883 | 0.620 | 0.817 |

Strict calibrated physiology model:

| Metric | Value |
| --- | ---: |
| Calibrated out-of-fold AUC | 0.831 |
| Calibrated Brier score | 0.112 |
| Best F1 threshold | 0.24 |
| Cost-sensitive threshold | 0.31 |

The calibrated model is used for fusion because it produces probabilities that
are more suitable for thresholds and downstream decision rules.

Rule-based strict fusion output:

| Status | Count |
| --- | ---: |
| `normal` | 1,026 |
| `high_physiological_risk` | 217 |
| `high_air_risk` | 112 |
| `rising_stress` | 92 |
| `elevated_context_risk` | 27 |
| `high_risk_combined` | 25 |

Learned and smoothed fusion output:

| Status | Count |
| --- | ---: |
| `normal` | 1,175 |
| `high_physiological_risk` | 248 |
| `high_air_risk` | 43 |
| `high_risk_combined` | 21 |
| `rising_stress` | 12 |

These results show that the prototype can separate different kinds of risk
instead of producing only a single generic alarm.

## Main Takeaways

- Subject-aware validation is important for wearable physiology.
- Accuracy alone is not enough because stress windows are the minority class.
- Calibration improves the usefulness of stress probabilities for fusion.
- Air quality changes how a physiological stress signal should be interpreted.
- Explainable fusion rules make the final recommendations easier to understand.
- Smoothing and hysteresis are important for realistic alert behavior.

## Limitations

- WESAD and UCI Air Quality were not collected from the same people at the same
  time. The fusion demo uses synthetic alignment.
- The ECG and respiration feature extraction is intentionally lightweight.
- The air-risk thresholds are engineering defaults, not clinical or regulatory
  thresholds.
- WESAD labels represent controlled experimental stress, not all forms of health
  deterioration.
- The system has not been validated on real synchronized wearable and
  environmental recordings.
- The recommendations are illustrative and should not be treated as medical
  advice.

## How to Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full workflow:

```bash
py scripts/test_wesad_loader.py
py scripts/build_wesad_features.py
py scripts/train_physio_model.py
py scripts/train_physio_model_strict.py
py scripts/test_air_quality.py
py scripts/test_air_risk.py
py scripts/run_fusion_demo.py
py scripts/train_fusion_model.py
py scripts/run_fusion_demo_learned.py
```

Important generated files:

```text
outputs/wesad_window_features.csv
outputs/models/physio_rf_strict_calibrated_bundle.joblib
outputs/models/fusion_decision_tree.joblib
outputs/fusion_demo_results_strict.csv
outputs/fusion_demo_results_learned_smoothed.csv
outputs/figures/
```

## Summary

This project demonstrates a transparent ambient-intelligence pipeline. It starts
with physiological and environmental data, converts them into interpretable risk
signals, and fuses them into warning states with actions and rationales. The
result is a compact but complete prototype that can be understood, reproduced,
and discussed in an academic setting.
