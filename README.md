# Ambient Intelligence for Health Deterioration Detection

This project is a prototype ambient-intelligence pipeline for detecting possible
health deterioration by combining two different kinds of signals:

- physiological state, estimated from wearable-like chest ECG and respiration
  signals in the WESAD dataset
- environmental context, estimated from outdoor pollutant measurements in the
  UCI Air Quality dataset

The main idea is that a person's risk state is not only a property of their body
and not only a property of their surroundings. A moderate physiological stress
signal can mean something different when the surrounding air is clean than when
outdoor pollution is high. The project therefore builds a small end-to-end
system that extracts physiological features, trains a stress detector, computes
an air-risk score, fuses both streams, and produces explainable recommendations.

This is not a medical device and the thresholds are not clinical thresholds. It
is a research and engineering prototype showing how multimodal sensing, model
calibration, explainable decision rules, and temporal smoothing can be combined
into a practical ambient health-monitoring workflow.

## What the Project Does

At a high level, the project answers this question:

> Can we combine wearable physiology and ambient air quality to produce an
> interpretable warning state before or during health deterioration?

To explore that question, the repository implements the following pipeline:

1. Load WESAD chest ECG, chest respiration, and labels for each subject.
2. Split the physiological recordings into overlapping 60-second windows.
3. Extract heart-rate, heart-rate-variability, and respiration features.
4. Convert WESAD labels into a binary stress target.
5. Train subject-aware stress classifiers using grouped cross-validation.
6. Train a stricter calibrated random-forest bundle for inference.
7. Load and clean UCI Air Quality data.
8. Convert pollutants into an interpretable air-risk score from 0 to 1.
9. Align physiology windows with air-risk values for a synthetic fusion demo.
10. Combine stress probability and air risk into status labels and actions.
11. Train an explainable decision tree to learn the fusion policy.
12. Smooth the final status stream with EMA and hysteresis to reduce flicker.

The final output is not just a prediction. It includes a status, an action, and a
rationale, for example:

- `normal`
- `high_air_risk`
- `high_physiological_risk`
- `elevated_context_risk`
- `rising_stress`
- `high_risk_combined`

## Datasets

### WESAD

WESAD, the Wearable Stress and Affect Detection dataset, is used for
physiological modeling. This project uses the chest signals:

- ECG
- respiration
- WESAD labels

The code assumes the usual WESAD subject structure:

```text
data/wesad/
  S2/S2.pkl
  S3/S3.pkl
  ...
```

The local project copy is under `data/WESAD`. That works on Windows because path
case is not significant there. On a case-sensitive system, either rename the
folder to `data/wesad` or adjust `WesadConfig.root_dir` in
`src/wesad_loader.py`.

The labels are converted into a binary stress task:

- WESAD label `2` becomes `1`, meaning stress
- WESAD labels `1`, `3`, and `4` become `0`, meaning non-stress
- undefined or unused labels are ignored

### UCI Air Quality

The UCI Air Quality dataset is used for environmental context. The loader handles
the dataset's semicolon-separated CSV format, decimal commas, the `-200` missing
value sentinel, datetime parsing, hourly resampling, and time interpolation.

Expected location:

```text
data/air_quality/AirQualityUCI.csv
```

The air-risk module uses the interpretable pollutant columns when available:

- `CO(GT)`
- `NO2(GT)`
- `NOx(GT)`

It can also apply a small heat/humidity modifier using temperature and relative
humidity.

## Repository Structure

```text
src/
  wesad_loader.py          Load WESAD subject files and extract chest signals.
  physio_features.py       Window ECG/respiration and compute physiological features.
  subject_norm.py          Subject-wise z-normalization transformer.
  physio_infer.py          Inference helper for calibrated stress probabilities.
  air_quality_loader.py    Load and clean UCI Air Quality data.
  air_risk.py              Convert pollutants into a 0-1 air-risk score.
  fusion_policy.py         Rule-based physiology plus air-risk decision logic.
  fusion_features.py       Build features for learned fusion.
  fusion_action_map.py     Map final states to actions and rationales.
  smoothing.py             EMA smoothing and status hysteresis.
  state_postprocess.py     Enforce special safety/early-warning states.

scripts/
  test_wesad_loader.py             Sanity-check WESAD loading.
  build_wesad_features.py          Build window-level WESAD feature CSV.
  train_physio_model.py            Train baseline logistic regression and random forest.
  train_physio_model_strict.py     Train calibrated subject-aware RF bundle.
  test_air_quality.py              Sanity-check air-quality loading.
  test_air_risk.py                 Compute air risk and save diagnostic figures.
  run_fusion_demo.py               Rule-based strict fusion demo.
  train_fusion_model.py            Train explainable decision-tree fusion model.
  run_fusion_demo_learned.py       Learned and smoothed fusion demo.

outputs/
  wesad_window_features.csv
  physio_model_metrics.json
  physio_model_metrics_strict.json
  rf_feature_importances.csv
  air_quality_with_risk_sample.csv
  fusion_demo_results_strict.csv
  fusion_demo_results_learned_smoothed.csv
  fusion_tree_rules.txt
  fusion_tree_meta.json
  models/
  figures/
```

## Physiological Feature Pipeline

The WESAD feature builder uses 60-second windows with 50 percent overlap. For
each window, the majority WESAD label is used, then converted into the binary
stress target.

Each window receives these engineered features:

- `hr_mean_bpm`: mean heart rate estimated from ECG R-R intervals
- `hr_std_bpm`: heart-rate variability expressed as heart-rate spread
- `hrv_rmssd_ms`: RMSSD, a short-term heart-rate-variability feature
- `hrv_sdnn_ms`: SDNN, another HRV summary
- `rpeaks_count`: count of detected ECG peaks
- `resp_rate_bpm`: estimated breathing rate
- `resp_std`: respiration signal variability
- `resp_peaks_count`: count of respiration peaks

The ECG peak detection is intentionally lightweight. It uses a derivative-energy
heuristic with an adaptive threshold and a refractory period. This keeps the
project understandable and fast, but it also means the results should be treated
as prototype results rather than a benchmark against specialized ECG toolkits.

Generated feature table:

- file: `outputs/wesad_window_features.csv`
- rows: 1,499 windows
- subjects: 15 WESAD subjects
- class distribution:
  - non-stress: 1,167 windows
  - stress: 332 windows

The class imbalance matters. Stress windows are the minority class, so accuracy
alone is not enough. The project reports AUC, F1, recall/precision tradeoffs,
confusion matrices, and calibrated threshold behavior.

## Physiological Stress Models

Two physiological-modeling stages are included.

### Baseline Training

`scripts/train_physio_model.py` trains:

- logistic regression
- random forest

Both are evaluated with `GroupKFold`, where the group is `subject_id`. This is
important because random train/test splits would leak subject-specific physiology
patterns across splits and give overly optimistic results.

Baseline results from `outputs/physio_model_metrics.json`:

| Model | AUC | F1 | Accuracy |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.855 | 0.581 | 0.777 |
| Random forest | 0.883 | 0.620 | 0.817 |

The random forest performs better on the engineered features. Its most important
features are:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `rpeaks_count` | 0.255 |
| 2 | `hr_mean_bpm` | 0.194 |
| 3 | `hrv_sdnn_ms` | 0.124 |
| 4 | `hrv_rmssd_ms` | 0.122 |
| 5 | `hr_std_bpm` | 0.109 |

This is consistent with the intuition that stress is reflected strongly in heart
rate and heart-rate dynamics, while respiration contributes additional context.

### Strict Calibrated Training

`scripts/train_physio_model_strict.py` builds a more deployment-oriented bundle:

- median imputation
- subject-wise z-normalization
- random forest classifier
- out-of-fold probability generation with grouped CV
- isotonic probability calibration
- threshold tuning for F1 and for an asymmetric cost objective

This bundle is saved as:

```text
outputs/models/physio_rf_strict_calibrated_bundle.joblib
```

Strict calibrated results from `outputs/physio_model_metrics_strict.json`:

| Metric | Value |
| --- | ---: |
| Base out-of-fold AUC | 0.821 |
| Base out-of-fold Brier score | 0.133 |
| Calibrated out-of-fold AUC | 0.831 |
| Calibrated out-of-fold Brier score | 0.112 |

The lower Brier score after calibration means the probabilities are better
calibrated. In other words, the model's `p_stress` values are more useful as
probabilities for downstream fusion, not only as classifier scores.

The tuned F1 threshold is `0.24`:

| Metric at threshold 0.24 | Value |
| --- | ---: |
| F1 | 0.579 |
| Precision | 0.485 |
| Recall | 0.720 |
| Accuracy | 0.769 |

The cost-tuned threshold is `0.31`, using false negatives as twice as costly as
false positives:

| Metric at threshold 0.31 | Value |
| --- | ---: |
| F1 | 0.561 |
| Precision | 0.749 |
| Recall | 0.449 |
| Accuracy | 0.845 |

These two thresholds illustrate a central design tradeoff:

- a lower threshold catches more stress windows but creates more false alarms
- a higher threshold is more conservative but misses more stress windows

For a health-warning prototype, that tradeoff is not purely technical. It is a
product and safety decision.

## Air-Risk Pipeline

`src/air_quality_loader.py` cleans the UCI Air Quality data. `src/air_risk.py`
then turns pollutant values into normalized risk components:

- `risk_CO`
- `risk_NO2`
- `risk_NOx`
- `risk_weather`
- `air_risk_score`

The default risk score is a weighted sum:

- CO weight: 0.2
- NO2 weight: 0.4
- NOx weight: 0.4

The pollutant values are normalized against configurable high-risk reference
levels and clipped to the range `[0, 1]`. The optional weather modifier adds only
a mild boost, so pollution remains the main driver.

The sample output is saved to:

```text
outputs/air_quality_with_risk_sample.csv
```

For the first 200 saved rows, the observed air-risk score ranges from about
`0.091` to `0.940`, with an average around `0.502`. Across the fusion demo, the
air-risk values range from `0.054` to `0.940`, with an average around `0.409`.

Diagnostic figures are saved in `outputs/figures/`:

- `air_risk_timeseries_first_30_days.png`
- `air_risk_hist.png`
- `air_risk_pollutant_correlations.png`

## Fusion Logic

The fusion stage combines:

- calibrated stress probability, `p_stress`
- air-risk score, `air_risk`
- recent stress trend

The rule-based policy in `src/fusion_policy.py` uses interpretable thresholds:

- medium stress threshold: tuned from the physiology bundle, currently `0.24`
- high stress threshold: medium threshold plus `0.15`, currently about `0.39`
- medium air-risk threshold: `0.40`
- high air-risk threshold: `0.65`
- rising-stress trend threshold: `0.10`

The policy maps combinations of physiology and air context into meaningful
states:

| State | Meaning |
| --- | --- |
| `normal` | Physiology and air context are not elevated. |
| `high_air_risk` | Air risk is high even if physiology is not strongly elevated. |
| `high_physiological_risk` | Physiological stress is high but air is not the main driver. |
| `elevated_context_risk` | Moderate stress plus high air risk suggests contextual risk. |
| `rising_stress` | Stress probability is rising before reaching high-risk level. |
| `high_risk_combined` | Physiology and air quality are both high risk. |

Each final state is mapped to an action and rationale. For example, combined high
risk recommends stopping or avoiding outdoor exertion, moving indoors, resting,
and limiting exposure.

## Learned Fusion Model

The rule-based fusion output is also used to train an explainable decision tree.
This is done in `scripts/train_fusion_model.py`.

Fusion features:

- `p_stress`
- `air_risk`
- `p_trend`
- `p_x_air`
- `p_plus_air`

The learned tree is intentionally shallow and interpretable. It is saved as:

```text
outputs/models/fusion_decision_tree.joblib
outputs/fusion_tree_rules.txt
```

The exported tree captures the intended logic in a compact way:

- when `p_stress` is high and `air_risk` is high, predict
  `high_risk_combined`
- when `p_stress` is high and `air_risk` is not high, predict
  `high_physiological_risk`
- when `p_stress` is low but `air_risk` is high, distinguish
  `high_air_risk` from `elevated_context_risk`
- otherwise predict `normal`

This is useful because the model can be inspected and discussed. The project does
not hide the final behavior inside an opaque fusion model.

## Temporal Smoothing and State Stability

Real alerting systems should not change state too rapidly. A warning that flips
between `normal`, `high_air_risk`, and `high_physiological_risk` every window
would be annoying and hard to trust.

`scripts/run_fusion_demo_learned.py` therefore adds:

- exponential moving average smoothing for `p_stress` and `air_risk`
- special-state enforcement for important states such as `high_risk_combined`
  and `rising_stress`
- hysteresis over status labels, requiring confirmation before switching states
- minimum dwell times so important states persist briefly once entered

This creates a more realistic alert stream. The system still reacts quickly to
combined high risk, but ordinary status changes are made less noisy.

## Results

### Physiological Stress Detection

The best baseline model, a random forest on engineered physiological features,
achieved:

- AUC: `0.883`
- F1: `0.620`
- accuracy: `0.817`

The stricter calibrated model achieved:

- calibrated out-of-fold AUC: `0.831`
- calibrated Brier score: `0.112`
- F1 threshold: `0.24`
- cost-sensitive threshold: `0.31`

The strict model is the more appropriate model for fusion because it produces
calibrated probabilities and uses subject-separated validation.

### Rule-Based Strict Fusion Demo

The strict fusion demo is saved to:

```text
outputs/fusion_demo_results_strict.csv
```

Status distribution:

| Status | Count |
| --- | ---: |
| `normal` | 1,026 |
| `high_physiological_risk` | 217 |
| `high_air_risk` | 112 |
| `rising_stress` | 92 |
| `elevated_context_risk` | 27 |
| `high_risk_combined` | 25 |

This shows that the policy can separate different kinds of risk instead of
collapsing everything into a single alarm state.

### Learned and Smoothed Fusion Demo

The learned and smoothed demo is saved to:

```text
outputs/fusion_demo_results_learned_smoothed.csv
```

Status distribution:

| Status | Count |
| --- | ---: |
| `normal` | 1,175 |
| `high_physiological_risk` | 248 |
| `high_air_risk` | 43 |
| `high_risk_combined` | 21 |
| `rising_stress` | 12 |

Compared with the rule-based strict output, smoothing makes the final state
stream more conservative and stable. It reduces short-lived `rising_stress` and
context-only warnings, while preserving important combined-risk events.

Generated figures include:

- `fusion_demo_signals_strict.png`
- `fusion_demo_status_counts_strict.png`
- `fusion_signals_learned_smoothed.png`
- `fusion_status_counts_learned_smoothed.png`

## What Was Achieved

This project achieved a working end-to-end ambient intelligence prototype:

- WESAD physiology can be loaded, windowed, labeled, and transformed into a
  machine-learning feature table.
- A subject-aware stress classifier can detect stress from ECG and respiration
  features with useful discrimination.
- Probability calibration improves the usefulness of model outputs for downstream
  decision-making.
- UCI air-quality signals can be transformed into a simple, interpretable
  environmental risk score.
- Physiology and environmental context can be fused into meaningful warning
  states.
- The final system can output not only a status but also a recommended action and
  rationale.
- A learned decision tree can approximate the explicit fusion policy while
  remaining human-readable.
- Temporal smoothing and hysteresis make the alert stream more realistic and less
  noisy.

The most important achievement is the full multimodal flow. The repository does
not stop at training a classifier. It demonstrates how a classifier can become
part of a larger context-aware decision system.

## What Can Be Learned

### 1. Subject-aware evaluation matters

Physiological data is highly person-specific. If windows from the same subject
appear in both train and test sets, the model can learn the subject instead of
learning stress. Grouped validation by `subject_id` gives a more honest estimate
of generalization.

### 2. Accuracy is not enough

The dataset is imbalanced, with many more non-stress windows than stress windows.
A model can get decent accuracy while missing too many stress cases. F1,
precision, recall, AUC, calibration, and confusion matrices all reveal different
parts of the story.

### 3. Probability calibration is valuable for fusion

The fusion layer needs probabilities that behave like probabilities. Calibration
lowered the Brier score from `0.133` to `0.112`, which makes the stress signal
more trustworthy as an input to thresholds and decision rules.

### 4. Context changes the meaning of physiology

A stress probability of `0.35` is not interpreted the same way under low air risk
and high air risk. The ambient context can turn a moderate physiological signal
into a reason to reduce outdoor exertion.

### 5. Explainability is useful at the fusion layer

The project uses explicit policies and a shallow decision tree because the final
states should be understandable. For health-related recommendations, it is not
enough to say "the model predicted risk." The system should be able to say why.

### 6. Alert stability is part of model quality

A technically correct state stream can still be unpleasant if it changes too
often. Smoothing, confirmation windows, and minimum dwell times are important for
turning predictions into usable interactions.

### 7. Synthetic alignment is useful but limited

The WESAD physiology and UCI Air Quality datasets are not recordings from the
same people at the same time. The fusion demo cycles air-risk values across WESAD
windows to test the mechanics of multimodal fusion. This is useful for prototype
development, but it cannot prove real-world causal health effects.

## How to Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Check WESAD loading:

```bash
py scripts/test_wesad_loader.py
```

Build WESAD features:

```bash
py scripts/build_wesad_features.py
```

Train baseline physiology models:

```bash
py scripts/train_physio_model.py
```

Train the strict calibrated physiology bundle:

```bash
py scripts/train_physio_model_strict.py
```

Check air-quality loading:

```bash
py scripts/test_air_quality.py
```

Compute air-risk diagnostics:

```bash
py scripts/test_air_risk.py
```

Run the strict rule-based fusion demo:

```bash
py scripts/run_fusion_demo.py
```

Train the learned fusion tree:

```bash
py scripts/train_fusion_model.py
```

Run the learned and smoothed fusion demo:

```bash
py scripts/run_fusion_demo_learned.py
```

## Important Limitations

- The fusion demo uses synthetic temporal alignment between WESAD and UCI Air
  Quality data. The two datasets were not collected together.
- The ECG and respiration feature extraction is intentionally simple.
- The air-risk thresholds are engineering defaults, not clinical or regulatory
  thresholds.
- WESAD stress labels represent controlled experimental stress, not all forms of
  health deterioration.
- The prototype has not been validated on real simultaneous wearable and
  environmental recordings.
- Recommendations are illustrative and should not be treated as medical advice.

## Next Steps

Useful extensions would include:

- replacing the simple ECG peak detector with a validated physiological-signal
  toolkit
- adding confidence intervals or uncertainty estimates for stress probabilities
- evaluating on external subjects or another wearable dataset
- collecting synchronized physiology and air-quality data
- tuning air-risk thresholds against accepted air-quality standards
- adding personalized baselines and online adaptation
- testing how users respond to alert frequency, wording, and dwell times

## Summary

The project demonstrates that an ambient health-monitoring system can be built as
a transparent pipeline rather than a black box. Physiological signals estimate
stress, environmental signals estimate air risk, and a fusion layer turns both
streams into understandable states and actions. The results show promising
prototype performance, especially for stress discrimination and calibrated
probability fusion, while also making clear what still needs real-world
validation.
