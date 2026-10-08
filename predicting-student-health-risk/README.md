# Predicting Student Health Risk

Gradient-boosting ensemble for the Kaggle Playground Series S6E7 competition, classifying students as at-risk, unhealthy or fit.

**Private leaderboard: 0.95049 balanced accuracy** (late submission; would have placed about 125 of 3,356, top 4%). Public leaderboard 0.94993.

## Problem

- **Task:** 3-class classification of `health_condition`: `at-risk` (85.9%), `unhealthy` (8.4%), `fit` (5.8%).
- **Metric:** balanced accuracy. Predicting at-risk for everyone scores 0.86 plain accuracy but only 0.333 balanced accuracy.
- **Data:** 690,088 training and 295,753 test rows:
  - 7 numeric: sleep duration, heart rate, BMI, calories, steps, exercise, water
  - 6 three-level categoricals: diet, stress, sleep quality, activity, smoking/alcohol, gender
  - **every column has 1-12% missing values**
- **Synthetic data:** generated from the [College Student Health Behavior Dataset](https://www.kaggle.com/datasets/ziya07/college-student-health-behavior-dataset) (50,000 rows, the *original dataset*).
- **Timeline:** July 2026. Late submission, scored on both leaderboards but not ranked.

## Notebook

| Notebook | Runs on | What it does | Score |
|---|---|---|---|
| `main.ipynb` | Local CPU (about 15 minutes) | EDA and label-rule check, rule features with missing-value bounds, exact-value target encoding, XGBoost + LightGBM, prior-corrected decision rule, submission via the Kaggle CLI | CV 0.95063 · public 0.94993 · private 0.95049 |

## Approach

### How the labels were made
A community post found that the original labels follow a points rule:

| Input | Points |
|---|---|
| sleep under 6 hours | 2 |
| stress: medium / high | 1 / 3 |
| activity: moderate or sedentary | 1 |

0 points means fit, 1-4 at-risk, 5-6 unhealthy. On synthetic rows where all three inputs are present (74% of the data), this rule alone scores 0.960 balanced accuracy.

The rest of the error has two sources:
- **The 0-point cell is ambiguous:** about 30% of students with no risk factor are labelled at-risk.
- **Missing inputs:** a quarter of the rows miss at least one rule input.

### Decision rule: the biggest lever
Models are trained on the natural class mix, then predict `argmax(p / class prior)` instead of `argmax(p)`. This is the balanced-accuracy-optimal rule for calibrated probabilities. It lifts the blend from 0.889 to 0.9505; two multipliers tuned on the OOF predictions give 0.95063.

### Features (29 columns + 39 target encodings per fold)
- **Rule features:**
  - the points from each input and their total
  - the total's lower and upper bounds when inputs are missing
  - The bounds often settle the class anyway: a lower bound of 5 means unhealthy whatever is missing.
- **Distance to the 6-hour sleep threshold**, and the number of missing values
- **Exact-value target encoding** of all 13 columns (the 4th-place idea): each value treated as a category and fitted inside each fold. Value counts are added for the numerics.
- Missing values are left as NaN; the trees route them natively.

### Models
| Model | OOF balanced accuracy (prior-corrected) |
|---|---|
| XGBoost (L2 12 and gamma 3 from the best single-XGBoost write-up; depth 6, lr 0.15) | 0.95040 |
| LightGBM (63 leaves, lr 0.1) | 0.95037 |
| **Blend + prior correction + multipliers** | **0.95063** |

The tuned multipliers (fit 0.74, unhealthy 0.96 on top of 1/prior) leave the per-class recalls almost equal: at-risk 0.939, fit 0.949, unhealthy 0.964.

## Running the notebook

- Uses the repository's shared Python 3.11 virtual environment (`../.venv`).
- Place `train.csv`, `test.csv` and `sample_submission.csv` from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e7) in this folder. The original dataset, used only to check the label rule, goes in `original/`:
  ```
  kaggle datasets download ziya07/college-student-health-behavior-dataset -p original --unzip
  ```
  Data files are git-ignored.
- The last cells submit with the Kaggle CLI and wait for the score. Set `SUBMIT = False` to re-run without submitting.

## Credits
- broccoli beef: generation model of the original labels
- Ricky (4th place): exact-value target encoding and the `p / prior` decision rule
- AbdullahSafwan333: single-XGBoost settings
- Masaya Kawamata: target-encoded XGBoost baseline and prior correction

## Technologies

- Python
- XGBoost, LightGBM
- scikit-learn
- Pandas, NumPy, Matplotlib
