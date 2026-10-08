# Predicting F1 Pit Stops

Gradient-boosting ensemble for the Kaggle Playground Series S6E5 competition, predicting whether a Formula 1 driver pits on the next lap.

**Private leaderboard: 0.95294 ROC-AUC** (late submission; would have placed about 659 of 3,023, top 22%). Public leaderboard 0.95235.

## Problem

- **Task:** binary classification. Predict the probability that a driver pits on the next lap (`PitNextLap`, 19.9% of laps).
- **Metric:** ROC-AUC. Only the ranking of predictions matters, so raw probabilities are submitted.
- **Data:** 439,140 training and 188,165 test laps with 14 features:
  - 3 categorical: Driver (887 codes), Compound, Race
  - 11 numeric: Year, PitStop, LapNumber, Stint, TyreLife, Position, lap time, lap-time delta, cumulative degradation, race progress, position change
- **Synthetic data:** generated from the [F1 Strategy Dataset](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction) (101,371 real laps, the *original dataset*). Kaggle removed its `Normalized_TyreLife` column because it makes the task trivial.
- **Timeline:** May 2026. The competition had closed, so the submission is a late submission, scored on both leaderboards but not ranked.

## Notebook

| Notebook | Runs on | What it does | Score |
|---|---|---|---|
| `main.ipynb` | Local CPU (about 2 hours, half of it CatBoost) | EDA, adversarial validation, feature engineering, LightGBM ×2 + XGBoost + CatBoost with 5-fold CV, logistic-regression stack, submission via the Kaggle CLI | CV 0.95321 · public 0.95235 · private 0.95294 |

## Approach

### What the winners did
- **1st place (private 0.95503):** 186 models of many families. The late breakthrough was training some models **without the Driver column**, the column that differs most from the original data.
- **2nd place (0.95502):** 218 models built by an autonomous coding agent on four A100 GPUs, blended by logistic regression on logits. The best single models were RealMLP (CV 0.9544) and XGBoost (CV 0.9536).
- **5th place:** "The one robustly positive signal was appending the original dataset as extra training rows."
- **7th place:** `LapNumber / RaceProgress`, the race's total laps, is a very strong feature.

### Original data as extra rows
All 101k original laps are appended to the training part of every fold, never to the validation part. On one fold this raised LightGBM's AUC from 0.95259 to 0.95329.

### Features (44 columns + 6 target encodings per fold)
- **Race length:** `total_laps = LapNumber / RaceProgress`, laps left, tyre life as a share of the race
- **Tyre wear:** tyre life per lap, lap of the last stop, degradation per tyre lap
- **Lap-time interactions:** lap time × degradation and variants, from the strongest public notebook
- **Keys:** Race × Year, Race × Compound, Compound × Stint, Race × Year × Compound, Driver × Year
- **Count encoding** of every key and of the main numeric columns (train + test + original)
- **Target encoding** of Driver and the five keys, fitted inside each fold with internal cross-fitting so no row sees its own label

### Models and stack
| Model | OOF AUC |
|---|---|
| LightGBM (127 leaves, lr 0.02) | 0.95252 |
| LightGBM without any driver feature | 0.95249 |
| XGBoost (depth 8, lr 0.03) | 0.95226 |
| CatBoost (depth 7, native categorical keys) | 0.95157 |
| **Logistic regression on the four models' OOF logits** | **0.95321** |

The stack is cross-validated on the same folds, so its OOF AUC is honest. Removing every driver feature cost nothing (0.95252 vs 0.95249): the synthetic driver codes carry almost no signal, which is why the 1st-place team could drop them.

Stack weights:
- **CatBoost: 0.30.** It is the least correlated model, at about 0.973 with each of the others, against 0.99+ among the other three.
- **The two LightGBMs: 0.59 between them** (0.23 and 0.36).
- **XGBoost: 0.07.** Its predictions correlate 0.994 with the first LightGBM.

## Running the notebook

- Uses the repository's shared Python 3.11 virtual environment (`../.venv`), which includes CatBoost.
- Place `train.csv`, `test.csv` and `sample_submission.csv` from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e5) in this folder, and the original dataset in `original/`:
  ```
  kaggle datasets download vanshjasuja16/f1-strategy-dataset-pit-stop-prediction -p original --unzip
  ```
  (The author's copy is no longer downloadable through the API; this is a mirror of the same file.) Data files are git-ignored.
- The last cells submit with the Kaggle CLI and wait for the score. Set `SUBMIT = False` to re-run without submitting.

## Credits
- Optimistix (1st place): models without the Driver column
- Chris Deotte (2nd place) and Data User (5th place): logistic-regression stacking on logits
- Vladimir Demidov: feature recipe of the public RealMLP notebook
- Jerry (7th place): the race-length feature

## Technologies

- Python
- LightGBM, XGBoost, CatBoost
- scikit-learn
- Pandas, NumPy, Matplotlib
