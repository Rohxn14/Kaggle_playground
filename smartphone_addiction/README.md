# Predicting Smartphone Addiction

Gradient-boosting ensemble for the Kaggle Playground Series S6E8 competition, predicting whether a person is addicted to their smartphone.

**Private leaderboard: 0.96964 ROC-AUC** (late submission; would have placed about 826 of 3,532, top 23%). Public leaderboard 0.96985.

## Problem

- **Task:** binary classification. Predict the probability of `addicted_label` (70.9% positive).
- **Metric:** ROC-AUC; raw probabilities are submitted.
- **Data:** 691,369 training and 296,302 test rows:
  - 9 numeric: age, daily screen time and its breakdown into social media, gaming and work/study, sleep, notifications, app opens, weekend screen time
  - 3 categorical: gender, stress level, academic/work impact
  - **every column is missing 4-20% of the time**, at different rates in train and test
- **Synthetic data:** the source dataset's page disappeared during the competition.
- **Timeline:** August 2026. Late submission, scored on both leaderboards but not ranked.

## Notebook

| Notebook | Runs on | What it does | Score |
|---|---|---|---|
| `main.ipynb` | Local CPU (about 25 minutes) | EDA, screen-time budget and rule features, exact-value frequency and target encoding, LightGBM + XGBoost blend, submission via the Kaggle CLI | CV 0.96855 · public 0.96985 · private 0.96964 |

## Approach

### How the labels were made
In the source data:

| Condition | Label |
|---|---|
| daily screen time > 8 h, or social media > 4 h | addicted |
| daily ≤ 6 h and social media ≤ 4 h | not addicted |
| in between (6-8 h daily, social ≤ 4 h) | coin flip |

The rule needs daily screen time, which is missing for 14% of rows.

### The screen-time budget
On every complete row, daily screen time is at least social media + gaming + work/study. When daily time is missing, the sum of the known parts is therefore a hard **lower bound** for it: if the parts already exceed 8 hours, the person is addicted whatever the missing value was. In the same way, a missing part is bounded above by daily time minus the other parts. These features were the 2nd-place team's breakthrough.

### Features (53 columns + 9 target encodings per fold)
- **Budget:**
  - sum of the known parts, and how many parts are missing
  - daily lower bound, unexplained remainder, part shares
  - upper bound for missing social media
- **Rule:** "surely addicted", "surely not", the rule's verdict where decidable, distances to the 8-hour and 4-hour thresholds
- **Weekend vs weekday**, notifications and app opens per screen hour, number of missing values
- **Exact values:** decimal part, frequency in train + test, and fold-safe target encoding of each numeric value (the generator reuses a finite set of values)

### Models and stack
| Model | OOF AUC |
|---|---|
| LightGBM (127 leaves, 35% of features per tree, 1023 bins, lr 0.06) | 0.96819 |
| XGBoost (depth 7, lr 0.08) | 0.96826 |
| **Logistic-regression stack of the two** | **0.96855** |

Learning rates are higher than in the public notebooks (0.06 vs 0.01) so the notebook runs in well under an hour on a laptop CPU.

## Running the notebook

- Uses the repository's shared Python 3.11 virtual environment (`../.venv`).
- Place `train.csv`, `test.csv` and `sample_submission.csv` from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e8) in this folder. Data files are git-ignored.
- The last cells submit with the Kaggle CLI and wait for the score. Set `SUBMIT = False` to re-run without submitting.

## Credits
- broccoli beef: generation model of the original labels
- Xin Feng (2nd place): budget features (`fake_daily`, `fake_social`)
- Naji: strongest public single LightGBM (budget, decimal, frequency and target-encoding features)
- chloeprice: target encoding of exact numeric values

## Technologies

- Python
- LightGBM, XGBoost
- scikit-learn
- Pandas, NumPy, Matplotlib
