# Predicting Airline Satisfaction

Gradient-boosting and neural-network ensemble for the Kaggle Playground Series S6E10 competition, predicting whether an airline passenger was satisfied with their flight.

**Best public leaderboard score: 0.96122 ROC-AUC** (about rank 212 of 936, top 23%).

## Problem

- **Task:** binary classification. Predict the probability that each passenger is satisfied.
- **Metric:** ROC-AUC. Only the ranking of predictions matters, so raw probabilities are submitted and no class rebalancing is needed.
- **Data:** 699,635 training and 299,844 test rows with 21 features:
  - 4 categorical: Gender, Customer Type, Type of Travel, Class
  - 4 numeric: Age, Flight Distance, departure and arrival delay
  - 13 service ratings on a 0–5 scale, where 0 means "not applicable"
- **Synthetic data:** Kaggle generated the data from a real survey of 129,880 passengers ([Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)), referred to below as the *original dataset*.

## Notebooks

| Notebook | Runs on | What it does | Score |
|---|---|---|---|
| `main.ipynb` | Local CPU | EDA, LightGBM + XGBoost baseline with 5-fold CV, rank blend, submission via the Kaggle CLI | CV 0.95896 · LB 0.95823 |
| `airline-satisfaction-dual-t4-gpu-ensemble.ipynb` | Kaggle, GPU T4 ×2 | Competition data only: pairwise target encoding, Optuna-tuned XGBoost, CatBoost, MLP, logistic stacking | best single model CV 0.95897 |
| `airline-satisfaction-v2-routes-original-lookup-gpu-stack.ipynb` | Kaggle, GPU T4 ×2 | **Final.** Original-data lookups, route features, XGBoost ×2, CatBoost and RealMLP, stacked | **LB 0.96122** |

## Approach

### What moved the score
Adding more models and hardware (the second notebook) left the score unchanged at about 0.959. The improvement came from two findings about how the data was made:

1. **Use the original dataset as a lookup, not as extra rows.**
   - Appending the original rows *lowered* CV.
   - Adversarial validation shows the original is a different population: a classifier separates it from the training data with AUC 0.835.
   - Instead, each value's satisfaction rate *in the original dataset* is used as a feature, along with the prediction of a model trained only on the original data.
   - No competition labels are involved, so this cannot leak.
2. **`Flight Distance` is also a route ID.**
   - It has about 3,500 distinct values.
   - Satisfaction varies by the exact distance value far beyond sampling noise.
   - Route-level features capture this: the passenger count per distance, the average of every other column on that route, and each passenger's deviation from that average.

### Final feature set (about 186 features per model)
- **Original-dataset lookups** for every column and 14 key pairs, plus an original-only LightGBM prediction
- **Digit features:** individual digits of age, distance and delays, which capture artefacts of the synthetic data generator
- **Frequency features:** how common each value and key pair is
- **Route profile:** the per-distance statistics described above
- **Target encoding** of 40 keys (columns, bins and pairs), computed inside each CV fold with inner cross-fitting so no row sees its own label

### Models and ensemble
- **XGBoost:** a deep and a shallow variant. Folds train two at a time, one per GPU.
- **CatBoost:** each model trains across both GPUs (`devices='0:1'`).
- **RealMLP** ([pytabkit](https://github.com/dholzmueller/pytabkit)): a configured neural network that is as accurate as the trees but makes different errors. It runs as two worker processes, one pinned to each GPU.
- **Stacking:** a logistic regression on the models' out-of-fold logits. It is scored out of fold and compared with a weighted rank blend; the better one is submitted.

All models share the same stratified 5-fold split. Decisions were based on CV rather than the public leaderboard, which scores only about 20% of the test set and is too noisy for differences of about 0.0002.

## Running the notebooks

**Local (`main.ipynb`)**
- Uses the repository's shared Python 3.11 virtual environment (`../.venv`).
- Place `train.csv`, `test.csv` and `sample_submission.csv` from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e10) in this folder. Data files are git-ignored.
- The last cells submit with the Kaggle CLI. A guard skips resubmitting an identical model.

**Kaggle (the two GPU notebooks)**
1. Import the notebook and add the competition data as input. For v2, also add the original dataset; if it isn't attached, the notebook downloads it with `kagglehub`.
2. Settings: Accelerator **GPU T4 ×2**, **Internet ON** (v2 installs `pytabkit`).
3. Run with *Save Version → Save & Run All* (about 1–1.5 h), then submit `submission.csv` from the version's output.

Off Kaggle, both GPU notebooks switch to a DEBUG mode that runs the whole pipeline on a 30,000-row sample on a CPU.

## Credits
The final notebook re-implements ideas published by other competitors and credits them in its last cell:
- evgendvorkin: original-data lookups, digit, frequency and target-encoding blocks
- goodpjw2008: Flight Distance as a route ID
- busyaprime: block-by-block measurements
- Vladimir Demidov: RealMLP configuration

## Technologies

- Python
- XGBoost, LightGBM, CatBoost
- PyTorch, pytabkit (RealMLP)
- scikit-learn, Optuna
- Pandas, NumPy, Matplotlib, Seaborn
