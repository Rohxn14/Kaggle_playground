# Store Sales: Time Series Forecasting

Direct multi-horizon LightGBM for the Kaggle Getting Started competition [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting): 16 days of daily sales for 1,782 store × product-family series of Corporación Favorita, an Ecuadorian grocery chain.

**Status:** the notebook is written but has not been run or submitted yet, so there is no score.

## Problem

- **Task:** forecast `sales` for 54 stores × 33 product families on every day from 16 to 31 August 2017.
- **Metric:** RMSLE (root mean squared logarithmic error). It measures relative errors, so small families count as much as large ones; training on `log1p(sales)` with squared-error loss optimises it directly.
- **Data:** 3.0M daily rows from 1 January 2013 to 15 August 2017, plus:
  - `onpromotion`, which is also given for the test days
  - store metadata (city, state, type, cluster)
  - national, regional and local holidays
  - oil prices and store transactions (not used)
- **Leaderboard:** permanent, rolling (entries drop off after about two months). In October 2026 the top was 0.3717, the top 10% 0.389 and the median 0.469. The best-known public notebook, a global LightGBM built with Darts, scores 0.380.

## Notebooks

| Notebook | What it is |
|---|---|
| `store-sales.ipynb` | Earlier exploratory notebook: sales plots for one store and family, weekday and monthly patterns, a first validation split |
| `main.ipynb` | Full pipeline: EDA, series × day matrices, 16 LightGBM models (one per forecast day), holdout validation against two baselines, refit, submission |

## Approach

### Why one model per forecast day
The original 2018 Favorita competition (same company, item-level data) was won by this design. On each of 136 past Wednesdays from January 2015 (the test window also starts on a Wednesday), the notebook snapshots the history and computes features from the days before it. It then trains a separate LightGBM model for each of the 16 days ahead. Nothing is forecast recursively, so errors do not compound, and each model learns its own horizon (day 1 relies on the last few days, day 16 on longer averages and the weekly profile).

### Features (119 per model)
- **Recent level:** means over 3 to 224 days, plus medians, standard deviations, zero-day shares and an exponentially weighted mean
- **Last week:** each of the last 7 days
- **Weekly profile:** the mean of each weekday's last 4, 12 and 26 occurrences, and the family-wide weekday profile
- **Context:** the store's total sales and the family's average over all stores, over the same windows
- **Promotions:** past intensity; promoted items on each of the 16 forecast days; mean sales on promoted vs non-promoted days
- **Target day:** promotion, its weekday profile, day of month, payday (15th and month end), national, regional and local holidays
- **Static:** store, family, type, cluster, city, state as categoricals

### Data handling
- Christmas Day (stores closed, no rows) is filled with 0.
- Targets during the April-May 2016 earthquake period are excluded from training.
- Series with no sales in the last 56 days are forecast as 0 and left out of training.
- **Validation:** holdout window 26 July - 10 August 2017. Each horizon trains only on snapshots whose target day falls before it, and early stopping chooses the trees. The models are then refitted on all data with 10% more trees.

## Running the notebook

- Uses the repository's shared Python 3.11 virtual environment (`../.venv`, kernel "Python 3.11 (kaggle_playground .venv)"); needs pandas, NumPy, Matplotlib and LightGBM only.
- Download the data into this folder (data files are git-ignored):
  ```
  kaggle competitions download store-sales-time-series-forecasting
  ```
- Runtime has not been measured. Expect roughly 15-30 minutes on a 4-core laptop CPU (32 LightGBM fits on about 220k rows each).
- `SUBMIT = False` by default. To submit, accept the competition rules on Kaggle first (otherwise the CLI is refused with 400 Bad Request), then set it to `True`.

## Credits
- Favorita Grocery Sales Forecasting (2018) 1st place: snapshot-based direct multi-horizon gradient boosting
- Chong Zhen Jie, "Ecuador Store Sales - Global Forecasting LightGBM": zero forecasts for dead series, holiday handling, the 0.380 reference score
- Ryan Holbrook, Kaggle Learn time-series course: why promotions and holidays are legitimate future covariates

## Technologies

- Python
- LightGBM
- Pandas, NumPy, Matplotlib
