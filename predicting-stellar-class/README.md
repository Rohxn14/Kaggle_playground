# Predicting Stellar Class

Gradient-boosting ensemble for the Kaggle Playground Series S6E6 competition, classifying sky objects as galaxies, quasars or stars.

**Private leaderboard: 0.96724 balanced accuracy** (late submission; would have placed about 688 of 2,817, top 24%). Public leaderboard 0.96774.

## Problem

- **Task:** 3-class classification: `GALAXY` (65.4%), `QSO` (quasar, 20.3%), `STAR` (14.3%).
- **Metric:** balanced accuracy, the mean of the per-class recalls. Rare classes count as much as the common one, so the decision rule matters.
- **Data:** 577,347 training and 247,435 test objects:
  - sky position (`alpha`, `delta`)
  - magnitudes in the five SDSS filters `u g r i z`
  - `redshift`
  - two categorical columns, which turn out to be thresholds on colour indices
- **Synthetic data:** generated from the [SDSS17 stellar classification dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) (100,000 real spectra, the *original dataset*).
- **Timeline:** June 2026. Late submission, scored on both leaderboards but not ranked.

## Notebook

| Notebook | Runs on | What it does | Score |
|---|---|---|---|
| `main.ipynb` | Local CPU (about 15 minutes) | EDA, astronomy feature engineering, original-data class priors, LightGBM + XGBoost with balanced class weights, class multipliers tuned on OOF, submission via the Kaggle CLI | CV 0.96720 · public 0.96774 · private 0.96724 |

## Approach

### What the winners did
- **Decision rule first:** training with balanced class weights was "the single biggest honest jump" (+0.008 for LightGBM, 12th place).
- **Features:** the strongest public XGBoost (Chris Deotte, 0.9677 CV) used colour indices, magnitude statistics, fluxes, redshift interactions, sky geometry, binned colour × redshift keys, the original data's class rates per key, and fold-safe target encoding.
- **Trust CV:** the winner fell to public rank 344 by not blending public submissions, and finished 1st on the private leaderboard.

### Features (116 columns + 33 target encodings per fold)
- **Colour indices:** all 10 pairwise magnitude differences (`u - g`, `g - r`, ...)
- **Brightness profile:** magnitude mean/std/min/max/range, slope and curvature across the filters, linear fluxes
- **Redshift interactions:** each magnitude and colour × redshift and ÷ |redshift|, distance-modulus and absolute-magnitude proxies
- **Sky geometry:** sin/cos of the coordinates and the 3-D unit vector
- **Keys (11):** colour × redshift cells, redshift in tenths, 5-degree sky patches, 64-quantile bins and their pairs. Each key gets:
  - its frequency
  - its class rates in the original data (no competition labels, so no leakage)
  - a multiclass target encoding fitted inside each fold

### Models and decision rule
| Step | OOF balanced accuracy |
|---|---|
| LightGBM, balanced class weights | 0.96664 |
| XGBoost, balanced class weights | 0.96668 |
| Probability blend | 0.96689 |
| **+ per-class multipliers tuned on OOF** | **0.96720** |

Balanced class weights make each model's argmax target balanced accuracy directly. The multipliers (GALAXY fixed at 1) absorb any remaining tilt; two parameters on 577k OOF rows cannot overfit meaningfully.

## Running the notebook

- Uses the repository's shared Python 3.11 virtual environment (`../.venv`).
- Place `train.csv`, `test.csv` and `sample_submission.csv` from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e6) in this folder, and the original dataset in `original/`:
  ```
  kaggle datasets download fedesoriano/stellar-classification-dataset-sdss17 -p original --unzip
  ```
  Data files are git-ignored.
- The last cells submit with the Kaggle CLI and wait for the score. Set `SUBMIT = False` to re-run without submitting.

## Credits
- broccoli beef: formulae for `spectral_type` and `galaxy_population`
- Chris Deotte: XGB v5 feature recipe and original-data priors
- Alexdruso (12th place): measured value of balanced class weights
- Jerry (8th place): weighted log-loss as a smooth stand-in for balanced accuracy

## Technologies

- Python
- LightGBM, XGBoost
- scikit-learn
- Pandas, NumPy, Matplotlib
