# NLP with Disaster Tweets

Transformer ensemble for Kaggle's [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) Getting Started competition, predicting whether a tweet is about a real disaster.

**Status:** the notebook is built for Kaggle's GPUs and has not been scored yet.
- **First Kaggle run:** Twitter-RoBERTa-large trained fine (24 minutes), but DeBERTa-v3-large crashed with *"Attempting to unscale FP16 gradients"*. Its checkpoint is stored in fp16, recent transformers versions load it that way, and mixed-precision training needs fp32 weights.
- **Fix:** every model is now converted to fp32 before training. Locally, only the CPU debug mode has been re-run.

## Problem

- **Task:** binary classification. Predict `1` if a tweet is about a real disaster, `0` otherwise. The hard cases use disaster words figuratively ("this song is fire").
- **Metric:** F1 score of the positive class. It ignores true negatives, so the decision threshold is tuned.
- **Data:** 7,613 labelled and 3,263 test tweets, with an optional `keyword` (221 values) and a free-text `location` (a third missing, not used). 43% of training tweets are disasters.
- **Leaderboard:** the perfect 1.000 scores at the top are look-ups of the publicly available source labels. Honest models score about 0.83-0.85.

## Notebook

| Notebook | Runs on | What it does |
|---|---|---|
| `disaster-tweets-gpu-transformer-ensemble.ipynb` | Kaggle, GPU T4 ×2 (or P100) | Cleaning and relabelling, 5-fold Twitter-RoBERTa-large and DeBERTa-v3-large, TF-IDF + logistic regression, blend and threshold tuned on out-of-fold F1 |

## Approach

### Data preparation
- **Cleaning:** HTML entities decoded, broken UTF-8 fragments (`\x89Û_`, `Ûª`) repaired or dropped, links replaced by `http` and user names by `@user` (the convention Twitter-RoBERTa was pre-trained with).
- **Relabelling:** 18 tweets appear several times with both labels (55 rows). Each group gets its majority label; exact ties become 0.
- **Input:** the keyword and the tweet are passed as a text pair, `<s> keyword </s></s> tweet </s>`.

### Models
- **[Twitter-RoBERTa-large](https://huggingface.co/cardiffnlp/twitter-roberta-large-2022-154m):** RoBERTa-large further pre-trained on 154M tweets, so it knows Twitter slang and hashtags.
- **[DeBERTa-v3-large](https://huggingface.co/microsoft/deberta-v3-large):** the strongest general-purpose encoder; a different model family for the blend.
- **TF-IDF + logistic regression:** word 1-2 grams and character 2-5 grams. Weaker, but it makes different mistakes.

Each transformer is fine-tuned on the same stratified 5 folds:
- 3 epochs, AdamW at learning rate 1e-5, warm-up over 10% of the steps then linear decay
- batches of 16 tweets of similar length, fp32 weights with fp16 mixed precision, gradient clipping at 1.0
- the epoch with the highest validation AUC is kept per fold. Validation log-loss was the first criterion, but on the first run it bottomed out after epoch 1 while F1 still improved at epoch 2.

The two models train at the same time as separate worker processes, one per GPU.

### Blend and threshold
The out-of-fold probabilities of the three models are blended with weights in steps of 0.1. The weights and the decision threshold are chosen to maximise out-of-fold F1.

## Running the notebook

**Kaggle**
1. Create a notebook from the competition page (the data is attached automatically) and import the `.ipynb`.
2. Settings: Accelerator **GPU T4 ×2**, **Internet ON** (the models download from Hugging Face).
3. *Save Version → Save & Run All* (about 1 hour), then submit `submission.csv` from the version's Output tab.

**Locally**
- Uses the repository's shared virtual environment (`../.venv`), which includes PyTorch (CPU) and Transformers.
- Place `train.csv`, `test.csv` and `sample_submission.csv` from the competition page in this folder. Data files are git-ignored.
- Without a GPU the notebook runs in **DEBUG** mode: 300 tweets, a base-size model, 2 folds and 1 epoch, in about 5 minutes. It checks the pipeline and does not write a submission.

## Credits
- Gunes Evitan, *NLP with Disaster Tweets - EDA, Cleaning and BERT*: cleaning and relabelling of the duplicated tweets
- Cardiff NLP (Twitter-RoBERTa) and Microsoft (DeBERTa-v3) for the pre-trained models

## Technologies

- Python
- PyTorch, Hugging Face Transformers
- scikit-learn
- Pandas, NumPy, Matplotlib
