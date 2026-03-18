# Area Feasibility Scoring Model

> **Can a user afford to live in a given area?**  
> A leakage-safe classification pipeline built on UK property price data.

---

## Project Overview

Given a user's budget and an area's historical property price distribution, this model predicts whether the area is **affordable** (binary classification: Yes / No).

A secondary output is the **estimated median price** of the area, giving users a concrete anchor alongside the yes/no decision.

### Why this matters

| Skill demonstrated | Detail |
|---|---|
| Feature engineering | Area-level price statistics, supply signals, interaction terms |
| Model selection | Logistic regression baseline vs Random Forest |
| Leakage awareness | `LeakageSafeFeaturizer` enforces fit-on-train-only |
| Real-world framing | UK Land Registry data, postcode districts |
| Geospatial thinking | Area aggregation by postcode district |

---

## Repository Structure

```
area-affordability-model/
│
├── data/                        ← Raw and processed data files
│   └── .gitkeep                 (add pp-complete.csv here)
│
├── notebooks/
│   ├── 01_eda.ipynb             Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  Feature pipeline walkthrough
│   └── 03_modeling.ipynb        Training, CV, evaluation, prediction
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           Data loading & area-level aggregation
│   ├── features.py              LeakageSafeFeaturizer + feature constants
│   ├── train.py                 Training pipeline (LR + RF, CV, persistence)
│   └── evaluate.py              Metrics, comparison, importances, plots
│
├── models/                      ← Trained model pickle files (git-ignored)
│   └── .gitkeep
│
├── requirements.txt
└── README.md
```

---

## Data

### Option A – UK Land Registry Price Paid (recommended)

Download the full dataset from:  
<https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads>

Save the file as `data/pp-complete.csv`.  
The loader handles the headerless CSV format automatically.

### Option B – Synthetic data

Run any script with `--use-synthetic` (or omit `--data`) and a realistic synthetic dataset will be generated automatically. No download required.

---

## Features

All features are computed inside a scikit-learn `Pipeline` via `LeakageSafeFeaturizer`, which is **fit on training data only** — preventing test-set statistics from leaking into the model.

| Feature | Description |
|---|---|
| `affordability_ratio` | `budget / median_price` |
| `budget_vs_p25` | `budget / price_25th` |
| `budget_vs_p75` | `budget / price_75th` |
| `price_spread` | `price_75th − price_25th` (IQR) |
| `price_spread_pct` | `price_spread / median_price` |
| `pct_within_budget` | Estimated % of listings ≤ budget (Gaussian approximation) |
| `budget_deficit` | `median_price − budget` (negative = affordable) |
| `log_budget` | `log1p(budget)` |
| `log_median_price` | `log1p(median_price)` |
| `log_num_listings` | `log1p(num_listings)` (supply proxy) |
| `ratio_x_spread` | `affordability_ratio × price_spread_pct` (interaction) |

### Classification target

```
affordable = 1  if  budget >= median_price
           = 0  otherwise
```

---

## Models

| Model | Role |
|---|---|
| `LogisticRegression` | Interpretable baseline; scale-sensitive, linear boundaries |
| `RandomForestClassifier` | Main comparator; captures non-linear interactions |

Both are wrapped in a `Pipeline`:

```
LeakageSafeFeaturizer → StandardScaler → Classifier
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train models

```bash
# Using synthetic data (no download required)
python src/train.py --use-synthetic --budget 300000

# Using real Land Registry data
python src/train.py --data data/pp-complete.csv --budget 300000
```

### 3. Evaluate models

```bash
python src/evaluate.py --use-synthetic --budget 300000
```

### 4. Run notebooks

```bash
jupyter lab
```

Open `notebooks/01_eda.ipynb` and run through in order.

---

## Leakage Prevention

Data leakage is the most common source of over-optimistic model performance. This project prevents it at three levels:

1. **Split before featurize** – The train/test split is performed on raw area data *before* any feature statistics are computed.
2. **Pipeline encapsulation** – `LeakageSafeFeaturizer` implements the scikit-learn `fit` / `transform` interface. Inside `cross_validate`, scikit-learn calls `fit` on each training fold and `transform` on the validation fold separately.
3. **Target exclusion** – `assert_no_leakage()` is called before every model fit to confirm that no feature column matches a target column.

---

## Evaluation Metrics

| Metric | Why chosen |
|---|---|
| Accuracy | Baseline overall correctness |
| Precision | Avoid falsely labelling unaffordable areas as affordable |
| Recall | Avoid missing genuinely affordable areas |
| F1 | Harmonic mean; preferred for potentially imbalanced classes |
| ROC-AUC | Threshold-independent ranking quality |

---

## Reproducibility

All random seeds are set via `RANDOM_STATE = 42` in `src/train.py`.  
Synthetic data generation uses `numpy.random.default_rng(42)`.

---

## License

This project is for educational and portfolio purposes.  
UK Land Registry data is © Crown copyright and licensed under the  
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).