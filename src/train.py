"""
train.py
--------
Model training pipeline for the Area Feasibility Scoring Model.

Two classifiers are trained and compared:
  1. LogisticRegression  – interpretable baseline
  2. RandomForestClassifier – tree-based model for non-linear patterns

Workflow
~~~~~~~~
1. Load prepared area-level data
2. Build the classification target (affordable: 1 / 0)
3. Stratified train / test split (80 / 20)
4. Build sklearn Pipelines (LeakageSafeFeaturizer → Scaler → Model)
5. Cross-validate both pipelines on the training set
6. Refit on full training set and evaluate on held-out test set
7. Persist trained pipelines to disk

Usage (CLI)
----------
    python src/train.py --use-synthetic
    python src/train.py --data data/area_prices.csv --budget 300000
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Allow running from the src/ directory directly
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import prepare_dataset
from features import (
    FEATURE_COLS,
    CLASSIFICATION_TARGET,
    LeakageSafeFeaturizer,
    assert_no_leakage,
    build_target,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def make_logistic_pipeline(C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    """Logistic Regression pipeline (baseline).

    Includes the leakage-safe featurizer so the Pipeline is fully
    self-contained and safe for cross-validation.
    """
    return Pipeline(
        [
            ("featurizer", LeakageSafeFeaturizer(add_interactions=False)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def make_rf_pipeline(n_estimators: int = 200, max_depth=None) -> Pipeline:
    """Random Forest pipeline (tree-based comparator).

    StandardScaler is included for API consistency even though Random
    Forests are scale-invariant.
    """
    return Pipeline(
        [
            ("featurizer", LeakageSafeFeaturizer(add_interactions=True)),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------

def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
) -> dict:
    """Run stratified k-fold CV and return mean ± std for key metrics.

    Parameters
    ----------
    pipeline : sklearn Pipeline
    X : pd.DataFrame
        Raw (pre-featurization) training data.
    y : pd.Series
        Binary labels.
    cv : int
        Number of folds.

    Returns
    -------
    dict
        ``{metric_name: (mean, std)}`` for accuracy, precision,
        recall, f1, roc_auc.
    """
    cv_strategy = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=False,
        error_score="raise",
    )

    results = {}
    for metric in scoring:
        key = f"test_{metric}"
        results[metric] = (float(scores[key].mean()), float(scores[key].std()))

    return results


def print_cv_results(model_name: str, results: dict, cv: int = CV_FOLDS):
    """Pretty-print cross-validation results."""
    print(f"\n{'─' * 55}")
    print(f"  {model_name} – {cv}-Fold Cross-Validation Results")
    print(f"{'─' * 55}")
    for metric, (mean, std) in results.items():
        print(f"  {metric:<15} {mean:.4f}  ±  {std:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    data_path: str = None,
    user_budget: float = 300_000,
    use_synthetic: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Full training workflow: load → split → CV → fit → persist.

    Parameters
    ----------
    data_path : str, optional
        Path to CSV. If None and ``use_synthetic=False``, raises an error.
    user_budget : float
        User's property budget in £.
    use_synthetic : bool
        Generate synthetic data if True.
    output_dir : str
        Directory where trained Pipeline pickles are saved.

    Returns
    -------
    dict
        Training artefacts including pipelines, splits, CV results,
        and test-set predictions.
    """
    # 1. Load data
    df = prepare_dataset(
        path=data_path,
        user_budget=user_budget,
        use_synthetic=use_synthetic,
    )

    # 2. Build target (before splitting – no statistics used, just a
    #    comparison between two columns, so no leakage risk)
    y = build_target(df)

    # Leakage guard: confirm no overlap between features and target
    assert_no_leakage(FEATURE_COLS, [CLASSIFICATION_TARGET])

    print(f"\nClass distribution:\n{y.value_counts().rename({0: 'Not Affordable', 1: 'Affordable'})}")
    print(f"Affordability rate: {y.mean():.1%}\n")

    # 3. Stratified train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        df,  # pass raw df; featurizer is inside the Pipeline
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # 4. Build pipelines
    lr_pipeline = make_logistic_pipeline()
    rf_pipeline = make_rf_pipeline()

    # 5. Cross-validate on training set only
    print("\n[Cross-validation on training set]")
    lr_cv = cross_validate_pipeline(lr_pipeline, X_train, y_train)
    print_cv_results("Logistic Regression", lr_cv)

    rf_cv = cross_validate_pipeline(rf_pipeline, X_train, y_train)
    print_cv_results("Random Forest", rf_cv)

    # 6. Refit on full training set
    print("[Fitting final models on full training set …]")
    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    # 7. Persist
    os.makedirs(output_dir, exist_ok=True)
    lr_path = os.path.join(output_dir, "logistic_regression.pkl")
    rf_path = os.path.join(output_dir, "random_forest.pkl")

    with open(lr_path, "wb") as f:
        pickle.dump(lr_pipeline, f)
    with open(rf_path, "wb") as f:
        pickle.dump(rf_pipeline, f)

    print(f"Models saved → {lr_path}")
    print(f"             → {rf_path}")

    return {
        "lr_pipeline": lr_pipeline,
        "rf_pipeline": rf_pipeline,
        "lr_cv_results": lr_cv,
        "rf_cv_results": rf_cv,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Area Feasibility Scoring Model."
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to area price CSV. Omit to use synthetic data.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=300_000,
        help="User budget in £ (default: 300000).",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Force synthetic data generation.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save trained models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    artefacts = train(
        data_path=args.data,
        user_budget=args.budget,
        use_synthetic=args.use_synthetic,
        output_dir=args.output,
    )
    print("\nTraining complete.")
