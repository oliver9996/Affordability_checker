"""
evaluate.py
-----------
Evaluation utilities for the Area Feasibility Scoring Model.

Provides:
  - evaluate_classifier   : full suite of classification metrics
  - compare_models        : side-by-side comparison table
  - get_feature_importances: extract importances from tree / linear models
  - print_* helpers       : human-readable console output
  - plot_* helpers        : matplotlib figures (saved to file or shown)

Usage (CLI)
----------
    python src/evaluate.py --models models/ --use-synthetic --budget 300000
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import prepare_dataset
from features import (
    CLASSIFICATION_TARGET,
    FEATURE_COLS,
    LeakageSafeFeaturizer,
    assert_no_leakage,
    build_target,
)

# ---------------------------------------------------------------------------
# Classifier evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """Compute a full suite of classification metrics.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        A *fitted* pipeline whose final step is a classifier.
    X_test : pd.DataFrame
        Raw (pre-featurization) test data.
    y_test : pd.Series
        True binary labels.
    model_name : str
        Human-readable name used in output.

    Returns
    -------
    dict with keys: model_name, accuracy, precision, recall, f1,
                    roc_auc, classification_report, confusion_matrix.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Not Affordable", "Affordable"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def print_classifier_metrics(metrics: dict):
    """Pretty-print metrics returned by ``evaluate_classifier``."""
    name = metrics["model_name"]
    print(f"\n{'=' * 60}")
    print(f"  {name} – Test-Set Evaluation")
    print(f"{'=' * 60}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print()
    print(metrics["classification_report"])
    print("Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    labels = ["Not Affordable", "Affordable"]
    print(f"  {'':18s}  Pred:Not  Pred:Yes")
    for i, label in enumerate(labels):
        print(f"  Actual {label:<12s} {cm[i, 0]:8d}  {cm[i, 1]:8d}")
    print()


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(metrics_list: list) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of metric dicts.

    Parameters
    ----------
    metrics_list : list of dict
        Each element is a dict returned by ``evaluate_classifier``.

    Returns
    -------
    pd.DataFrame
        One row per model, columns: model_name, accuracy, precision,
        recall, f1, roc_auc.
    """
    rows = [
        {
            "model": m["model_name"],
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "roc_auc": round(m["roc_auc"], 4),
        }
        for m in metrics_list
    ]
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------

def get_feature_importances(pipeline, feature_names: list = None) -> pd.DataFrame:
    """Extract feature importances from the final estimator.

    Works for:
    - ``RandomForestClassifier``  (``feature_importances_`` attribute)
    - ``LogisticRegression``      (``coef_`` attribute)

    Parameters
    ----------
    pipeline : sklearn Pipeline
    feature_names : list of str, optional
        If None, uses ``FEATURE_COLS``.

    Returns
    -------
    pd.DataFrame
        Columns ``feature`` and ``importance``, sorted descending.
    """
    if feature_names is None:
        feature_names = FEATURE_COLS

    estimator = pipeline.steps[-1][1]  # last Pipeline step

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
    else:
        raise AttributeError(
            f"{type(estimator).__name__} has neither feature_importances_ "
            "nor coef_."
        )

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Plot helpers (optional – require matplotlib)
# ---------------------------------------------------------------------------

def plot_feature_importances(importances_df: pd.DataFrame, title: str = "Feature Importances", save_path: str = None):
    """Bar chart of feature importances."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        importances_df["feature"][::-1],
        importances_df["importance"][::-1],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curves(pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None):
    """Overlay ROC curves for multiple pipelines.

    Parameters
    ----------
    pipelines : dict
        ``{model_name: fitted_pipeline}``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipeline in pipelines.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, name=name, ax=ax)

    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set_title("ROC Curves – Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Area Feasibility models."
    )
    parser.add_argument(
        "--models",
        default=os.path.join(os.path.dirname(__file__), "..", "models"),
        help="Directory containing trained model pickle files.",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Load / generate data
    df = prepare_dataset(
        path=args.data,
        user_budget=args.budget,
        use_synthetic=args.use_synthetic,
    )
    y = build_target(df)
    assert_no_leakage(FEATURE_COLS, [CLASSIFICATION_TARGET])

    # Use last 20 % as test set (must match split in train.py)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        df, y, test_size=0.20, stratify=y, random_state=42
    )

    # Load models
    lr_path = os.path.join(args.models, "logistic_regression.pkl")
    rf_path = os.path.join(args.models, "random_forest.pkl")

    if not os.path.exists(lr_path) or not os.path.exists(rf_path):
        print("Trained models not found. Run train.py first.")
        sys.exit(1)

    with open(lr_path, "rb") as f:
        lr_pipeline = pickle.load(f)
    with open(rf_path, "rb") as f:
        rf_pipeline = pickle.load(f)

    # Evaluate
    lr_metrics = evaluate_classifier(lr_pipeline, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluate_classifier(rf_pipeline, X_test, y_test, "Random Forest")

    print_classifier_metrics(lr_metrics)
    print_classifier_metrics(rf_metrics)

    print("\nModel Comparison Table:")
    print(compare_models([lr_metrics, rf_metrics]).to_string())

    print("\nLogistic Regression – Feature Importances (|coef|):")
    print(get_feature_importances(lr_pipeline).to_string(index=False))

    print("\nRandom Forest – Feature Importances:")
    print(get_feature_importances(rf_pipeline).to_string(index=False))
