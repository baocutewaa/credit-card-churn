"""
Inference and submission generation module.
Implements Phase 9 & Phase 13 from Classification_Plan.md:
Applies 5-Fold Bagging Ensemble, validates output contract, and generates submission.csv.
"""

from typing import List, Optional
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    SUBMISSIONS_DIR,
    SAMPLE_SUBMISSION_PATH,
    FINAL_SUBMISSION_PATH,
    ID_COL,
    TARGET_COL,
    N_SPLITS,
)
from src.preprocessing import clean_raw_dataframe
from src.features import generate_all_features
from src.data_validation import validate_submission


def generate_ensemble_predictions(
    test_df: pd.DataFrame,
    model_type: str = "logistic_regression",
    n_splits: int = N_SPLITS,
    models_dir: Path = MODELS_DIR
) -> np.ndarray:
    """
    Generate predictions for test set using 5-Fold Bagging Ensemble.
    Averages predicted probabilities across all fold models.
    """
    fold_preds = []

    # Clean and create features for test set using identical logic
    print("Preprocessing and engineering features for test data...")
    cleaned_test = clean_raw_dataframe(test_df)
    feat_test = generate_all_features(cleaned_test)

    for fold in range(1, n_splits + 1):
        artifact_path = models_dir / f"{model_type}_fold_{fold}.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {artifact_path}. "
                f"Run 'python src/train.py --model {model_type}' first."
            )

        artifact = joblib.load(artifact_path)
        preprocessor = artifact["preprocessor"]
        model = artifact["model"]

        # Transform using fold-specific preprocessor
        X_test_trans = preprocessor.transform(feat_test)

        # Predict probability for churn class 1
        probs = model.predict_proba(X_test_trans)[:, 1]
        fold_preds.append(probs)
        print(f"Computed predictions for fold {fold}/{n_splits} (mean prob: {probs.mean():.4f})")

    # Average probabilities across 5 folds
    ensemble_prob = np.mean(fold_preds, axis=0)
    return ensemble_prob


def create_submission(
    test_csv_path: Path,
    model_type: str = "logistic_regression",
    output_path: Path = FINAL_SUBMISSION_PATH,
    template_path: Path = SAMPLE_SUBMISSION_PATH
) -> Path:
    """
    Main prediction pipeline: loads test set, runs ensemble inference,
    validates submission contract, and writes submission.csv.
    """
    if not test_csv_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found at: {test_csv_path}. "
            "Please ensure test.csv has been downloaded from Kaggle into data/raw/."
        )

    print(f"Loading test dataset from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)

    # Compute predictions
    test_probs = generate_ensemble_predictions(test_df, model_type=model_type)

    # Format submission dataframe
    sub_df = pd.DataFrame({
        ID_COL: test_df[ID_COL].values,
        TARGET_COL: np.round(test_probs, 5)
    })

    # Validate against template if exists
    template_df = pd.read_csv(template_path) if template_path.exists() else None
    is_valid, errors = validate_submission(sub_df, expected_rows=len(test_df), template_df=template_df)

    if not is_valid:
        raise ValueError(f"Submission contract validation failed: {errors}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Valid submission generated with {len(sub_df)} rows at: {output_path}")
    print(f"Submission stats: Min={test_probs.min():.4f}, Max={test_probs.max():.4f}, Mean={test_probs.mean():.4f}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission predictions using 5-Fold Bagging Ensemble.")
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/raw/test.csv"),
        help="Path to raw test.csv",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "lightgbm", "catboost"],
        help="Model architecture used for inference",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FINAL_SUBMISSION_PATH,
        help="Path to save submission CSV",
    )
    args = parser.parse_args()
    create_submission(test_csv_path=args.test_path, model_type=args.model, output_path=args.output)
