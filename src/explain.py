"""
Model interpretation and error analysis module.
Implements Phase 8 from Classification_Plan.md:
Feature importance extraction, coefficients analysis, and error profiling (FP/FN).
"""

from typing import Dict, List, Tuple
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    DATA_PROCESSED_DIR,
    ID_COL,
)


def get_logistic_regression_coefficients(fold_artifact_path: Path) -> pd.DataFrame:
    """
    Extract and format Logistic Regression feature coefficients from a saved fold model.
    """
    artifact = joblib.load(fold_artifact_path)
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values(by="abs_coefficient", ascending=False)

    return coef_df


def analyze_prediction_errors(
    oof_csv_path: Path,
    top_n: int = 50
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze OOF errors:
    1. False Negatives: Churned (actual=1) but lowest predicted churn probabilities.
    2. False Positives: Not churned (actual=0) but highest predicted churn probabilities.
    """
    oof_df = pd.read_csv(oof_csv_path)

    # False Negatives
    churners = oof_df[oof_df["actual"] == 1]
    top_fn = churners.sort_values(by="oof_pred_prob", ascending=True).head(top_n)

    # False Positives
    non_churners = oof_df[oof_df["actual"] == 0]
    top_fp = non_churners.sort_values(by="oof_pred_prob", ascending=False).head(top_n)

    return top_fn, top_fp


def save_importance_report(df: pd.DataFrame, report_name: str) -> Path:
    """
    Save feature importance dataframe to reports/metrics/.
    """
    metrics_dir = REPORTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"{report_name}.csv"
    df.to_csv(out_path, index=False)
    return out_path
