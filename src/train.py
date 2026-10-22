"""
Model training orchestrator with Stratified 5-Fold Cross-Validation.
Implements Phase 4 & Phase 5 training pipelines from Classification_Plan.md.
"""

from typing import Dict, Any, Tuple, List, Optional
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from src.config import (
    TRAIN_RAW_PATH,
    MODELS_DIR,
    METRICS_DIR,
    DATA_PROCESSED_DIR,
    TARGET_COL,
    ID_COL,
    RANDOM_SEED,
    N_SPLITS,
    SHUFFLE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.data_validation import check_and_remove_duplicates, validate_target
from src.preprocessing import clean_raw_dataframe, build_preprocessor
from src.features import generate_all_features
from src.evaluate import compute_metrics, summarize_cv_metrics, save_metrics_report


def get_model(model_type: str, random_state: int = RANDOM_SEED):
    """
    Factory function for candidate models.
    """
    if model_type == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbose=-1,
            )
        except ImportError:
            raise ImportError("lightgbm is not installed. Please install it via requirements.txt.")
    elif model_type == "catboost":
        try:
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=500,
                learning_rate=0.03,
                depth=6,
                random_seed=random_state,
                verbose=0,
            )
        except ImportError:
            raise ImportError("catboost is not installed. Please install it via requirements.txt.")
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'logistic_regression', 'lightgbm', or 'catboost'.")


def prepare_training_data(raw_train_path: Path = TRAIN_RAW_PATH) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Load raw train data, deduplicate, clean, and generate features.
    """
    print(f"Loading raw train data from: {raw_train_path}")
    raw_df = pd.read_csv(raw_train_path)

    # 1. Deduplication
    clean_df, dedup_stats = check_and_remove_duplicates(raw_df)
    print(f"Deduplication complete: removed {dedup_stats['duplicate_rows_removed']} duplicates, "
          f"remaining {dedup_stats['clean_rows']} unique rows.")

    # 2. Target validation
    target_info = validate_target(clean_df)
    print(f"Target Churn Rate: {target_info['churn_rate']:.3%}")

    # 3. Clean raw dataframe & parse dates
    cleaned_df = clean_raw_dataframe(clean_df)

    # 4. Feature engineering
    feat_df = generate_all_features(cleaned_df)

    # Separate target and features
    y = feat_df[TARGET_COL].astype(int)

    # Determine numeric and categorical feature columns present
    cols_to_exclude = {ID_COL, TARGET_COL, "open_date", "open_date_dt", "bank_code", "currency"}
    candidate_cols = [c for c in feat_df.columns if c not in cols_to_exclude]

    num_cols = [c for c in candidate_cols if feat_df[c].dtype in ["int64", "float64", "int32", "float32"]]
    cat_cols = [c for c in candidate_cols if c not in num_cols]

    return feat_df, y, num_cols, cat_cols


def run_cross_validation(
    model_type: str = "logistic_regression",
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Execute Stratified 5-Fold Cross-Validation, strictly fitting preprocessing
    inside each fold to prevent data leakage.
    """
    df, y, num_cols, cat_cols = prepare_training_data()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=SHUFFLE, random_state=random_state)

    oof_preds = np.zeros(len(df))
    fold_metrics = []
    models_saved = []

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Starting Stratified {n_splits}-Fold CV for model: '{model_type}' ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        X_train_fold = df.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = df.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Fit preprocessor on training fold ONLY
        scale_num = (model_type == "logistic_regression")
        preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=scale_num)

        X_train_trans = preprocessor.fit_transform(X_train_fold)
        X_val_trans = preprocessor.transform(X_val_fold)

        # Train model
        model = get_model(model_type, random_state=random_state + fold)
        model.fit(X_train_trans, y_train_fold)

        # Predict validation probabilities for churn class 1
        val_probs = model.predict_proba(X_val_trans)[:, 1]
        oof_preds[val_idx] = val_probs

        # Compute fold metrics
        metrics = compute_metrics(y_val_fold.values, val_probs)
        fold_metrics.append(metrics)
        print(f"Fold {fold + 1}/{n_splits} - ROC AUC: {metrics['roc_auc']:.4f} | PR AUC: {metrics['pr_auc']:.4f}")

        # Save fold artifacts
        fold_artifact = {
            "preprocessor": preprocessor,
            "model": model,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        }
        artifact_path = MODELS_DIR / f"{model_type}_fold_{fold + 1}.joblib"
        joblib.dump(fold_artifact, artifact_path)
        models_saved.append(str(artifact_path))

    # Calculate overall OOF metrics
    overall_oof_metrics = compute_metrics(y.values, oof_preds)
    summary = summarize_cv_metrics(fold_metrics)
    summary["overall_oof_roc_auc"] = overall_oof_metrics["roc_auc"]
    summary["overall_oof_pr_auc"] = overall_oof_metrics["pr_auc"]
    summary["model_type"] = model_type
    summary["saved_models"] = models_saved

    print("\n=== Validation Results ===")
    print(f"Mean ROC AUC across {n_splits} folds: {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}")
    print(f"Overall OOF ROC AUC: {summary['overall_oof_roc_auc']:.4f}")
    print(f"Overall OOF PR AUC:  {summary['overall_oof_pr_auc']:.4f}")

    # Save summary report & OOF predictions
    save_metrics_report(model_type, summary)

    oof_df = pd.DataFrame({
        ID_COL: df[ID_COL].values,
        "actual": y.values,
        "oof_pred_prob": oof_preds
    })
    oof_path = DATA_PROCESSED_DIR / f"oof_{model_type}.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved OOF predictions to: {oof_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit card churn models with 5-Fold CV.")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "lightgbm", "catboost"],
        help="Model architecture to train",
    )
    args = parser.parse_args()
    run_cross_validation(model_type=args.model)
