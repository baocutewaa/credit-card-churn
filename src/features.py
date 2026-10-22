"""
Feature engineering module.
Implements domain-specific features and transformations from Phase 3 of Classification_Plan.md.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np

from src.config import SNAPSHOT_DATE, MISSING_INDICATOR_COLS


def compute_tenure_features(df: pd.DataFrame, snapshot_date: str = SNAPSHOT_DATE) -> pd.DataFrame:
    """
    Compute customer tenure features relative to the business snapshot date (2026-06-30).
    Ensures no data leakage past snapshot_date.
    """
    df_feat = df.copy()
    snapshot_dt = pd.to_datetime(snapshot_date)

    if "open_date_dt" in df_feat.columns:
        # Check for open dates after snapshot
        future_mask = df_feat["open_date_dt"] > snapshot_dt
        if future_mask.any():
            # Mark abnormal future dates as NaN
            df_feat.loc[future_mask, "open_date_dt"] = pd.NaT

        # Tenure in days and months
        tenure_days = (snapshot_dt - df_feat["open_date_dt"]).dt.days
        df_feat["tenure_days"] = tenure_days
        df_feat["tenure_months"] = (tenure_days / 30.4375).round(1)
        df_feat["tenure_years"] = (tenure_days / 365.25).round(1)
    return df_feat


def add_missing_indicators(df: pd.DataFrame, columns: List[str] = MISSING_INDICATOR_COLS) -> pd.DataFrame:
    """
    Add binary indicator columns for features with significant missingness
    so tree models / linear models can learn missingness signals.
    """
    df_feat = df.copy()
    for col in columns:
        if col in df_feat.columns:
            df_feat[f"{col}_isna"] = df_feat[col].isna().astype(int)
    return df_feat


def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute transaction, credit limit, and spending change ratios.
    """
    df_feat = df.copy()

    # Average transaction amount per transaction
    if "txn_amount_3m" in df_feat.columns and "txn_count_3m" in df_feat.columns:
        # Avoid division by zero
        df_feat["avg_txn_amount_3m"] = df_feat["txn_amount_3m"] / (df_feat["txn_count_3m"].clip(lower=0) + 1)

    # Monthly transactions frequency (approx 3 months)
    if "txn_count_3m" in df_feat.columns:
        df_feat["monthly_txn_count"] = (df_feat["txn_count_3m"] / 3.0).round(2)

    # Utilization capping / clean feature
    if "utilization_6m" in df_feat.columns:
        df_feat["utilization_clipped"] = df_feat["utilization_6m"].clip(lower=0.0, upper=2.0)

    # Log transforms for skewed monetary values (ensuring non-negative inputs)
    for col in ["income_month", "credit_limit", "txn_amount_3m"]:
        if col in df_feat.columns:
            df_feat[f"{col}_log1p"] = np.log1p(np.maximum(df_feat[col].fillna(0), 0))

    return df_feat


def compute_behavioral_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute customer dissatisfaction and churn risk signals.
    """
    df_feat = df.copy()

    # Dissatisfaction flag
    if "complaints_12m" in df_feat.columns:
        df_feat["has_complaints"] = (df_feat["complaints_12m"] > 0).astype(int)

    # Payment risk flag
    if "late_payments_12m" in df_feat.columns:
        df_feat["has_late_payments"] = (df_feat["late_payments_12m"] > 0).astype(int)

    # Spend drop indicator (drop in spending velocity)
    if "spend_change_pct" in df_feat.columns:
        df_feat["is_spending_declining"] = (df_feat["spend_change_pct"] < 0).astype(int)

    # Domain interaction hypothesis: Spending dropped AND promo ended
    if "spend_change_pct" in df_feat.columns and "promo_ended_60d" in df_feat.columns:
        df_feat["spend_drop_x_promo_ended"] = (
            (df_feat["spend_change_pct"] < 0) & (df_feat["promo_ended_60d"] == 1)
        ).astype(int)

    # Interaction: High utilization and late payments
    if "utilization_6m" in df_feat.columns and "late_payments_12m" in df_feat.columns:
        df_feat["utilization_x_late_payments"] = (
            df_feat["utilization_6m"].fillna(0).clip(lower=0) * df_feat["late_payments_12m"].fillna(0)
        )

    return df_feat


def generate_all_features(df: pd.DataFrame, snapshot_date: str = SNAPSHOT_DATE) -> pd.DataFrame:
    """
    Master feature pipeline: applies tenure, missingness indicators,
    financial ratios, and behavioral signals.
    """
    df_out = compute_tenure_features(df, snapshot_date=snapshot_date)
    df_out = add_missing_indicators(df_out)
    df_out = compute_financial_ratios(df_out)
    df_out = compute_behavioral_signals(df_out)
    return df_out
