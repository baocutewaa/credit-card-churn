"""
Preprocessing module: text cleaning, alias mapping, date parsing,
and scikit-learn ColumnTransformer / Pipeline builders.
Based on Phase 1 & Phase 5 in Classification_Plan.md.
"""

from typing import List, Optional
import re
import unicodedata
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import (
    GENDER_MAP,
    SALARY_VIA_BANK_MAP,
    OCCUPATION_MAP,
    LAST_CONTACT_CHANNEL_MAP,
    SNAPSHOT_DATE,
)


def normalize_text(text: Optional[str]) -> str:
    """Normalize text: strip, lowercase, normalize unicode spaces."""
    if pd.isna(text):
        return ""
    text_str = str(text).strip()
    text_str = unicodedata.normalize("NFKC", text_str)
    return text_str


def clean_gender(val: any) -> str:
    """Standardize gender values."""
    if pd.isna(val):
        return "Unknown"
    norm = normalize_text(val).lower()
    return GENDER_MAP.get(norm, norm.title() if norm else "Unknown")


def clean_salary_via_bank(val: any) -> str:
    """Standardize salary_via_bank values to 'Y', 'N' or 'Unknown'."""
    if pd.isna(val):
        return "Unknown"
    norm = normalize_text(val).lower()
    return SALARY_VIA_BANK_MAP.get(norm, "Unknown")


def clean_province(val: any) -> str:
    """Standardize province names and handle common aliases (TP.HCM, Hà Nội, etc.)."""
    if pd.isna(val):
        return "Unknown"
    norm = normalize_text(val)
    lower = norm.lower()

    if re.search(r"\b(hcm|hồ chí minh|tp\.hcm|tp hcm|ho chi minh)\b", lower):
        return "TP.HCM"
    if re.search(r"\b(hn|hà nội|ha noi)\b", lower):
        return "Hà Nội"
    if re.search(r"\b(đà nẵng|da nang)\b", lower):
        return "Đà Nẵng"
    if re.search(r"\b(hải phòng|hai phong)\b", lower):
        return "Hải Phòng"
    if re.search(r"\b(cần thơ|can tho)\b", lower):
        return "Cần Thơ"
    if re.search(r"\b(bình dương|binh duong)\b", lower):
        return "Bình Dương"
    if re.search(r"\b(đồng nai|dong nai)\b", lower):
        return "Đồng Nai"

    return norm.title() if norm else "Unknown"


def clean_occupation(val: any) -> str:
    """Standardize occupation labels."""
    if pd.isna(val):
        return "Unknown"
    norm = normalize_text(val).lower()
    # Replace multiple spaces or dashes
    norm = re.sub(r"[\s\-]+", " ", norm)
    return OCCUPATION_MAP.get(norm, norm)


def clean_last_contact_channel(val: any) -> str:
    """
    Standardize contact channel.
    Note: 'none' is a meaningful valid channel (no contact), not missing!
    """
    if pd.isna(val):
        return "Unknown"
    norm = normalize_text(val).lower()
    return LAST_CONTACT_CHANNEL_MAP.get(norm, norm)


def parse_open_date(series: pd.Series) -> pd.Series:
    """
    Parse open_date with explicit multiple format matching:
    %Y-%m-%d, %Y%m%d, %d/%m/%Y.
    """
    # Try parsing format 1: %Y-%m-%d
    parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")

    # For remaining NaTs, try %Y%m%d
    mask = parsed.isna() & series.notna()
    if mask.any():
        parsed[mask] = pd.to_datetime(series[mask], format="%Y%m%d", errors="coerce")

    # For remaining NaTs, try %d/%m/%Y
    mask = parsed.isna() & series.notna()
    if mask.any():
        parsed[mask] = pd.to_datetime(series[mask], format="%d/%m/%Y", errors="coerce")

    # Generic fallback
    mask = parsed.isna() & series.notna()
    if mask.any():
        parsed[mask] = pd.to_datetime(series[mask], errors="coerce")

    return parsed


def standardize_monetary_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize income_month and credit_limit values:
    Values > 1000 were recorded in raw VND rather than million VND.
    Dividing by 1e6 aligns them with the majority of the dataset.
    """
    df_out = df.copy()
    if "income_month" in df_out.columns:
        high_mask = df_out["income_month"] > 1000
        df_out.loc[high_mask, "income_month"] = df_out.loc[high_mask, "income_month"] / 1e6
    if "credit_limit" in df_out.columns:
        high_mask = df_out["credit_limit"] > 1000
        df_out.loc[high_mask, "credit_limit"] = df_out.loc[high_mask, "credit_limit"] / 1e6
    return df_out


def clean_raw_dataframe(df: pd.DataFrame, fix_unit_inconsistencies: bool = True) -> pd.DataFrame:
    """
    Apply initial data cleaning and sentinel replacement before feature engineering.
    """
    df_clean = df.copy()

    # Clean categorical text columns
    if "gender" in df_clean.columns:
        df_clean["gender"] = df_clean["gender"].apply(clean_gender)
    if "province" in df_clean.columns:
        df_clean["province"] = df_clean["province"].apply(clean_province)
    if "occupation" in df_clean.columns:
        df_clean["occupation"] = df_clean["occupation"].apply(clean_occupation)
    if "salary_via_bank" in df_clean.columns:
        df_clean["salary_via_bank"] = df_clean["salary_via_bank"].apply(clean_salary_via_bank)
    if "last_contact_channel" in df_clean.columns:
        df_clean["last_contact_channel"] = df_clean["last_contact_channel"].apply(clean_last_contact_channel)

    # Sentinel handling for age: 999 is sentinel error, < 18 or > 120 are abnormal
    if "age" in df_clean.columns:
        df_clean.loc[df_clean["age"] > 120, "age"] = np.nan
        df_clean.loc[df_clean["age"] < 18, "age"] = np.nan

    # Standardize monetary unit discrepancies
    if fix_unit_inconsistencies:
        df_clean = standardize_monetary_units(df_clean)

    # Parse open_date
    if "open_date" in df_clean.columns:
        df_clean["open_date_dt"] = parse_open_date(df_clean["open_date"])

    return df_clean


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Build a leak-free scikit-learn ColumnTransformer.
    - Numeric: Median Imputation + Optional StandardScaler
    - Categorical: Impute 'Unknown' + OneHotEncoder(handle_unknown='ignore')
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor
