"""
Configuration and constants for Credit Card Churn Prediction project.
Based on Classification_Plan.md.
"""

from pathlib import Path

# Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# File Paths
TRAIN_RAW_PATH = DATA_RAW_DIR / "train.csv"
TEST_RAW_PATH = DATA_RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_RAW_DIR / "sample_submission.csv"
PROCESSED_TRAIN_PATH = DATA_PROCESSED_DIR / "train_cleaned.csv"
PROCESSED_TEST_PATH = DATA_PROCESSED_DIR / "test_cleaned.csv"
FINAL_SUBMISSION_PATH = SUBMISSIONS_DIR / "submission.csv"

# Project Constants
RANDOM_SEED = 42
SNAPSHOT_DATE = "2026-06-30"  # Business snapshot date
TARGET_COL = "churn_3m"
ID_COL = "customer_id"

# Validation Strategy
N_SPLITS = 5
SHUFFLE = True
PRIMARY_METRIC = "roc_auc"

# Features Definition
ID_COLS = ["customer_id"]
TARGET_COLS = ["churn_3m"]

NUMERIC_FEATURES = [
    "age",
    "income_month",
    "credit_limit",
    "annual_fee_waived",
    "utilization_6m",
    "txn_count_3m",
    "txn_amount_3m",
    "spend_change_pct",
    "app_logins_30d",
    "complaints_12m",
    "late_payments_12m",
    "other_bank_cards",
    "promo_ended_60d",
]

CATEGORICAL_FEATURES = [
    "gender",
    "province",
    "occupation",
    "card_type",
    "salary_via_bank",
    "last_contact_channel",
]

DATE_FEATURES = ["open_date"]

# Columns with constant or single-value variance to drop
LOW_VARIANCE_COLS = ["bank_code", "currency"]

# High missingness features to add explicit indicator
MISSING_INDICATOR_COLS = [
    "other_bank_cards",
    "income_month",
    "app_logins_30d",
    "salary_via_bank",
    "occupation",
    "last_contact_channel",
    "age",
]

# Categorical Standardization Mappings
GENDER_MAP = {
    "f": "Female",
    "female": "Female",
    "nữ": "Female",
    "nu": "Female",
    "m": "Male",
    "male": "Male",
    "nam": "Male",
}

SALARY_VIA_BANK_MAP = {
    "y": "Y",
    "yes": "Y",
    "có": "Y",
    "1": "Y",
    "n": "N",
    "no": "N",
    "không": "N",
    "khong": "N",
    "0": "N",
}

OCCUPATION_MAP = {
    "salaried": "salaried",
    "self employed": "self_employed",
    "self-employed": "self_employed",
    "civil servant": "civil_servant",
    "civil_servant": "civil_servant",
    "retired": "retired",
    "student": "student",
    "other": "other",
}

LAST_CONTACT_CHANNEL_MAP = {
    "app": "app",
    "branch": "branch",
    "hotline": "hotline",
    "none": "none",
}
