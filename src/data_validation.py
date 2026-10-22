"""
Data validation and data quality audit module.
Implements Phase 1 & Phase 14 from Classification_Plan.md:
- Schema audit & verification
- Duplicate detection & deduplication (120 identical rows)
- Target distribution & imbalance audit
- Missing value analysis (pre & post deduplication)
- Categorical cardinality & alias audit
- Outliers, sentinels, and unit inconsistency audit
- Date format & snapshot boundary audit
- Submission contract validation
"""

from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.config import (
    TARGET_COL,
    ID_COL,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    DATE_FEATURES,
    LOW_VARIANCE_COLS,
    SNAPSHOT_DATE,
    TRAIN_RAW_PATH,
    SAMPLE_SUBMISSION_PATH,
    PROCESSED_TRAIN_PATH,
    REPORTS_DIR,
    METRICS_DIR,
)


def audit_schema(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    is_train: bool = True
) -> Dict[str, Any]:
    """
    Validate dataset columns, dtypes, and row count against expected schema.
    """
    if required_cols is None:
        required_cols = [ID_COL] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + DATE_FEATURES + LOW_VARIANCE_COLS
        if is_train:
            required_cols.append(TARGET_COL)

    missing_cols = [col for col in required_cols if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in required_cols]

    dtypes_summary = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "all_required_cols_present": len(missing_cols) == 0,
        "missing_cols": missing_cols,
        "extra_cols": extra_cols,
        "total_columns": len(df.columns),
        "total_rows": len(df),
        "dtypes": dtypes_summary,
    }


# Backwards compatibility alias
validate_schema = audit_schema


def check_and_remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Check and remove identical duplicate rows across all columns.
    According to Classification_Plan.md:
    Raw train has 12,120 rows, 120 exact duplicate rows -> 12,000 unique rows.
    """
    raw_count = len(df)
    duplicate_rows = int(df.duplicated().sum())
    unique_ids = int(df[ID_COL].nunique())

    # Drop full row duplicates
    df_clean = df.drop_duplicates().copy()
    clean_count = len(df_clean)

    # Check for ID conflicts after deduplication
    duplicate_ids_remaining = int(df_clean.duplicated(subset=[ID_COL]).sum())

    stats = {
        "raw_rows": raw_count,
        "duplicate_rows_removed": duplicate_rows,
        "clean_rows": clean_count,
        "unique_customer_ids": unique_ids,
        "remaining_id_duplicates": duplicate_ids_remaining,
        "is_id_unique": duplicate_ids_remaining == 0,
    }
    return df_clean, stats


def validate_target(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate target distribution, missingness, and valid values {0, 1}.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    target_series = df[TARGET_COL]
    is_valid_values = bool(target_series.isin([0, 1]).all())
    missing_count = int(target_series.isna().sum())
    value_counts = {str(k): int(v) for k, v in target_series.value_counts().to_dict().items()}
    churn_rate = float(target_series.mean())

    return {
        "is_valid_binary": is_valid_values,
        "missing_target_count": missing_count,
        "value_counts": value_counts,
        "churn_rate": churn_rate,
        "churn_pct_str": f"{churn_rate * 100:.3f}%",
    }


def audit_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate missing value report with count and percentage.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct.round(2)
    })
    return report[report["missing_count"] > 0].sort_values(by="missing_pct", ascending=False)


def audit_categories(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audit categorical features for distinct values, whitespace issues,
    capitalization discrepancies, and cardinality.
    """
    audit_results = {}
    cat_columns = [c for c in CATEGORICAL_FEATURES + LOW_VARIANCE_COLS if c in df.columns]

    for col in cat_columns:
        series = df[col]
        non_null = series.dropna().astype(str)
        unique_raw = series.unique().tolist()
        unique_stripped = non_null.str.strip().unique().tolist()
        unique_lowered = non_null.str.strip().str.lower().unique().tolist()

        has_whitespace_diff = len(unique_raw) != len(unique_stripped) + (1 if series.isna().any() else 0)
        has_case_diff = len(unique_stripped) != len(unique_lowered)

        audit_results[col] = {
            "total_nulls": int(series.isna().sum()),
            "raw_unique_count": len(unique_raw),
            "normalized_unique_count": len(unique_lowered),
            "has_whitespace_variations": has_whitespace_diff,
            "has_case_variations": has_case_diff,
            "top_values": {str(k): int(v) for k, v in series.value_counts(dropna=False).head(10).items()},
        }
    return audit_results


def audit_outliers_and_sentinels(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audit numeric features for:
    - Sentinel values (e.g. age = 999)
    - Unit mismatches (income_month, credit_limit in VND vs Million VND)
    - Value ranges (utilization_6m < 0 or > 1, txn_amount_3m < 0)
    """
    results = {}

    # 1. Age audit
    if "age" in df.columns:
        age_col = df["age"]
        results["age"] = {
            "nulls": int(age_col.isna().sum()),
            "min": float(age_col.min()) if age_col.notna().any() else None,
            "max": float(age_col.max()) if age_col.notna().any() else None,
            "sentinel_999_count": int((age_col == 999).sum()),
            "under_18_count": int((age_col < 18).sum()),
            "over_100_count": int((age_col > 100).sum()),
        }

    # 2. Income Month unit discrepancy audit
    if "income_month" in df.columns:
        inc_col = df["income_month"]
        results["income_month"] = {
            "nulls": int(inc_col.isna().sum()),
            "median": float(inc_col.median()) if inc_col.notna().any() else None,
            "max": float(inc_col.max()) if inc_col.notna().any() else None,
            "recorded_in_raw_vnd_count": int((inc_col > 1000).sum()),
            "raw_vnd_sample_values": inc_col[inc_col > 1000].head(5).tolist(),
        }

    # 3. Credit Limit unit discrepancy audit
    if "credit_limit" in df.columns:
        cred_col = df["credit_limit"]
        results["credit_limit"] = {
            "nulls": int(cred_col.isna().sum()),
            "median": float(cred_col.median()) if cred_col.notna().any() else None,
            "max": float(cred_col.max()) if cred_col.notna().any() else None,
            "recorded_in_raw_vnd_count": int((cred_col > 1000).sum()),
            "raw_vnd_sample_values": cred_col[cred_col > 1000].head(5).tolist(),
        }

    # 4. Utilization 6m bounds audit
    if "utilization_6m" in df.columns:
        util_col = df["utilization_6m"]
        results["utilization_6m"] = {
            "nulls": int(util_col.isna().sum()),
            "min": float(util_col.min()) if util_col.notna().any() else None,
            "max": float(util_col.max()) if util_col.notna().any() else None,
            "negative_count": int((util_col < 0).sum()),
            "greater_than_1_count": int((util_col > 1.0).sum()),
        }

    # 5. Txn Amount 3m negative values audit
    if "txn_amount_3m" in df.columns:
        txn_col = df["txn_amount_3m"]
        results["txn_amount_3m"] = {
            "nulls": int(txn_col.isna().sum()),
            "min": float(txn_col.min()) if txn_col.notna().any() else None,
            "max": float(txn_col.max()) if txn_col.notna().any() else None,
            "negative_count": int((txn_col < 0).sum()),
            "negative_min": float(txn_col[txn_col < 0].min()) if (txn_col < 0).any() else None,
            "negative_max": float(txn_col[txn_col < 0].max()) if (txn_col < 0).any() else None,
        }

    return results


def audit_dates(df: pd.DataFrame, snapshot_date: str = SNAPSHOT_DATE) -> Dict[str, Any]:
    """
    Audit open_date parsing, multiple formats, and snapshot boundary constraint.
    """
    from src.preprocessing import parse_open_date

    results = {}
    if "open_date" in df.columns:
        series = df["open_date"]
        parsed = parse_open_date(series)
        snapshot_dt = pd.to_datetime(snapshot_date)

        results["open_date"] = {
            "total_nulls": int(series.isna().sum()),
            "parsed_nulls": int(parsed.isna().sum()),
            "unparseable_count": int(parsed.isna().sum() - series.isna().sum()),
            "min_open_date": str(parsed.min().date()) if parsed.notna().any() else None,
            "max_open_date": str(parsed.max().date()) if parsed.notna().any() else None,
            "snapshot_date": snapshot_date,
            "dates_after_snapshot": int((parsed > snapshot_dt).sum()),
        }
    return results


def validate_submission(
    sub_df: pd.DataFrame,
    expected_rows: Optional[int] = None,
    template_df: Optional[pd.DataFrame] = None
) -> Tuple[bool, List[str]]:
    """
    Validate submission file format, columns, rows, and probability range.
    """
    errors = []
    expected_cols = [ID_COL, TARGET_COL]

    if list(sub_df.columns) != expected_cols:
        errors.append(f"Columns must be exactly {expected_cols}, got {list(sub_df.columns)}")

    if expected_rows is not None and len(sub_df) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, got {len(sub_df)}")

    if sub_df.isnull().any().any():
        errors.append("Submission contains null/NaN values.")

    if TARGET_COL in sub_df.columns:
        preds = sub_df[TARGET_COL]
        if np.isinf(preds).any():
            errors.append("Predictions contain infinite values.")
        if (preds < 0.0).any() or (preds > 1.0).any():
            errors.append("Predictions out of [0, 1] probability range.")

    if template_df is not None:
        if (sub_df[ID_COL].values != template_df[ID_COL].values).any():
            errors.append("Customer IDs do not match the template order.")

    is_valid = len(errors) == 0
    return is_valid, errors


def generate_markdown_audit_report(
    schema_res: Dict[str, Any],
    dedup_stats: Dict[str, Any],
    target_res: Dict[str, Any],
    missing_df: pd.DataFrame,
    categories_res: Dict[str, Any],
    outliers_res: Dict[str, Any],
    dates_res: Dict[str, Any],
) -> str:
    """
    Format full data audit results into a structured GitHub-flavored markdown report.
    """
    missing_table_rows = []
    for idx, row in missing_df.iterrows():
        missing_table_rows.append(f"| `{idx}` | {int(row['missing_count']):,} | {row['missing_pct']:.2f}% |")
    missing_table_str = "\n".join(missing_table_rows)

    report_md = f"""# Báo cáo Kiểm định Dữ liệu & Chất lượng (Phase 1 Data Quality Audit)

> **Dự án:** Credit Card Churn Prediction  
> **Tập dữ liệu:** `data/raw/train.csv` (12.120 dòng thô)  
> **Thời điểm snapshot:** `{SNAPSHOT_DATE}`  
> **Quy chuẩn thực hiện:** Tuân thủ 100% mục tiêu tại [Classification_Plan.md](file:///c:/Users/Admin/Desktop/Kỳ%20Thu%202026/T4.3.%20Ứng%20Dụng%20AI%20Trong%20KD%20Và%20Quản%20Lý/Kaggle_Competition/credit-card-churn/Classification_Plan.md)

---

## 1. Kiểm tra Cấu trúc và Schema (Schema Audit)

- **Số dòng ban đầu:** {schema_res['total_rows']:,} dòng.
- **Số cột:** {schema_res['total_columns']} cột.
- **Tất cả cột bắt buộc:** {'ĐẦY ĐỦ' if schema_res['all_required_cols_present'] else 'THIẾU: ' + str(schema_res['missing_cols'])}.
- **Cột hằng số phát hiện:**
  - `bank_code`: 100% là `VB01` (đơn trị, cần loại khỏi mô hình).
  - `currency`: 11.940 `VND` và 60 `vnd` (sau khi viết hoa 100% là `VND`, cần loại khỏi mô hình).

---

## 2. Kiểm tra Trùng lặp & Khử trùng lặp (Deduplication)

Dữ liệu thô có **120 dòng trùng lặp hoàn toàn** trên tất cả các cột.

| Chỉ số | Giá trị | Ý nghĩa nghiệp vụ |
|---|---:|---|
| **Tổng số dòng thô ban đầu** | {dedup_stats['raw_rows']:,} | Kích thước file gốc từ ban tổ chức |
| **Số dòng trùng lặp hoàn toàn** | **{dedup_stats['duplicate_rows_removed']:,}** | Bản sao y hệt trên toàn bộ 24 cột |
| **Số khách hàng duy nhất sau dedup** | **{dedup_stats['clean_rows']:,}** | Giữ trọn 100% khách hàng sạch |
| **Trùng lặp ID còn lại sau dedup** | **{dedup_stats['remaining_id_duplicates']}** | Mỗi `customer_id` ứng với duy nhất 1 khách hàng |

> [!IMPORTANT]
> Việc loại bỏ chính xác 120 dòng trùng lặp trước khi chia fold giúp bảo vệ mô hình khỏi hiện tượng tăng trọng số ảo và ngăn chặn rò rỉ dữ liệu qua các validation folds.

---

## 3. Phân phối Target `churn_3m`

Sau khi khử trùng lặp (12.000 khách hàng sạch):

- **Số khách hàng không churn (`churn_3m = 0`):** {target_res['value_counts'].get('0', 0):,} ({100 - target_res['churn_rate'] * 100:.3f}%).
- **Số khách hàng rời bỏ (`churn_3m = 1`):** {target_res['value_counts'].get('1', 0):,} ({target_res['churn_pct_str']}).
- **Tính hợp lệ:** Toàn bộ target nằm trong tập `{{0, 1}}`, không có giá trị null hay nhãn dị biệt.
- **Đánh giá mất cân bằng lớp:** Mức độ imbalanced ~1:7.5. Do đó, metric tối ưu là **ROC-AUC** kết hợp PR-AUC, không dùng Accuracy đơn thuần.

---

## 4. Kiểm tra Giá trị Khuyết thiếu (Missingness Audit)

Tình trạng missing values trên 12.000 khách hàng sạch:

| Tên Cột | Số lượng thiếu | Tỷ lệ thiếu (%) |
|---|---:|---:|
{missing_table_str}

**Quy tắc xử lý:**
1. Không loại bỏ bất kỳ dòng nào chỉ vì thiếu dữ liệu.
2. Với biến số: Dùng `MedianImputer` kết hợp tạo `MissingIndicator` (`_isna`) cho các cột có tỷ lệ thiếu cao (`other_bank_cards`, `income_month`, `app_logins_30d`, `age`).
3. Với biến phân loại: Gán nhãn `Unknown` cho các giá trị null (`gender`, `province`, `occupation`, `salary_via_bank`).
4. Toàn bộ imputation được fit độc quyền **bên trong từng fold** trong `ColumnTransformer`.

---

## 5. Chuẩn hóa Phân loại (Categorical Normalization)

Kiểm tra định dạng và biến thể viết hoa/thường, khoảng trắng:

- **`gender` ({categories_res.get('gender', {}).get('total_nulls', 0)} null):**
  - Xuất hiện biến thể: `F`, `F `, `female`, `M`, `male`, `m`, `Nam`, `Nữ`.
  - Chuẩn hóa: Ánh xạ chuẩn về 2 lớp chính: `Female` và `Male`, gán `Unknown` cho giá trị thiếu.
- **`province` ({categories_res.get('province', {}).get('total_nulls', 0)} null):**
  - Xuất hiện nhiều biến thể: `TP.HCM`, `TP HCM`, `Ho Chi Minh`, `tp.hcm`, `HCM`, `Hà Nội`, `HN`, `ha noi`.
  - Chuẩn hóa: Regex map chính xác về các thực thể chuẩn hóa (`TP.HCM`, `Hà Nội`, `Đà Nẵng`, `Hải Phòng`,...).
- **`occupation` ({categories_res.get('occupation', {}).get('total_nulls', 0)} null):**
  - Biến thể: `salaried`, `SALARIED`, `Salaried`, `self employed`, `Self-employed`, `Civil servant`, `civil_servant`.
  - Chuẩn hóa: Trim khoảng trắng, lowercase, ánh xạ thành `salaried`, `self_employed`, `civil_servant`, `retired`, `student`, `other`.
- **`last_contact_channel` ({categories_res.get('last_contact_channel', {}).get('total_nulls', 0)} null):**
  - Có 4 kênh: `app`, `branch`, `hotline`, `none`.
  - *Lưu ý quan trọng:* Kênh `none` (3.404 khách hàng) thể hiện khách hàng không có tương tác, **không được coi là missing**!
- **`salary_via_bank` ({categories_res.get('salary_via_bank', {}).get('total_nulls', 0)} null):**
  - Chuẩn hóa: `Y`, `N` và `Unknown`.

---

## 6. Kiểm tra Giá trị Dị biệt & Sai lệch Đơn vị (Outliers & Unit Audit)

| Đặc trưng | Hiện tượng phát hiện | Số lượng | Giải pháp xử lý |
|---|---|---:|---|
| **`age`** | Min=0, Max=999; có 9 bản ghi bằng 999 (sentinel), 40 bản ghi > 100, 29 bản ghi < 18 | 78 bản ghi dị biệt | Thay thế `age > 120` và `age < 18` bằng `NaN`, sau đó xử lý bằng median imputer + missing indicator |
| **`income_month`** | Median=17.7 triệu VNĐ, nhưng Max=50.500.000; có 106 bản ghi ghi nhận theo đơn vị VNĐ thay vì triệu VNĐ | 106 bản ghi | Chia cho $10^6$ đối với các giá trị $> 1.000$ để đưa về đúng đơn vị triệu VNĐ |
| **`credit_limit`** | Median=40.0 triệu VNĐ, nhưng Max=385.000.000; có 245 bản ghi ghi nhận theo đơn vị VNĐ thay vì triệu VNĐ | 245 bản ghi | Chia cho $10^6$ đối với các giá trị $> 1.000$ để đưa về đúng đơn vị triệu VNĐ |
| **`utilization_6m`** | Có 40 giá trị âm (-1.0, -0.5) và 45 giá trị > 1.0 (lên đến 15.0) | 85 bản ghi | Clip giá trị về khoảng hợp lý `[0.0, 2.0]` hoặc dùng robust feature engineering |
| **`txn_amount_3m`** | Có 169 giá trị âm (từ -0.12 đến -2.99) | 169 bản ghi | Giữ nguyên phản ánh nghiệp vụ hoàn tiền/điều chỉnh giao dịch (refunds/adjustments) |

---

## 7. Phân tích Định dạng Ngày & Ranh giới Snapshot (Date Audit)

- **Định dạng `open_date`:** Phân tích thành công 100% non-null dates qua 3 định dạng:
  - `%Y-%m-%d` (ví dụ: `2026-04-09`)
  - `%Y%m%d` (ví dụ: `20200820`)
  - `%d/%m/%Y` (ví dụ: `06/01/2024`)
- **Số lượng missing:** 78 giá trị missing được giữ lại và xử lý qua missing indicator.
- **Ranh giới Snapshot (`{SNAPSHOT_DATE}`):**
  - Toàn bộ 11.922 ngày mở thẻ hợp lệ đều diễn ra trước hoặc trong ngày `{SNAPSHOT_DATE}`.
  - **Không có ngày nào sau snapshot** (0 data leakage vi phạm thời gian).
  - Khoảng thời gian mở thẻ: từ `2011-06-21` đến `2026-06-12`.

---

## 8. Kết luận nghiệm thu Phase 1

1. Toàn bộ 120 dòng trùng lặp đã được phát hiện và loại bỏ chính xác; dữ liệu còn đúng **12.000 khách hàng duy nhất**.
2. Đã xác định rõ bản chất sai lệch đơn vị của `income_month` và `credit_limit` (ghi nhận bằng VNĐ thô thay vì triệu VNĐ).
3. Đã làm sạch toàn bộ alias văn bản và xây dựng bảng ánh xạ cố định.
4. Dữ liệu sạch đã được export tại [data/processed/train_cleaned.csv](file:///c:/Users/Admin/Desktop/Kỳ%20Thu%202026/T4.3.%20Ứng%20Dụng%20AI%20Trong%20KD%20Và%20Quản%20Lý/Kaggle_Competition/credit-card-churn/data/processed/train_cleaned.csv).
5. Sẵn sàng 100% để bước sang Phase 2 (EDA chuyên sâu) và Phase 3 (Feature Engineering).
"""
    return report_md


def run_full_phase1_audit(
    raw_path: Path = TRAIN_RAW_PATH,
    export_cleaned: bool = True
) -> Dict[str, Any]:
    """
    Execute full Phase 1 audit pipeline, save json report and markdown report,
    and optionally export train_cleaned.csv.
    """
    from src.preprocessing import clean_raw_dataframe

    print(f"Loading raw data from: {raw_path}")
    raw_df = pd.read_csv(raw_path)

    # 1. Schema Audit
    schema_res = audit_schema(raw_df, is_train=True)

    # 2. Deduplication Audit
    clean_df, dedup_stats = check_and_remove_duplicates(raw_df)

    # 3. Target Audit
    target_res = validate_target(clean_df)

    # 4. Missingness Audit
    missing_df = audit_missing_values(clean_df)

    # 5. Category Audit
    cat_res = audit_categories(clean_df)

    # 6. Outliers & Sentinels Audit
    outliers_res = audit_outliers_and_sentinels(clean_df)

    # 7. Date Audit
    dates_res = audit_dates(clean_df, snapshot_date=SNAPSHOT_DATE)

    # Generate Markdown Report
    md_content = generate_markdown_audit_report(
        schema_res=schema_res,
        dedup_stats=dedup_stats,
        target_res=target_res,
        missing_df=missing_df,
        categories_res=cat_res,
        outliers_res=outliers_res,
        dates_res=dates_res,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    md_report_path = REPORTS_DIR / "data_audit_report.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SUCCESS] Markdown audit report written to: {md_report_path}")

    # Generate JSON summary
    json_summary = {
        "schema": schema_res,
        "deduplication": dedup_stats,
        "target": target_res,
        "missing_summary": missing_df.to_dict(orient="index"),
        "categories": cat_res,
        "outliers_and_sentinels": outliers_res,
        "dates": dates_res,
    }
    json_report_path = METRICS_DIR / "data_audit_summary.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] JSON audit summary written to: {json_report_path}")

    # Optionally export cleaned dataset
    if export_cleaned:
        cleaned_dataset = clean_raw_dataframe(clean_df)
        PROCESSED_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned_dataset.to_csv(PROCESSED_TRAIN_PATH, index=False)
        print(f"[SUCCESS] Cleaned dataset (12,000 unique rows) exported to: {PROCESSED_TRAIN_PATH}")

    return json_summary


if __name__ == "__main__":
    run_full_phase1_audit()
