"""
Unit tests for data validation, preprocessing, date parsing, and submission contract.
Compatible with standard library unittest and pytest.
Based on Phase 10 & Phase 14 in Classification_Plan.md.
"""

import unittest
import pandas as pd
import numpy as np

from src.data_validation import (
    validate_schema,
    check_and_remove_duplicates,
    validate_target,
    validate_submission,
)
from src.preprocessing import (
    clean_gender,
    clean_province,
    clean_salary_via_bank,
    clean_occupation,
    parse_open_date,
)


class TestValidationAndPreprocessing(unittest.TestCase):

    def test_clean_gender(self):
        self.assertEqual(clean_gender("female"), "Female")
        self.assertEqual(clean_gender("F"), "Female")
        self.assertEqual(clean_gender("nam"), "Male")
        self.assertEqual(clean_gender("M"), "Male")
        self.assertEqual(clean_gender(np.nan), "Unknown")

    def test_clean_province(self):
        self.assertEqual(clean_province("TP.HCM"), "TP.HCM")
        self.assertEqual(clean_province("hồ chí minh"), "TP.HCM")
        self.assertEqual(clean_province("HCM"), "TP.HCM")
        self.assertEqual(clean_province("Hà Nội"), "Hà Nội")
        self.assertEqual(clean_province("ha noi"), "Hà Nội")
        self.assertEqual(clean_province(np.nan), "Unknown")

    def test_clean_salary_via_bank(self):
        self.assertEqual(clean_salary_via_bank("Y"), "Y")
        self.assertEqual(clean_salary_via_bank("yes"), "Y")
        self.assertEqual(clean_salary_via_bank("N"), "N")
        self.assertEqual(clean_salary_via_bank("no"), "N")
        self.assertEqual(clean_salary_via_bank(np.nan), "Unknown")

    def test_parse_open_date_formats(self):
        dates = pd.Series([
            "2021-05-15",   # YYYY-MM-DD
            "20200820",     # YYYYMMDD
            "25/12/2019",   # DD/MM/YYYY
            np.nan,
        ])
        parsed = parse_open_date(dates)

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2021-05-15"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2020-08-20"))
        self.assertEqual(parsed.iloc[2], pd.Timestamp("2019-12-25"))
        self.assertTrue(pd.isna(parsed.iloc[3]))

    def test_deduplication(self):
        df = pd.DataFrame({
            "customer_id": ["C1", "C2", "C1"],
            "feature": [10, 20, 10],
            "churn_3m": [0, 1, 0]
        })
        clean_df, stats = check_and_remove_duplicates(df)
        self.assertEqual(len(clean_df), 2)
        self.assertEqual(stats["duplicate_rows_removed"], 1)
        self.assertEqual(stats["clean_rows"], 2)

    def test_validate_target(self):
        df_valid = pd.DataFrame({"churn_3m": [0, 1, 0, 1, 0]})
        res = validate_target(df_valid)
        self.assertTrue(res["is_valid_binary"])
        self.assertAlmostEqual(res["churn_rate"], 0.4)

        df_invalid = pd.DataFrame({"churn_3m": [0, 2, 0]})
        res_inv = validate_target(df_invalid)
        self.assertFalse(res_inv["is_valid_binary"])

    def test_submission_contract(self):
        # Valid submission
        valid_sub = pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "churn_3m": [0.123, 0.854]
        })
        is_valid, errors = validate_submission(valid_sub, expected_rows=2)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Invalid columns
        invalid_sub_cols = pd.DataFrame({
            "id": ["C1", "C2"],
            "churn": [0.1, 0.2]
        })
        is_valid, errors = validate_submission(invalid_sub_cols)
        self.assertFalse(is_valid)

        # Out of bounds probability
        invalid_sub_prob = pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "churn_3m": [1.5, -0.2]
        })
        is_valid, errors = validate_submission(invalid_sub_prob)
        self.assertFalse(is_valid)

    def test_standardize_monetary_units(self):
        from src.preprocessing import standardize_monetary_units
        df = pd.DataFrame({
            "income_month": [15.0, 31000000.0, 25.5],
            "credit_limit": [50.0, 40000000.0, 100.0]
        })
        standardized = standardize_monetary_units(df)
        self.assertAlmostEqual(standardized["income_month"].iloc[1], 31.0)
        self.assertAlmostEqual(standardized["credit_limit"].iloc[1], 40.0)
        self.assertAlmostEqual(standardized["income_month"].iloc[0], 15.0)

    def test_audit_outliers_and_sentinels(self):
        from src.data_validation import audit_outliers_and_sentinels
        df = pd.DataFrame({
            "age": [25, 999, 15, 105],
            "income_month": [15.0, 25000000.0, np.nan, 20.0],
            "credit_limit": [40.0, 35000000.0, 50.0, 100.0],
            "utilization_6m": [-1.0, 0.5, 2.5, 0.2],
            "txn_amount_3m": [-1.5, 10.0, 25.0, 0.0]
        })
        res = audit_outliers_and_sentinels(df)
        self.assertEqual(res["age"]["sentinel_999_count"], 1)
        self.assertEqual(res["age"]["under_18_count"], 1)
        self.assertEqual(res["age"]["over_100_count"], 2)  # 999 and 105 are > 100
        self.assertEqual(res["income_month"]["recorded_in_raw_vnd_count"], 1)
        self.assertEqual(res["utilization_6m"]["negative_count"], 1)
        self.assertEqual(res["txn_amount_3m"]["negative_count"], 1)


if __name__ == "__main__":
    unittest.main()
