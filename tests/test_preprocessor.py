from __future__ import annotations

import unittest

import pandas as pd

from src.data.preprocessor import clean_data, feature_engineering


class PreprocessorTests(unittest.TestCase):
    def setUp(self) -> None:
        row_count = 40
        self.frame = pd.DataFrame(
            {
                "DATE": pd.date_range("2024-01-01", periods=row_count, freq="D"),
                "Price_gold": [100.0] * 39 + [10000.0],
                "Price_silver": [20.0] * row_count,
                "GPR": [1.0] * row_count,
            }
        )
        self.frame.loc[5, "Price_silver"] = None

    def test_clean_data_normalizes_dates_and_removes_outlier(self) -> None:
        cleaned = clean_data(self.frame)

        self.assertIn("Date", cleaned.columns)
        self.assertNotIn("DATE", cleaned.columns)
        self.assertEqual(len(cleaned), 39)
        self.assertEqual(cleaned["Price_gold"].max(), 100.0)
        self.assertFalse(cleaned[["Price_gold", "Price_silver", "GPR"]].isna().any().any())

    def test_feature_engineering_creates_expected_features(self) -> None:
        cleaned = clean_data(self.frame)
        features = feature_engineering(cleaned)

        expected_columns = {
            "gold_lag1",
            "gold_lag7",
            "silver_lag1",
            "silver_lag7",
            "GPR_lag7",
            "gold_roll_mean30",
            "gold_volatility30",
            "silver_roll_mean30",
            "silver_volatility30",
        }
        self.assertTrue(expected_columns.issubset(features.columns))
        self.assertTrue(features["Date"].is_monotonic_increasing)
        self.assertFalse(features.isna().any().any())


if __name__ == "__main__":
    unittest.main()