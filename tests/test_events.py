from __future__ import annotations

import unittest

import pandas as pd

from src.analysis.events import (
    build_event_impact_table,
    build_event_reference,
    build_event_study_frame,
    classify_event_type,
    filter_event_studies,
)


class EventAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=7, freq="D"),
                "Price_gold": [100, 101, 102, 103, 104, 105, 106],
                "Price_silver": [20, 21, 22, 23, 24, 25, 26],
                "EVENT": ["", "", "War escalates", "", "", "", ""],
            }
        )

    def test_classify_event_type_uses_keywords(self) -> None:
        self.assertEqual(classify_event_type("War escalates in region"), "War / Military Conflict")
        self.assertEqual(classify_event_type("Terror threat rises"), "Terrorism / Security Threat")
        self.assertEqual(classify_event_type("Diplomatic visit"), "Other")

    def test_build_event_study_frame_creates_window(self) -> None:
        study_frame = build_event_study_frame(self.frame, "Price_gold", window_size=2)

        self.assertEqual(len(study_frame), 5)
        self.assertListEqual(study_frame["relative_day"].tolist(), [-2, -1, 0, 1, 2])
        self.assertTrue((study_frame["event_type"] == "War / Military Conflict").all())
        event_day_row = study_frame.loc[study_frame["relative_day"] == 0].iloc[0]
        self.assertAlmostEqual(event_day_row["return_vs_event_%"], 0.0)

    def test_event_filters_and_impact_table_preserve_event_metadata(self) -> None:
        gold_study = build_event_study_frame(self.frame, "Price_gold", window_size=2)
        silver_study = build_event_study_frame(self.frame, "Price_silver", window_size=2)
        event_reference = build_event_reference(gold_study)

        filtered_reference, filtered_gold, filtered_silver = filter_event_studies(
            gold_study,
            silver_study,
            event_reference,
            year_range=(2024, 2024),
            event_type="War / Military Conflict",
        )
        event_impact_table = build_event_impact_table(filtered_reference, filtered_gold, filtered_silver)

        self.assertEqual(len(filtered_reference), 1)
        self.assertEqual(event_impact_table.loc[0, "Event"], "War escalates")
        self.assertIn("Gold % Change Pre->Post", event_impact_table.columns)
        self.assertIn("Silver % Change Pre->Post", event_impact_table.columns)


if __name__ == "__main__":
    unittest.main()