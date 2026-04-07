"""Event classification and event-study helpers."""

from __future__ import annotations

import pandas as pd

from src.config import DATE_COLUMN, DEFAULT_EVENT_WINDOW, EVENT_COLUMN, EVENT_TYPE_KEYWORDS
from src.validation import ensure_columns, normalize_date_column


EVENT_STUDY_COLUMNS = [
    DATE_COLUMN,
    "relative_day",
    "event_name",
    "event_date",
    "event_year",
    "event_type",
    "price_value",
    "return_vs_event_%",
]


def classify_event_type(event_name: str) -> str:
    """Classify an event name using a small keyword rule set."""
    event_text = str(event_name).strip().lower()
    for label, keywords in EVENT_TYPE_KEYWORDS.items():
        if any(keyword in event_text for keyword in keywords):
            return label
    return "Other"


def build_event_study_frame(
    price_df: pd.DataFrame,
    price_col: str,
    event_col: str = EVENT_COLUMN,
    window_size: int = DEFAULT_EVENT_WINDOW,
) -> pd.DataFrame:
    """Build an event-window frame for a single price series."""
    frame = normalize_date_column(price_df)
    ensure_columns(frame, [DATE_COLUMN, price_col, event_col], f"event study for {price_col}")

    study_frame = frame[[DATE_COLUMN, price_col, event_col]].copy()
    study_frame[event_col] = study_frame[event_col].fillna("").astype(str)
    event_positions = study_frame.index[study_frame[event_col].str.strip().ne("")]

    study_rows: list[pd.DataFrame] = []
    for event_pos in event_positions:
        start_index = event_pos - window_size
        end_index = event_pos + window_size
        if start_index < 0 or end_index >= len(study_frame):
            continue

        event_name = study_frame.at[event_pos, event_col].strip()
        event_price = study_frame.at[event_pos, price_col]
        if pd.isna(event_price) or event_price == 0:
            continue

        window = study_frame.iloc[start_index : end_index + 1].copy()
        window["relative_day"] = range(-window_size, window_size + 1)
        window["event_name"] = event_name
        window["event_date"] = study_frame.at[event_pos, DATE_COLUMN]
        window["event_year"] = study_frame.at[event_pos, DATE_COLUMN].year
        window["event_type"] = classify_event_type(event_name)
        window["price_value"] = window[price_col]
        window["return_vs_event_%"] = ((window[price_col] / event_price) - 1) * 100
        study_rows.append(window[EVENT_STUDY_COLUMNS])

    if not study_rows:
        return pd.DataFrame(columns=EVENT_STUDY_COLUMNS)

    return pd.concat(study_rows, ignore_index=True)


def build_event_reference(event_study_frame: pd.DataFrame) -> pd.DataFrame:
    """Return distinct event metadata for filtering and table joins."""
    if event_study_frame.empty:
        return pd.DataFrame(columns=["Event Date", "Event", "Event Year", "Event Type"])

    return (
        event_study_frame[["event_date", "event_name", "event_year", "event_type"]]
        .drop_duplicates()
        .sort_values("event_date")
        .rename(
            columns={
                "event_date": "Event Date",
                "event_name": "Event",
                "event_year": "Event Year",
                "event_type": "Event Type",
            }
        )
        .reset_index(drop=True)
    )


def filter_event_studies(
    gold_event_study: pd.DataFrame,
    silver_event_study: pd.DataFrame,
    event_reference: pd.DataFrame,
    year_range: tuple[int, int],
    event_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply year and event-type filters to paired event-study frames."""
    filtered_reference = event_reference[event_reference["Event Year"].between(year_range[0], year_range[1])].copy()
    if event_type != "All":
        filtered_reference = filtered_reference[filtered_reference["Event Type"] == event_type]

    selected_event_names = filtered_reference["Event"].tolist()
    filtered_gold = gold_event_study[gold_event_study["event_name"].isin(selected_event_names)].copy()
    filtered_silver = silver_event_study[silver_event_study["event_name"].isin(selected_event_names)].copy()
    return filtered_reference, filtered_gold, filtered_silver


def summarize_event_returns(gold_event_study: pd.DataFrame, silver_event_study: pd.DataFrame) -> pd.DataFrame:
    """Summarize pre-, event-, and post-event returns for both metals."""
    summaries = []
    for metal_name, study_frame in (("Gold", gold_event_study), ("Silver", silver_event_study)):
        summaries.append(
            {
                "Metal": metal_name,
                "Average Pre-Event Return %": study_frame.loc[study_frame["relative_day"] < 0, "return_vs_event_%"].mean(),
                "Average Post-Event Return %": study_frame.loc[study_frame["relative_day"] > 0, "return_vs_event_%"].mean(),
                "Average Event-Day Return %": study_frame.loc[study_frame["relative_day"] == 0, "return_vs_event_%"].mean(),
                "Events Included": int(study_frame["event_name"].nunique()),
            }
        )
    return pd.DataFrame(summaries)


def summarize_event_prices(gold_event_study: pd.DataFrame, silver_event_study: pd.DataFrame) -> pd.DataFrame:
    """Summarize pre-, event-, and post-event prices for both metals."""
    summaries = []
    for metal_name, study_frame in (("Gold", gold_event_study), ("Silver", silver_event_study)):
        summaries.append(
            {
                "Metal": metal_name,
                "Average Pre-Event Price": study_frame.loc[study_frame["relative_day"] < 0, "price_value"].mean(),
                "Average Event-Day Price": study_frame.loc[study_frame["relative_day"] == 0, "price_value"].mean(),
                "Average Post-Event Price": study_frame.loc[study_frame["relative_day"] > 0, "price_value"].mean(),
                "Events Included": int(study_frame["event_name"].nunique()),
            }
        )
    return pd.DataFrame(summaries)


def _aggregate_event_prices(event_study_frame: pd.DataFrame, metal_name: str) -> pd.DataFrame:
    """Aggregate average pre-, event-, and post-event prices per event."""
    rows: list[dict[str, object]] = []
    for (event_name, event_date), frame in event_study_frame.groupby(["event_name", "event_date"]):
        rows.append(
            {
                "event_name": event_name,
                "event_date": event_date,
                f"{metal_name} Pre-Event Price": frame.loc[frame["relative_day"] < 0, "price_value"].mean(),
                f"{metal_name} Event-Day Price": frame.loc[frame["relative_day"] == 0, "price_value"].mean(),
                f"{metal_name} Post-Event Price": frame.loc[frame["relative_day"] > 0, "price_value"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_event_impact_table(
    filtered_event_reference: pd.DataFrame,
    gold_event_study: pd.DataFrame,
    silver_event_study: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-event impact table across both metals."""
    gold_event_impact = _aggregate_event_prices(gold_event_study, "Gold")
    silver_event_impact = _aggregate_event_prices(silver_event_study, "Silver")

    event_impact_table = (
        filtered_event_reference.rename(columns={"Event": "event_name", "Event Date": "event_date"})
        .merge(gold_event_impact, on=["event_name", "event_date"], how="left")
        .merge(silver_event_impact, on=["event_name", "event_date"], how="left")
        .rename(columns={"event_name": "Event", "event_date": "Event Date"})
    )

    event_impact_table["Gold % Change Pre->Event"] = (
        (event_impact_table["Gold Event-Day Price"] / event_impact_table["Gold Pre-Event Price"] - 1) * 100
    )
    event_impact_table["Gold % Change Event->Post"] = (
        (event_impact_table["Gold Post-Event Price"] / event_impact_table["Gold Event-Day Price"] - 1) * 100
    )
    event_impact_table["Gold % Change Pre->Post"] = (
        (event_impact_table["Gold Post-Event Price"] / event_impact_table["Gold Pre-Event Price"] - 1) * 100
    )
    event_impact_table["Silver % Change Pre->Event"] = (
        (event_impact_table["Silver Event-Day Price"] / event_impact_table["Silver Pre-Event Price"] - 1) * 100
    )
    event_impact_table["Silver % Change Event->Post"] = (
        (event_impact_table["Silver Post-Event Price"] / event_impact_table["Silver Event-Day Price"] - 1) * 100
    )
    event_impact_table["Silver % Change Pre->Post"] = (
        (event_impact_table["Silver Post-Event Price"] / event_impact_table["Silver Pre-Event Price"] - 1) * 100
    )
    return event_impact_table