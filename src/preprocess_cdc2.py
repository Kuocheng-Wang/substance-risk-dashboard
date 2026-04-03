from __future__ import annotations

import argparse
import calendar
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def normalize_column_name(col: str) -> str:
    """Standardize column names."""
    col = str(col).strip().lower()
    col = col.replace("%", "percent")
    col = re.sub(r"[^\w\s]", "", col)
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def month_to_number(value) -> Optional[int]:
    """Convert month values to 1-12. Supports January / Jan / 1 / 01."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.isdigit():
        num = int(text)
        return num if 1 <= num <= 12 else None

    month_map = {}
    for i in range(1, 13):
        month_map[calendar.month_name[i].lower()] = i
        month_map[calendar.month_abbr[i].lower()] = i

    return month_map.get(text.lower())


def build_period_start(df: pd.DataFrame) -> pd.Series:
    """Generate the first day of each month."""
    year_num = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    month_num = df["month"].apply(month_to_number).astype("Int64")

    return pd.to_datetime(
        {
            "year": year_num,
            "month": month_num,
            "day": 1,
        },
        errors="coerce",
    )


def build_period_end(df: pd.DataFrame) -> pd.Series:
    """Generate the last day of each month."""
    period_start = build_period_start(df)
    return period_start + pd.offsets.MonthEnd(0)


def parse_states(states_text: Optional[str]) -> Optional[set[str]]:
    """Parse the state list passed from the command line."""
    if states_text is None or not str(states_text).strip():
        return None
    return {
        x.strip().lower()
        for x in str(states_text).split(",")
        if x.strip()
    }


def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        sys.exit(1)


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by state + time + indicator, keeping the more complete row."""
    quality_cols = [
        "predicted_value",
        "data_value",
        "percent_complete",
        "percent_pending_investigation",
    ]
    existing_quality_cols = [c for c in quality_cols if c in df.columns]

    df = df.copy()
    df["_quality_score"] = df[existing_quality_cols].notna().sum(axis=1)
    df = df.sort_values(
        ["state_name", "indicator", "year", "month", "_quality_score"],
        ascending=[True, True, True, True, False],
    )
    df = df.drop_duplicates(
        subset=["state", "state_name", "year", "month", "period", "indicator"],
        keep="first",
    ).copy()
    df = df.drop(columns=["_quality_score"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess CDC overdose table.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/VSRR_Provisional_Drug_Overdose_Death_Counts_20260331.csv",
        help="Input CSV path relative to the project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/cleaned",
        help="Output directory relative to the project root.",
    )
    parser.add_argument(
        "--states",
        type=str,
        default="United States,Missouri,Kansas",
        help="Comma-separated state/region names. Example: 'United States,Missouri,Kansas'",
    )
    parser.add_argument(
        "--min-percent-complete",
        type=float,
        default=None,
        help="Optional filter. Keep rows with percent_complete >= this value.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    input_path = (project_root / args.input).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    ensure_directory(output_dir)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    print(f"📂 Reading file: {input_path}")
    df = pd.read_csv(input_path)

    if df.empty:
        print("❌ The input file is empty.")
        sys.exit(1)

    # 1. Standardize column names
    df.columns = [normalize_column_name(c) for c in df.columns]

    print("\n✅ Standardized column names:")
    print(df.columns.tolist())

    # 2. Check required columns
    required_cols = [
        "state",
        "year",
        "month",
        "period",
        "indicator",
        "data_value",
        "percent_complete",
        "percent_pending_investigation",
        "state_name",
        "predicted_value",
    ]
    validate_required_columns(df, required_cols)

    # 3. Convert numeric columns
    numeric_cols = [
        "data_value",
        "predicted_value",
        "percent_complete",
        "percent_pending_investigation",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Keep only 12 month-ending rows
    df["period_norm"] = df["period"].astype(str).str.strip().str.lower()
    df = df[
        df["period_norm"].str.contains("12", na=False)
        & df["period_norm"].str.contains("ending", na=False)
    ].copy()

    # 5. Standardized helper columns
    df["state_name_norm"] = df["state_name"].astype(str).str.strip().str.lower()
    df["indicator_norm"] = df["indicator"].astype(str).str.strip().str.lower()

    # 6. Build date columns
    df["month_num"] = df["month"].apply(month_to_number)
    df["period_start"] = build_period_start(df)
    df["period_end"] = build_period_end(df)
    df["year_month"] = df["period_start"].dt.strftime("%Y-%m")

    # 7. Deduplicate
    before_dedup = len(df)
    df = deduplicate_rows(df)
    dropped_dups = before_dedup - len(df)

    # 8. Optional quality filter
    if args.min_percent_complete is not None:
        df = df[df["percent_complete"] >= args.min_percent_complete].copy()

    # 9. Unified analysis value and source
    df["analysis_value"] = df["predicted_value"].fillna(df["data_value"])
    df["value_source"] = df["predicted_value"].notna().map(
        {True: "predicted_value", False: "data_value"}
    )

    # 10. Indicator type
    total_indicator = "number of drug overdose deaths"
    percent_indicator = "percent with drugs specified"

    df["metric_type"] = "other"
    df.loc[df["indicator_norm"] == total_indicator, "metric_type"] = "count"
    df.loc[df["indicator_norm"] == percent_indicator, "metric_type"] = "percent"

    # 11. Simple cleaning for percentage indicators: keep only 0-100
    percent_mask = df["metric_type"] == "percent"
    df.loc[
        percent_mask & ~df["analysis_value"].between(0, 100, inclusive="both"),
        "analysis_value",
    ] = pd.NA

    # 12. Save full cleaned table
    clean_full = df.sort_values(["state_name", "period_end", "indicator"]).copy()
    clean_full_out = output_dir / "cdc2_clean_full.csv"
    clean_full.to_csv(clean_full_out, index=False)

    # 13. Select states
    selected_states = parse_states(args.states)
    if selected_states is None:
        selected_df = df.copy()
    else:
        selected_df = df[df["state_name_norm"].isin(selected_states)].copy()

    if selected_df.empty:
        print(f"❌ No states/regions were selected. Please check the --states argument: {args.states}")
        sys.exit(1)

    # 14. Main analysis table: keep only the two core indicators
    main_selected = selected_df[
        selected_df["indicator_norm"].isin({total_indicator, percent_indicator})
    ].copy()

    main_selected = main_selected.sort_values(["state_name", "indicator", "period_end"])
    main_selected_out = output_dir / "cdc2_main_selected.csv"
    main_selected.to_csv(main_selected_out, index=False)

    # 15. Total overdose trend table
    total_df = main_selected[
        (main_selected["indicator_norm"] == total_indicator)
        & main_selected["analysis_value"].notna()
    ][
        [
            "state",
            "state_name",
            "year",
            "month",
            "month_num",
            "year_month",
            "period",
            "period_start",
            "period_end",
            "indicator",
            "metric_type",
            "data_value",
            "predicted_value",
            "analysis_value",
            "value_source",
            "percent_complete",
            "percent_pending_investigation",
        ]
    ].sort_values(["state_name", "period_end"])

    total_out = output_dir / "cdc2_total_overdose_trend.csv"
    total_df.to_csv(total_out, index=False)

    # 16. Percent with drugs specified trend table
    percent_df = main_selected[
        (main_selected["indicator_norm"] == percent_indicator)
        & main_selected["analysis_value"].notna()
    ][
        [
            "state",
            "state_name",
            "year",
            "month",
            "month_num",
            "year_month",
            "period",
            "period_start",
            "period_end",
            "indicator",
            "metric_type",
            "data_value",
            "predicted_value",
            "analysis_value",
            "value_source",
            "percent_complete",
            "percent_pending_investigation",
        ]
    ].sort_values(["state_name", "period_end"])

    percent_out = output_dir / "cdc2_percent_specified_trend.csv"
    percent_df.to_csv(percent_out, index=False)

    # 17. Print summary
    print("\n✅ Processing completed")
    print(f"Full cleaned table: {clean_full_out}")
    print(f"Main analysis table: {main_selected_out}")
    print(f"Total trend table: {total_out}")
    print(f"Percentage table: {percent_out}")

    print("\n📊 Row count summary")
    print(f"- clean full: {len(clean_full):,}")
    print(f"- main selected: {len(main_selected):,}")
    print(f"- total overdose trend: {len(total_df):,}")
    print(f"- percent specified trend: {len(percent_df):,}")
    print(f"- deduplicated rows removed: {dropped_dups:,}")

    print("\n📍 Actually selected states/regions:")
    print(sorted(main_selected["state_name"].dropna().unique().tolist()))

    if len(total_df) > 0:
        print("\nFirst 5 rows of total overdose trend:")
        print(total_df[["state_name", "period_end", "analysis_value", "value_source"]].head())

    if len(percent_df) > 0:
        print("\nFirst 5 rows of percent specified trend:")
        print(percent_df[["state_name", "period_end", "analysis_value", "value_source"]].head())


if __name__ == "__main__":
    main()