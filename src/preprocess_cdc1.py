from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def normalize_column_name(col: str) -> str:
    col = col.strip().lower()
    col = col.replace("%", "percent")
    col = re.sub(r"[^\w\s]", "", col)
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm_text(x) -> str:
    return str(x).strip().lower()


def load_specific_drug_table(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("The input file is empty.")

    df.columns = [normalize_column_name(c) for c in df.columns]

    print("\n✅ Standardized column names:")
    print(df.columns.tolist())

    required_cols = [
        "death_year",
        "death_month",
        "jurisdiction_occurrence",
        "drug_involved",
        "month_ending_date",
        "drug_overdose_deaths",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.rename(
        columns={
            "death_year": "year",
            "death_month": "month",
            "jurisdiction_occurrence": "state_name",
            "drug_involved": "indicator",
            "drug_overdose_deaths": "overdose_value",
        }
    )

    if "time_period" in df.columns:
        df = df.rename(columns={"time_period": "period"})
    else:
        df["period"] = pd.NA

    if "footnote" not in df.columns:
        df["footnote"] = pd.NA

    df["period_end"] = pd.to_datetime(df["month_ending_date"], errors="coerce")
    df["overdose_value"] = pd.to_numeric(df["overdose_value"], errors="coerce")

    # Keep only 12-month ending data
    if "period" in df.columns:
        period_text = df["period"].astype(str).str.lower()
        mask_12m = period_text.str.contains("12", na=False) & period_text.str.contains("ending", na=False)
        if mask_12m.any():
            df = df[mask_12m].copy()

    df["state_name_norm"] = df["state_name"].apply(norm_text)
    df["indicator_norm"] = df["indicator"].apply(norm_text)

    # Remove rows with completely missing values
    df = df[df["overdose_value"].notna()].copy()

    df = df.sort_values(["state_name", "indicator", "period_end"]).reset_index(drop=True)
    return df


def export_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    # 1. Full cleaned table
    clean_df = df.drop(columns=["state_name_norm", "indicator_norm"], errors="ignore").copy()
    clean_df.to_csv(output_dir / "cdc_overdose_clean.csv", index=False)

    # 2. Coverage check table
    coverage_df = (
        df.groupby(["state_name", "indicator"], dropna=False)["overdose_value"]
        .apply(lambda s: s.notna().sum())
        .reset_index(name="non_null_count")
        .sort_values(["state_name", "non_null_count", "indicator"], ascending=[True, False, True])
    )
    coverage_df.to_csv(output_dir / "cdc_indicator_coverage.csv", index=False)

    # 3. United States trend table (keep non-null values only)
    us_df = df[df["state_name_norm"] == "united states"].copy()
    us_df = us_df.drop(columns=["state_name_norm", "indicator_norm"], errors="ignore")
    us_df = us_df.sort_values(["period_end", "indicator"])
    us_df.to_csv(output_dir / "cdc_us_trend.csv", index=False)

    # 4. Selected regional trend table: keep only Region 7 / 8 / 10 + United States
    selected_regions = {"region 7", "region 8", "region 10", "united states"}
    selected_df = df[df["state_name_norm"].isin(selected_regions)].copy()
    selected_df = selected_df.drop(columns=["state_name_norm", "indicator_norm"], errors="ignore")
    selected_df = selected_df.sort_values(["state_name", "period_end", "indicator"])
    selected_df.to_csv(output_dir / "cdc_selected_trends.csv", index=False)

    # 5. Top drug trends: keep only the drugs most suitable for presentation
    # Updated here to match the indicators that actually have values in your current file
    top_regions = {"region 7", "region 8", "region 10"}
    top_indicators = {"heroin", "cocaine", "methamphetamine"}

    top_drug_df = df[
        df["state_name_norm"].isin(top_regions)
        & df["indicator_norm"].isin(top_indicators)
    ].copy()

    top_drug_df = top_drug_df[["state_name", "indicator", "period_end", "overdose_value"]]
    top_drug_df = top_drug_df.sort_values(["state_name", "indicator", "period_end"])
    top_drug_df.to_csv(output_dir / "cdc_top_drug_trends.csv", index=False)

    # 6. Region summary: number of non-null rows for each region and indicator
    region_summary_df = (
        top_drug_df.groupby(["state_name", "indicator"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["state_name", "indicator"])
    )
    region_summary_df.to_csv(output_dir / "cdc_region_summary.csv", index=False)

    print("\n✅ Output completed:")
    print(output_dir / "cdc_overdose_clean.csv")
    print(output_dir / "cdc_indicator_coverage.csv")
    print(output_dir / "cdc_us_trend.csv")
    print(output_dir / "cdc_selected_trends.csv")
    print(output_dir / "cdc_top_drug_trends.csv")
    print(output_dir / "cdc_region_summary.csv")

    print("\n📊 Row count summary:")
    print(f"cdc_overdose_clean.csv: {len(clean_df):,}")
    print(f"cdc_indicator_coverage.csv: {len(coverage_df):,}")
    print(f"cdc_us_trend.csv: {len(us_df):,}")
    print(f"cdc_selected_trends.csv: {len(selected_df):,}")
    print(f"cdc_top_drug_trends.csv: {len(top_drug_df):,}")
    print(f"cdc_region_summary.csv: {len(region_summary_df):,}")

    if len(top_drug_df) > 0:
        print("\nFirst 5 rows of top drug trends:")
        print(top_drug_df.head())


def main() -> None:
    parser = argparse.ArgumentParser(description="Process CDC specific-drug regional data.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/Provisional_drug_overdose_death_counts_for_specific_drugs_20260331.csv",
        help="Path to the raw CDC CSV file (relative to project root)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/cleaned",
        help="Output directory (relative to project root)",
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

    print(f"📂 Reading raw CDC file: {input_path}")

    try:
        df = load_specific_drug_table(input_path)
        export_outputs(df, output_dir)
        print("\n✅ All tasks completed.")
    except Exception as e:
        print(f"\n❌ Run failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()