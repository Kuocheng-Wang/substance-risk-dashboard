from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"state_name", "indicator", "period_end", "overdose_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["overdose_value"] = pd.to_numeric(df["overdose_value"], errors="coerce")
    df = df.dropna(subset=["state_name", "indicator", "period_end", "overdose_value"]).copy()

    if df.empty:
        raise ValueError("Input data is empty. Cannot generate plots.")

    df = df.sort_values(["state_name", "indicator", "period_end"]).reset_index(drop=True)
    return df


def plot_region_drug_trends(df: pd.DataFrame, output_dir: Path) -> None:
    """
    One figure for each region:
    Plot all drug trends for that region in the same figure.
    """
    regions = sorted(df["state_name"].unique())

    for region in regions:
        region_df = df[df["state_name"] == region].copy()
        indicators = sorted(region_df["indicator"].unique())

        plt.figure(figsize=(11, 6))
        for indicator in indicators:
            temp = region_df[region_df["indicator"] == indicator].copy()
            plt.plot(temp["period_end"], temp["overdose_value"], label=indicator)

        plt.title(f"CDC Drug Trends - {region}")
        plt.xlabel("Period End")
        plt.ylabel("Overdose Value")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        filename = output_dir / f"cdc_region_trends_{region.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


def plot_drug_region_comparisons(df: pd.DataFrame, output_dir: Path) -> None:
    """
    One figure for each drug:
    Compare trends across different regions in the same figure.
    """
    indicators = sorted(df["indicator"].unique())

    for indicator in indicators:
        drug_df = df[df["indicator"] == indicator].copy()
        regions = sorted(drug_df["state_name"].unique())

        plt.figure(figsize=(11, 6))
        for region in regions:
            temp = drug_df[drug_df["state_name"] == region].copy()
            plt.plot(temp["period_end"], temp["overdose_value"], label=region)

        plt.title(f"CDC Regional Comparison - {indicator}")
        plt.xlabel("Period End")
        plt.ylabel("Overdose Value")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        safe_name = indicator.lower().replace(" ", "_").replace("/", "_")
        filename = output_dir / f"cdc_drug_comparison_{safe_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


def plot_latest_bar_chart(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Take the latest value for each region + drug combination
    and create one summary bar chart.
    """
    latest_df = (
        df.sort_values("period_end")
        .groupby(["state_name", "indicator"], as_index=False)
        .tail(1)
        .copy()
    )

    pivot_df = latest_df.pivot(index="state_name", columns="indicator", values="overdose_value")

    plt.figure(figsize=(11, 6))
    pivot_df.plot(kind="bar", ax=plt.gca())
    plt.title("Latest CDC Drug Values by Region")
    plt.xlabel("Region")
    plt.ylabel("Overdose Value")
    plt.xticks(rotation=0)
    plt.tight_layout()

    filename = output_dir / "cdc_latest_values_bar.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    input_path = project_root / "outputs" / "cleaned" / "cdc_top_drug_trends.csv"
    output_dir = project_root / "outputs" / "figures"
    ensure_dir(output_dir)

    print(f"Reading file: {input_path}")
    df = load_data(input_path)

    print("\nData summary:")
    print(f"- Rows: {len(df)}")
    print(f"- Regions: {sorted(df['state_name'].unique().tolist())}")
    print(f"- Indicators: {sorted(df['indicator'].unique().tolist())}")
    print(f"- Date range: {df['period_end'].min().date()} to {df['period_end'].max().date()}")

    plot_region_drug_trends(df, output_dir)
    plot_drug_region_comparisons(df, output_dir)
    plot_latest_bar_chart(df, output_dir)

    print("\n✅ Plotting completed. Output files are saved in:")
    print(output_dir)
    for p in sorted(output_dir.glob("cdc_*.png")):
        print("-", p.name)


if __name__ == "__main__":
    main()