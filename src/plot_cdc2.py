from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("❌ matplotlib is not installed.")
    print("Please install it first: pip install matplotlib")
    sys.exit(1)


TOTAL_INDICATOR = "Number of Drug Overdose Deaths"
PERCENT_INDICATOR = "Percent with drugs specified"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"File is empty: {path}")
    return df


def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "period_end" in df.columns:
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    if "period_start" in df.columns:
        df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "analysis_value" in df.columns:
        df["analysis_value"] = pd.to_numeric(df["analysis_value"], errors="coerce")

    if "percent_complete" in df.columns:
        df["percent_complete"] = pd.to_numeric(df["percent_complete"], errors="coerce")

    return df


def validate_trend_df(df: pd.DataFrame, name: str) -> None:
    required = ["state_name", "period_end", "analysis_value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def rebuild_trend_from_main(main_df: pd.DataFrame, indicator_name: str) -> pd.DataFrame:
    df = main_df.copy()

    if "indicator" not in df.columns:
        raise ValueError("The indicator column is missing in cdc2_main_selected.csv, so the trend table cannot be rebuilt.")

    df = df[df["indicator"] == indicator_name].copy()

    if df.empty:
        raise ValueError(f"Indicator not found in main_selected: {indicator_name}")

    keep_cols = [
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

    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()
    df = standardize_dates(df)
    df = df.sort_values(["state_name", "period_end"]).reset_index(drop=True)
    return df


def load_or_rebuild_trend(
    trend_path: Path,
    main_df: pd.DataFrame,
    indicator_name: str,
    expected_min_states: int = 2,
) -> tuple[pd.DataFrame, bool]:
    rebuilt = False

    if trend_path.exists():
        try:
            df = load_csv(trend_path)
            df = standardize_dates(df)
            validate_trend_df(df, trend_path.name)

            n_states = df["state_name"].nunique(dropna=True)
            if n_states < expected_min_states:
                print(
                    f"⚠️ {trend_path.name} contains only {n_states} region(s), which may be incomplete. "
                    f"It will be rebuilt from cdc2_main_selected.csv."
                )
                df = rebuild_trend_from_main(main_df, indicator_name)
                rebuilt = True
            else:
                df = df.sort_values(["state_name", "period_end"]).reset_index(drop=True)

            return df, rebuilt

        except Exception as e:
            print(f"⚠️ Failed to read {trend_path.name}: {e}")
            print("It will be rebuilt from cdc2_main_selected.csv.")

    df = rebuild_trend_from_main(main_df, indicator_name)
    rebuilt = True
    return df, rebuilt


def save_rebuilt_if_needed(df: pd.DataFrame, output_path: Path, rebuilt: bool) -> None:
    if rebuilt:
        df.to_csv(output_path, index=False)
        print(f"🛠 Rebuilt and saved: {output_path}")


def filter_states(df: pd.DataFrame, states: list[str] | None) -> pd.DataFrame:
    if not states:
        return df.copy()

    target = {s.strip().lower() for s in states if s.strip()}
    out = df[df["state_name"].astype(str).str.lower().isin(target)].copy()

    if out.empty:
        print("⚠️ No matching states/regions were found. Falling back to all regions.")
        return df.copy()

    return out


def subset_states(df: pd.DataFrame, states: list[str]) -> pd.DataFrame:
    target = {s.strip().lower() for s in states}
    return df[df["state_name"].astype(str).str.lower().isin(target)].copy()


def make_latest_summary(total_df: pd.DataFrame, percent_df: pd.DataFrame) -> pd.DataFrame:
    total_latest = (
        total_df.sort_values(["state_name", "period_end"])
        .groupby("state_name", as_index=False)
        .tail(1)[["state_name", "period_end", "analysis_value"]]
        .rename(columns={"analysis_value": "latest_total_overdose"})
    )

    percent_latest = (
        percent_df.sort_values(["state_name", "period_end"])
        .groupby("state_name", as_index=False)
        .tail(1)[["state_name", "period_end", "analysis_value"]]
        .rename(columns={"analysis_value": "latest_percent_specified"})
    )

    merged = pd.merge(
        total_latest,
        percent_latest[["state_name", "latest_percent_specified"]],
        on="state_name",
        how="outer",
    )

    return merged.sort_values("state_name").reset_index(drop=True)


def make_yearly_summary(total_df: pd.DataFrame, percent_df: pd.DataFrame) -> pd.DataFrame:
    total_yearly = (
        total_df.groupby(["state_name", "year"], as_index=False)["analysis_value"]
        .mean()
        .rename(columns={"analysis_value": "avg_total_overdose"})
    )

    percent_yearly = (
        percent_df.groupby(["state_name", "year"], as_index=False)["analysis_value"]
        .mean()
        .rename(columns={"analysis_value": "avg_percent_specified"})
    )

    yearly = pd.merge(
        total_yearly,
        percent_yearly,
        on=["state_name", "year"],
        how="outer",
    )

    return yearly.sort_values(["state_name", "year"]).reset_index(drop=True)


def add_end_labels(ax, df: pd.DataFrame, y_col: str) -> None:
    for state_name, sub_df in df.groupby("state_name"):
        sub_df = sub_df.sort_values("period_end").dropna(subset=[y_col])
        if sub_df.empty:
            continue
        last_row = sub_df.iloc[-1]
        ax.annotate(
            state_name,
            xy=(last_row["period_end"], last_row[y_col]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=10,
            va="center",
        )


def plot_presentation_line_chart(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    subtitle: str,
    ylabel: str,
    output_path: Path,
    show_legend: bool = True,
    add_labels: bool = False,
) -> None:
    if df.empty:
        print(f"⚠️ Data is empty. Skipping plot: {title}")
        return

    fig, ax = plt.subplots(figsize=(12, 6.75))

    for state_name, sub_df in df.groupby("state_name"):
        sub_df = sub_df.sort_values("period_end")
        ax.plot(sub_df["period_end"], sub_df[y_col], linewidth=2.6, label=state_name)

    ax.set_title(title, fontsize=18, pad=16)
    fig.text(0.5, 0.93, subtitle, ha="center", fontsize=11)
    ax.set_xlabel("Period End", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="x", rotation=35, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, alpha=0.25)

    if show_legend:
        ax.legend(frameon=True, fontsize=11)

    if add_labels:
        add_end_labels(ax, df, y_col)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Saved figure: {output_path}")


def build_index_2015(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a normalized index with 2015 = 100.
    Prefer each state's 2015 average as the baseline; if missing, fall back to the first non-null value.
    """
    df = df.copy().sort_values(["state_name", "period_end"])
    out = []

    for state_name, sub_df in df.groupby("state_name"):
        sub_df = sub_df.copy().dropna(subset=["analysis_value"])
        if sub_df.empty:
            continue

        baseline_2015 = sub_df.loc[sub_df["year"] == 2015, "analysis_value"].mean()

        if pd.notna(baseline_2015) and baseline_2015 != 0:
            baseline = baseline_2015
            baseline_source = "2015_average"
        else:
            baseline = sub_df["analysis_value"].iloc[0]
            baseline_source = "first_observation"

        if pd.isna(baseline) or baseline == 0:
            continue

        sub_df["index_2015_100"] = sub_df["analysis_value"] / baseline * 100
        sub_df["index_baseline_source"] = baseline_source
        out.append(sub_df)

    if not out:
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True)


def plot_latest_bar_chart(
    latest_summary: pd.DataFrame,
    value_col: str,
    title: str,
    subtitle: str,
    ylabel: str,
    output_path: Path,
) -> None:
    df = latest_summary.copy().dropna(subset=[value_col])
    if df.empty:
        print(f"⚠️ Data is empty. Skipping bar chart: {title}")
        return

    df = df.sort_values(value_col, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["state_name"], df[value_col], width=0.6)

    ax.set_title(title, fontsize=18, pad=16)
    fig.text(0.5, 0.92, subtitle, ha="center", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.25)

    for i, v in enumerate(df[value_col]):
        if pd.notna(v):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved figure: {output_path}")


def plot_optional_quality_chart(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    if df.empty or "percent_complete" not in df.columns:
        print("⚠️ percent_complete is missing. Skipping appendix quality chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 6.75))

    for state_name, sub_df in df.groupby("state_name"):
        sub_df = sub_df.sort_values("period_end")
        ax.plot(sub_df["period_end"], sub_df["percent_complete"], linewidth=2, label=state_name)

    ax.set_title("Appendix: Data Quality (Percent Complete)", fontsize=18, pad=16)
    ax.set_xlabel("Period End", fontsize=12)
    ax.set_ylabel("Percent Complete", fontsize=12)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📎 Saved appendix figure: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CDC cleaned trend data for presentation.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/cleaned",
        help="Directory containing cleaned CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root output directory; figures will be saved to outputs/figures/",
    )
    parser.add_argument(
        "--states",
        type=str,
        default="Kansas,Missouri,United States",
        help='Regions used for the main presentation, comma-separated, for example "Kansas,Missouri,United States"',
    )
    parser.add_argument(
        "--include-quality-appendix",
        action="store_true",
        help="Whether to additionally output a percent_complete chart for the appendix",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    figures_dir = output_dir / "figures"

    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    main_selected_path = input_dir / "cdc2_main_selected.csv"
    total_trend_path = input_dir / "cdc2_total_overdose_trend.csv"
    percent_trend_path = input_dir / "cdc2_percent_specified_trend.csv"

    print(f"📂 Input directory: {input_dir}")
    print(f"📁 Output root directory: {output_dir}")
    print(f"🖼 Figure output directory: {figures_dir}")

    main_df = load_csv(main_selected_path)
    main_df = standardize_dates(main_df)

    state_filter = [s.strip() for s in args.states.split(",") if s.strip()]

    total_df, total_rebuilt = load_or_rebuild_trend(
        total_trend_path,
        main_df,
        TOTAL_INDICATOR,
        expected_min_states=2,
    )

    percent_df, percent_rebuilt = load_or_rebuild_trend(
        percent_trend_path,
        main_df,
        PERCENT_INDICATOR,
        expected_min_states=2,
    )

    rebuilt_total_path = output_dir / "cdc2_total_overdose_trend_rebuilt.csv"
    rebuilt_percent_path = output_dir / "cdc2_percent_specified_trend_rebuilt.csv"

    save_rebuilt_if_needed(total_df, rebuilt_total_path, total_rebuilt)
    save_rebuilt_if_needed(percent_df, rebuilt_percent_path, percent_rebuilt)

    total_df = filter_states(total_df, state_filter)
    percent_df = filter_states(percent_df, state_filter)

    validate_trend_df(total_df, "total_df")
    validate_trend_df(percent_df, "percent_df")

    latest_summary = make_latest_summary(total_df, percent_df)
    latest_summary_path = output_dir / "cdc2_latest_summary.csv"
    latest_summary.to_csv(latest_summary_path, index=False)
    print(f"🧾 Saved latest summary: {latest_summary_path}")

    yearly_summary = make_yearly_summary(total_df, percent_df)
    yearly_summary_path = output_dir / "cdc2_yearly_summary.csv"
    yearly_summary.to_csv(yearly_summary_path, index=False)
    print(f"🧾 Saved yearly summary: {yearly_summary_path}")

    # Main Figure 1: Percent with Drugs Specified (three-region comparison)
    plot_presentation_line_chart(
        df=percent_df,
        y_col="analysis_value",
        title="Percent with Drugs Specified Over Time",
        subtitle="Kansas, Missouri, and United States comparison",
        ylabel="Percent",
        output_path=figures_dir / "01_percent_specified_comparison.png",
        show_legend=True,
        add_labels=False,
    )

    # Main Figure 2: Kansas vs Missouri death trend
    km_total_df = subset_states(total_df, ["Kansas", "Missouri"])
    plot_presentation_line_chart(
        df=km_total_df,
        y_col="analysis_value",
        title="12-Month Rolling Drug Overdose Deaths",
        subtitle="Kansas vs Missouri comparison",
        ylabel="Deaths",
        output_path=figures_dir / "02_overdose_deaths_kansas_missouri.png",
        show_legend=True,
        add_labels=False,
    )

    # Main Figure 3: United States standalone trend
    us_total_df = subset_states(total_df, ["United States"])
    plot_presentation_line_chart(
        df=us_total_df,
        y_col="analysis_value",
        title="United States 12-Month Rolling Drug Overdose Deaths",
        subtitle="National trend shown separately to avoid scale distortion",
        ylabel="Deaths",
        output_path=figures_dir / "03_overdose_deaths_united_states.png",
        show_legend=False,
        add_labels=True,
    )

    # Main Figure 4: Normalized index chart (2015=100)
    total_index_df = build_index_2015(total_df)
    total_index_path = output_dir / "cdc2_total_index_2015_100.csv"
    total_index_df.to_csv(total_index_path, index=False)
    print(f"🧾 Saved index table: {total_index_path}")

    plot_presentation_line_chart(
        df=total_index_df,
        y_col="index_2015_100",
        title="Normalized Overdose Death Trend (2015 = 100)",
        subtitle="Relative growth comparison across Kansas, Missouri, and the United States",
        ylabel="Index (2015=100)",
        output_path=figures_dir / "04_overdose_index_2015_100.png",
        show_legend=True,
        add_labels=False,
    )

    # Main Figure 5: Latest value bar chart (percent specified)
    plot_latest_bar_chart(
        latest_summary=latest_summary,
        value_col="latest_percent_specified",
        title="Latest Percent with Drugs Specified",
        subtitle="Most recent available observation by region",
        ylabel="Percent",
        output_path=figures_dir / "05_latest_percent_specified_bar.png",
    )

    # Optional appendix
    if args.include_quality_appendix:
        plot_optional_quality_chart(
            df=total_df,
            output_path=figures_dir / "appendix_percent_complete.png",
        )

    print("\n✅ Competition presentation version of plot_cdc2 completed")
    print("\nRecommended 5 figures for the main presentation:")
    print(f"1. {figures_dir / '01_percent_specified_comparison.png'}")
    print(f"2. {figures_dir / '02_overdose_deaths_kansas_missouri.png'}")
    print(f"3. {figures_dir / '03_overdose_deaths_united_states.png'}")
    print(f"4. {figures_dir / '04_overdose_index_2015_100.png'}")
    print(f"5. {figures_dir / '05_latest_percent_specified_bar.png'}")

    print("\nOutput data tables:")
    print(f"- {latest_summary_path}")
    print(f"- {yearly_summary_path}")
    print(f"- {total_index_path}")

    if not latest_summary.empty:
        print("\nFirst few rows of the latest summary:")
        print(latest_summary.head())


if __name__ == "__main__":
    main()