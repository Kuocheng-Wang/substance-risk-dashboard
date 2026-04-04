import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import html

st.set_page_config(page_title="Substance Risk Dashboard", layout="wide")


# =========================================================
# Helpers
# =========================================================
def truncate_text(text, max_len=180):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def safe_read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def first_existing_path(candidates):
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def make_pivot(df: pd.DataFrame, index_col: str, column_col: str, value_col: str) -> pd.DataFrame:
    out = df.pivot_table(
        index=index_col,
        columns=column_col,
        values=value_col,
        aggfunc="mean"
    )
    out = out.sort_index()
    return out


def add_percentage_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Substance %"] = (df["Substance Count"] / df["Total Posts"] * 100).round(2)
    df["Distress %"] = (df["Distress Count"] / df["Total Posts"] * 100).round(2)
    df["Relapse %"] = (df["Relapse Count"] / df["Total Posts"] * 100).round(2)
    return df


def render_filled_line_chart(
    pivot_df: pd.DataFrame,
    x_title: str = "Period End",
    y_title: str = "Value",
    hard_min: float | None = None,
    hard_max: float | None = None,
    pad_ratio: float = 0.08,
    min_abs_pad: float = 1.0,
    height: int = 360,
):
    """
    用 Altair 画自动缩放纵轴的折线图，让曲线更充满图。
    """
    if pivot_df.empty:
        st.info("No data available for this chart.")
        return

    data = pivot_df.reset_index().melt(
        id_vars=pivot_df.index.name or "index",
        var_name="series",
        value_name="value"
    )
    x_col = pivot_df.index.name or "index"
    data = data.dropna(subset=["value"]).copy()

    if data.empty:
        st.info("No data available for this chart.")
        return

    y_min = float(data["value"].min())
    y_max = float(data["value"].max())

    if y_min == y_max:
        lower = y_min - 1
        upper = y_max + 1
    else:
        pad = max((y_max - y_min) * pad_ratio, min_abs_pad)
        lower = y_min - pad
        upper = y_max + pad

    if hard_min is not None:
        lower = max(hard_min, lower)
    if hard_max is not None:
        upper = min(hard_max, upper)

    if lower >= upper:
        upper = lower + 1

    chart = (
        alt.Chart(data)
        .mark_line(point=False)
        .encode(
            x=alt.X(f"{x_col}:T", title=x_title),
            y=alt.Y(
                "value:Q",
                title=y_title,
                scale=alt.Scale(domain=[lower, upper], zero=False),
            ),
            color=alt.Color("series:N", title=None),
            tooltip=[
                alt.Tooltip(f"{x_col}:T", title=x_title),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title=y_title, format=".2f"),
            ],
        )
        .properties(height=height)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


# =========================================================
# Paths
# =========================================================
BASE_PATH = Path(__file__).resolve().parent.parent

TABLES_DIR = first_existing_path([
    BASE_PATH / "outputs" / "tables",
])

CLEANED_DIR = first_existing_path([
    BASE_PATH / "outputs" / "cleaned",
    BASE_PATH / "outputs" / "cleanned",
    BASE_PATH / "data" / "processed",
])

TRAIN_PRED_PATH = first_existing_path([
    TABLES_DIR / "predictions_drugsComTrain.csv",
    TABLES_DIR / "predictions_baseline.csv",
])

TEST_PRED_PATH = first_existing_path([
    TABLES_DIR / "predictions_drugsComTest.csv",
])

CDC2_TOTAL_PATH = first_existing_path([
    CLEANED_DIR / "cdc2_total_overdose_trend.csv",
    CLEANED_DIR / "cdc2_total_overdose_trend(1).csv",
])

CDC2_PERCENT_PATH = first_existing_path([
    CLEANED_DIR / "cdc2_percent_specified_trend.csv",
])

CDC1_TOP_DRUG_PATH = first_existing_path([
    CLEANED_DIR / "cdc_top_drug_trends.csv",
])


# =========================================================
# Data prep
# =========================================================
def prepare_text_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    required_cols = {
        "post_id", "date", "text",
        "substance_label", "distress_label", "relapse_label"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{source_name} file is missing columns: {missing}")

    df = df.copy()
    df["text"] = df["text"].astype(str).apply(html.unescape)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    for col in ["substance_label", "distress_label", "relapse_label"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["source"] = source_name
    return df


def prepare_cdc2_df(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"state_name", "period_end", "analysis_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CDC2 file is missing columns: {missing}")

    df = df.copy()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["analysis_value"] = pd.to_numeric(df["analysis_value"], errors="coerce")
    df = df.dropna(subset=["state_name", "period_end", "analysis_value"]).copy()
    df = df.sort_values(["state_name", "period_end"]).reset_index(drop=True)
    return df


def prepare_cdc1_df(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"state_name", "indicator", "period_end", "overdose_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CDC1 file is missing columns: {missing}")

    df = df.copy()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["overdose_value"] = pd.to_numeric(df["overdose_value"], errors="coerce")
    df = df.dropna(subset=["state_name", "indicator", "period_end", "overdose_value"]).copy()
    df = df.sort_values(["state_name", "indicator", "period_end"]).reset_index(drop=True)
    return df


# =========================================================
# Cached loaders
# =========================================================
@st.cache_data
def load_text_data():
    train_df = prepare_text_df(safe_read_csv(TRAIN_PRED_PATH), "Train")

    if TEST_PRED_PATH.exists():
        test_df = prepare_text_df(safe_read_csv(TEST_PRED_PATH), "Test")
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        test_df = pd.DataFrame(columns=train_df.columns)
        combined_df = train_df.copy()

    return train_df, test_df, combined_df


@st.cache_data
def load_cdc2_data():
    total_df = prepare_cdc2_df(safe_read_csv(CDC2_TOTAL_PATH))
    percent_df = prepare_cdc2_df(safe_read_csv(CDC2_PERCENT_PATH))
    return total_df, percent_df


@st.cache_data
def load_cdc1_data():
    return prepare_cdc1_df(safe_read_csv(CDC1_TOP_DRUG_PATH))


# =========================================================
# Load text data
# =========================================================
try:
    train_df, test_df, combined_df = load_text_data()
except Exception as e:
    st.error(f"Failed to load drugs.com prediction files:\n{e}")
    st.stop()


# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("Dashboard Controls")

text_view_options = ["Combined", "Train"]
if not test_df.empty:
    text_view_options.append("Test")

dataset_view = st.sidebar.selectbox("Text Dataset View", text_view_options)

label_option = st.sidebar.selectbox(
    "Example Posts Label",
    ["Substance", "Distress", "Relapse"]
)

max_examples = st.sidebar.slider(
    "Number of Example Posts",
    min_value=3,
    max_value=15,
    value=5,
    step=1
)

if dataset_view == "Train":
    df = train_df.copy()
elif dataset_view == "Test" and not test_df.empty:
    df = test_df.copy()
else:
    df = combined_df.copy()


# =========================================================
# Title
# =========================================================
st.title("Substance Risk Dashboard")
st.write(
    "This dashboard integrates three components: "
    "(1) rule-based text risk detection from the drugs.com review dataset, "
    "(2) CDC aggregate overdose trends, and "
    "(3) CDC regional drug-specific comparisons."
)


# =========================================================
# SECTION A — TEXT RISK SIGNALS
# =========================================================
st.header("A. Text Risk Signals (drugs.com)")
st.caption(f"Current text view: {dataset_view}")

substance_count = int(df["substance_label"].sum())
distress_count = int(df["distress_label"].sum())
relapse_count = int(df["relapse_label"].sum())
total_posts = len(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Posts", f"{total_posts:,}")
c2.metric("Substance Posts", f"{substance_count:,}")
c3.metric("Distress Posts", f"{distress_count:,}")
c4.metric("Relapse Posts", f"{relapse_count:,}")

# 1. Risk Distribution
st.subheader("1. Risk Distribution")
risk_dist = pd.DataFrame({
    "Risk Type": ["Substance", "Distress", "Relapse"],
    "Percentage": [
        (substance_count / total_posts * 100) if total_posts else 0,
        (distress_count / total_posts * 100) if total_posts else 0,
        (relapse_count / total_posts * 100) if total_posts else 0,
    ],
})
risk_dist = risk_dist.set_index("Risk Type").reindex(["Substance", "Distress", "Relapse"])
st.bar_chart(risk_dist["Percentage"])

# 2. Monthly Trend Analysis
st.subheader("2. Monthly Trend Analysis")
trend_df = df.groupby("month")[["substance_label", "distress_label", "relapse_label"]].sum()
trend_df = trend_df.rename(columns={
    "substance_label": "Substance",
    "distress_label": "Distress",
    "relapse_label": "Relapse"
})
trend_df = trend_df[["Substance", "Distress", "Relapse"]]

if len(trend_df) > 1:
    trend_df = trend_df.iloc[:-1]

render_filled_line_chart(
    trend_df,
    x_title="Month",
    y_title="Post Count",
    hard_min=0,
    pad_ratio=0.06,
    min_abs_pad=5,
    height=340,
)

# 3. Train vs Test Comparison
st.subheader("3. Train vs Test Comparison")
split_rows = [
    {
        "Split": "Train",
        "Total Posts": len(train_df),
        "Substance Count": int(train_df["substance_label"].sum()),
        "Distress Count": int(train_df["distress_label"].sum()),
        "Relapse Count": int(train_df["relapse_label"].sum()),
    }
]

if not test_df.empty:
    split_rows.append(
        {
            "Split": "Test",
            "Total Posts": len(test_df),
            "Substance Count": int(test_df["substance_label"].sum()),
            "Distress Count": int(test_df["distress_label"].sum()),
            "Relapse Count": int(test_df["relapse_label"].sum()),
        }
    )

split_rows.append(
    {
        "Split": "Combined",
        "Total Posts": len(combined_df),
        "Substance Count": int(combined_df["substance_label"].sum()),
        "Distress Count": int(combined_df["distress_label"].sum()),
        "Relapse Count": int(combined_df["relapse_label"].sum()),
    }
)

split_summary = pd.DataFrame(split_rows)
split_summary = add_percentage_columns(split_summary)

st.dataframe(
    split_summary.style.format({
        "Total Posts": "{:,.0f}",
        "Substance Count": "{:,.0f}",
        "Distress Count": "{:,.0f}",
        "Relapse Count": "{:,.0f}",
        "Substance %": "{:.2f}%",
        "Distress %": "{:.2f}%",
        "Relapse %": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)

# 4. Topic Discovery
st.subheader("4. Topic Discovery")
st.write("Top observed keywords based on labeled posts:")

manual_keywords = {
    "Substance": ["suboxone", "methadone", "oxycodone", "heroin", "alcohol", "cravings"],
    "Distress": ["anxiety", "depression", "panic", "hopeless", "suicidal", "stress"],
    "Relapse": ["withdrawal", "detox", "sober", "relapse", "recovery", "cravings"],
}

st.markdown(f"**Substance:** {', '.join(manual_keywords['Substance'])}")
st.markdown(f"**Distress:** {', '.join(manual_keywords['Distress'])}")
st.markdown(f"**Relapse:** {', '.join(manual_keywords['Relapse'])}")

# 5. Example Posts
st.subheader("5. Example Posts")
label_map = {
    "Substance": "substance_label",
    "Distress": "distress_label",
    "Relapse": "relapse_label"
}
selected_label_col = label_map[label_option]

filtered = df[df[selected_label_col] == 1].copy()
filtered = filtered.sort_values("date", ascending=False).head(max_examples).copy()
filtered["text_preview"] = filtered["text"].apply(truncate_text)

display_df = filtered[["source", "date_str", "text_preview"]].rename(
    columns={
        "source": "Dataset",
        "date_str": "Date",
        "text_preview": "Text",
    }
)

st.dataframe(display_df, use_container_width=True, hide_index=True)


# =========================================================
# SECTION B — CDC2 AGGREGATE TRENDS
# =========================================================
st.markdown("---")

try:
    cdc2_total_df, cdc2_percent_df = load_cdc2_data()

    available_states = sorted(cdc2_total_df["state_name"].unique().tolist())
    default_states = [s for s in ["Kansas", "Missouri", "United States"] if s in available_states]

    if not default_states:
        default_states = available_states[:1]

    single_state_mode = len(available_states) == 1

    if single_state_mode:
        selected_states = available_states
        section_b_title = f"B. {available_states[0]} CDC Aggregate Trends"
    else:
        selected_states = st.multiselect(
            "Select states/regions for CDC aggregate trends",
            options=available_states,
            default=default_states
        )
        section_b_title = "B. CDC Aggregate Trends"

    st.header(section_b_title)

    if selected_states:
        total_view = cdc2_total_df[cdc2_total_df["state_name"].isin(selected_states)].copy()
        percent_view = cdc2_percent_df[cdc2_percent_df["state_name"].isin(selected_states)].copy()

        latest_total = (
            total_view.sort_values("period_end")
            .groupby("state_name", as_index=False)
            .tail(1)[["state_name", "period_end", "analysis_value"]]
            .rename(columns={"analysis_value": "latest_total"})
        )

        latest_percent = (
            percent_view.sort_values("period_end")
            .groupby("state_name", as_index=False)
            .tail(1)[["state_name", "analysis_value"]]
            .rename(columns={"analysis_value": "latest_percent"})
        )

        latest_summary = latest_total.merge(latest_percent, on="state_name", how="outer")
        latest_summary = latest_summary.sort_values("state_name").reset_index(drop=True)

        st.subheader("1. Latest Values")

        if len(latest_summary) == 1:
            row = latest_summary.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Latest Total Overdose Value", f"{row['latest_total']:,.2f}")
            m2.metric("Latest Percent with Drugs Specified", f"{row['latest_percent']:.2f}%")
            m3.metric("Latest Period", pd.to_datetime(row["period_end"]).strftime("%Y-%m-%d"))
        else:
            for _, row in latest_summary.iterrows():
                st.markdown(f"**{row['state_name']}**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Latest Total Overdose Value", f"{row['latest_total']:,.2f}")
                m2.metric("Latest Percent with Drugs Specified", f"{row['latest_percent']:.2f}%")
                m3.metric("Latest Period", pd.to_datetime(row["period_end"]).strftime("%Y-%m-%d"))
                st.markdown("")

        st.subheader("2. 12-Month Rolling Total Drug Overdose Deaths")
        total_pivot = make_pivot(total_view, "period_end", "state_name", "analysis_value")
        render_filled_line_chart(
            total_pivot,
            x_title="Period End",
            y_title="Deaths",
            hard_min=0,
            pad_ratio=0.08,
            min_abs_pad=10,
            height=340,
        )

        st.subheader("3. Percent with Drugs Specified Over Time")
        percent_pivot = make_pivot(percent_view, "period_end", "state_name", "analysis_value")
        render_filled_line_chart(
            percent_pivot,
            x_title="Period End",
            y_title="Percent",
            hard_min=0,
            hard_max=100,
            pad_ratio=0.03,
            min_abs_pad=2.0,
            height=340,
        )

    else:
        st.info("Select at least one state/region to view CDC aggregate trends.")

except Exception as e:
    st.warning(f"CDC aggregate section could not be loaded: {e}")


# =========================================================
# SECTION C — CDC1 REGIONAL DRUG TRENDS
# =========================================================
st.markdown("---")
st.header("C. CDC Regional Drug Trends")

try:
    cdc1_df = load_cdc1_data()

    region_options = sorted(cdc1_df["state_name"].unique().tolist())
    drug_options = sorted(cdc1_df["indicator"].unique().tolist())

    default_region_index = region_options.index("Region 10") if "Region 10" in region_options else 0
    default_drug_index = drug_options.index("Cocaine") if "Cocaine" in drug_options else 0

    col_a, col_b = st.columns(2)
    region_choice = col_a.selectbox(
        "Select one region for multi-drug trend",
        region_options,
        index=default_region_index
    )
    drug_choice = col_b.selectbox(
        "Select one drug for regional comparison",
        drug_options,
        index=default_drug_index
    )

    st.subheader("1. Multi-Drug Trend in Selected Region")
    region_df = cdc1_df[cdc1_df["state_name"] == region_choice].copy()
    region_pivot = make_pivot(region_df, "period_end", "indicator", "overdose_value")
    render_filled_line_chart(
        region_pivot,
        x_title="Period End",
        y_title="Overdose Value",
        hard_min=0,
        pad_ratio=0.08,
        min_abs_pad=10,
        height=340,
    )

    st.subheader("2. Regional Comparison for Selected Drug")
    drug_df = cdc1_df[cdc1_df["indicator"] == drug_choice].copy()
    drug_pivot = make_pivot(drug_df, "period_end", "state_name", "overdose_value")
    render_filled_line_chart(
        drug_pivot,
        x_title="Period End",
        y_title="Overdose Value",
        hard_min=0,
        pad_ratio=0.08,
        min_abs_pad=10,
        height=340,
    )

except Exception as e:
    st.warning(f"CDC regional drug section could not be loaded: {e}")


# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("Combined text view = Train + Test merged together for overall text-risk statistics and visualization.")
