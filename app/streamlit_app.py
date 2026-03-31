from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Substance Risk Dashboard", layout="wide")

st.title("Substance Abuse Risk Dashboard")
st.write("This dashboard presents risk signal detection, trend analysis, and topic discovery from anonymized social discussions.")

base_path = Path(__file__).resolve().parent.parent
csv_path = base_path / "outputs" / "tables" / "predictions_baseline.csv"

try:
    df = pd.read_csv(csv_path)
    st.success("Loaded predictions_baseline.csv successfully.")
except:
    df = pd.DataFrame({
        "post_id": [1, 2, 3, 4, 5],
        "date": ["2025-01-01", "2025-01-05", "2025-01-10", "2025-01-15", "2025-01-20"],
        "text": [
            "I drank again after being sober for two weeks.",
            "I feel depressed and stressed every day.",
            "My friend keeps using opioids and I am worried.",
            "I cannot stop craving alcohol at night.",
            "I feel anxious but I am trying to recover."
        ],
        "substance_label": [1, 0, 1, 1, 0],
        "distress_label": [0, 1, 0, 0, 1],
        "relapse_label": [1, 0, 0, 1, 0]
    })
    st.warning("No predictions_baseline.csv found. Showing demo data instead.")
# ===== 1. Risk Distribution =====
col1, col2, col3 = st.columns(3)
col1.metric("Substance Posts", int(df["substance_label"].sum()))
col2.metric("Distress Posts", int(df["distress_label"].sum()))
col3.metric("Relapse Posts", int(df["relapse_label"].sum()))
st.header("1. Risk Distribution")

risk_counts = pd.DataFrame({
    "Risk Type": ["Substance", "Distress", "Relapse"],
    "Count": [
        df["substance_label"].sum(),
        df["distress_label"].sum(),
        df["relapse_label"].sum()
    ]
})

st.bar_chart(risk_counts.set_index("Risk Type"))

# ===== 2. Trend Analysis =====
st.header("2. Trend Analysis")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
trend_df = df.groupby("date")[["substance_label", "distress_label", "relapse_label"]].sum()
st.line_chart(trend_df)

# ===== 3. Topic Discovery =====
st.header("3. Topic Discovery")
st.write("Preliminary themes identified from the dataset:")
st.write("- Alcohol use and drinking relapse")
st.write("- Drug use and opioid-related mentions")
st.write("- Emotional distress such as anxiety and depression")
st.write("- Recovery attempts, cravings, and relapse signals")

# ===== 4. Example Posts =====
st.write("Sample labeled posts:")
st.header("4. Example Posts")
st.dataframe(df[["post_id", "date", "text", "substance_label", "distress_label", "relapse_label"]])
