import streamlit as st
import pandas as pd
from pathlib import Path
import html
import re
from collections import Counter

st.set_page_config(page_title="Substance Risk Dashboard", layout="wide")

# -----------------------------
# Helper functions
# -----------------------------
def extract_top_keywords(text_series, n=8):
    stopwords = {
        "the", "and", "for", "that", "this", "with", "have", "had", "was", "are", "but",
        "not", "you", "your", "they", "them", "from", "been", "were", "when", "what",
        "will", "would", "could", "should", "about", "there", "their", "then", "than",
        "just", "very", "more", "some", "into", "over", "after", "before", "again",
        "back", "still", "also", "only", "like", "feel", "felt", "because", "while",
        "really", "much", "made", "make", "took", "take", "taking", "medication",
        "medicine", "pill", "pills", "drug", "drugs", "day", "days", "week", "weeks",
        "month", "months", "year", "years", "time", "times", "help", "helps", "helped",
        "side", "effects", "effect", "doctor", "mg", "one", "two", "did", "does", "doing",
        "get", "got", "going", "gone", "being", "life", "normal", "better", "great",
        "good", "bad", "using", "used", "use"
    }

    words = []
    for text in text_series.dropna():
        text = str(text).lower()
        tokens = re.findall(r"[a-z']+", text)
        tokens = [t for t in tokens if len(t) > 2 and t not in stopwords]
        words.extend(tokens)

    counter = Counter(words)
    return [w for w, _ in counter.most_common(n)]


def truncate_text(text, max_len=160):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# -----------------------------
# Load data
# -----------------------------
base_path = Path(__file__).resolve().parent.parent
csv_path = base_path / "outputs" / "tables" / "predictions_baseline.csv"

df = pd.read_csv(csv_path)

# Decode HTML entities like &#039;
df["text"] = df["text"].astype(str).apply(html.unescape)

# Clean date
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

# Make sure labels are integers
for col in ["substance_label", "distress_label", "relapse_label"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# -----------------------------
# Title
# -----------------------------
st.title("Substance Abuse Risk Dashboard")
st.write(
    "This dashboard presents risk signal detection, trend analysis, and topic discovery from anonymized social discussions."
)

# -----------------------------
# Top metrics
# -----------------------------
substance_count = int(df["substance_label"].sum())
distress_count = int(df["distress_label"].sum())
relapse_count = int(df["relapse_label"].sum())
total_posts = len(df)

col1, col2, col3 = st.columns(3)
col1.metric("Substance Posts", substance_count)
col2.metric("Distress Posts", distress_count)
col3.metric("Relapse Posts", relapse_count)

# -----------------------------
# 1. Risk Distribution
# -----------------------------
st.header("1. Risk Distribution")

risk_dist = pd.DataFrame({
    "Risk Type": ["Substance", "Distress", "Relapse"],
    "Percentage": [
        substance_count / total_posts * 100,
        distress_count / total_posts * 100,
        relapse_count / total_posts * 100
    ],
    "Count": [substance_count, distress_count, relapse_count]
})

st.bar_chart(risk_dist.set_index("Risk Type")["Percentage"])


# -----------------------------
# 2. Trend Analysis (monthly)
# -----------------------------
st.header("2. Trend Analysis")

df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

trend_df = df.groupby("month")[["substance_label", "distress_label", "relapse_label"]].sum()
trend_df = trend_df.rename(columns={
    "substance_label": "Substance",
    "distress_label": "Distress",
    "relapse_label": "Relapse"
})

st.line_chart(trend_df)

# -----------------------------
# 3. Topic Discovery
# -----------------------------
st.header("3. Topic Discovery")
st.write("Top observed keywords based on labeled posts:")

substance_keywords = extract_top_keywords(df.loc[df["substance_label"] == 1, "text"])
distress_keywords = extract_top_keywords(df.loc[df["distress_label"] == 1, "text"])
relapse_keywords = extract_top_keywords(df.loc[df["relapse_label"] == 1, "text"])

st.markdown(f"**Substance:** {', '.join(substance_keywords) if substance_keywords else 'N/A'}")
st.markdown(f"**Distress:** {', '.join(distress_keywords) if distress_keywords else 'N/A'}")
st.markdown(f"**Relapse:** {', '.join(relapse_keywords) if relapse_keywords else 'N/A'}")

# -----------------------------
# 4. Example Posts
# -----------------------------
st.header("4. Example Posts")

label_option = st.selectbox(
    "Select label to display",
    ["Substance", "Distress", "Relapse"]
)

if label_option == "Substance":
    filtered = df[df["substance_label"] == 1].copy()
elif label_option == "Distress":
    filtered = df[df["distress_label"] == 1].copy()
else:
    filtered = df[df["relapse_label"] == 1].copy()

filtered["text_preview"] = filtered["text"].apply(truncate_text)

display_df = filtered[[
    "post_id", "date_str", "text_preview",
    "substance_label", "distress_label", "relapse_label"
]].rename(columns={
    "date_str": "date",
    "text_preview": "text"
})

st.dataframe(display_df.head(10), use_container_width=True)
