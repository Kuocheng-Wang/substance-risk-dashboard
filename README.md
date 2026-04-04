# Substance Risk Dashboard

This project was developed for **UMKC NSF NRT Challenge 1 (AI)** and focuses on building an end-to-end dashboard for **substance risk detection, temporal trend analysis, and interpretable decision-support insights**.

## Team Members
- Yuchen Zhang
- Kuocheng Wang

The project integrates three analytical streams into one dashboard:

1. **Text risk signals** from the drugs.com review dataset  
2. **CDC aggregate overdose trends** from VSRR-style public-health data  
3. **CDC regional drug-specific trends** for selected overdose indicators  

The final goal is to transform raw public data into **interpretable, population-level, decision-support insights**.

---

## Project Pipeline

The overall workflow is:

**Text Data**  
→ Preprocessing  
→ Cleaned Text Data  
→ Baseline Labeling  
→ Predictions  
→ Early Behavioral Risk Signals  

**CDC VSRR Data**  
→ Preprocessing  
→ Trend Tables  
→ Core Trend Figures  
→ Population-Level Overdose Trends  

**CDC Specific Drug Data**  
→ Preprocessing  
→ Drug Trend Table  
→ Plotting  
→ Drug-Specific Regional Figures  

**All Streams Combined**  
→ Interpretable Decision-Support Insights  

---

## Current Repository Structure

```text
substance-risk-dashboard/
├── app/
│   └── streamlit_app.py
├── outputs/
│   ├── figures/
│   │   ├── *.png
│   │   └── Data-to-Output Map .docx
│   └── tables/
│       ├── predictions_drugsComTrain_sample.csv
│       └── predictions_drugsComTest_sample.csv
├── src/
│   ├── baseline_drugsComTest_raw.py
│   ├── baseline_drugsComTrain_raw.py
│   ├── make_cleaned_sample.py
│   ├── plot_cdc1.py
│   ├── plot_cdc2.py
│   ├── preprocess_cdc1.py
│   ├── preprocess_cdc2.py
│   ├── preprocess_drugsComTest_raw.py
│   └── preprocess_drugsComTrain_raw.py
├── README.md
└── requirements.txt
```

---

## What This Repository Includes

This GitHub version includes:
- source code for preprocessing, baseline labeling, and plotting
- the integrated Streamlit dashboard
- selected figures for presentation
- lightweight sample prediction outputs
- the project pipeline map

This repository is intended as a **clean submission / demonstration version** of the project.

---

## Important Notes About Removed Folders

### `data/` folder
The full `data/` folder is **not included** in this repository because the raw data files were too large to upload to GitHub.

### `outputs/cleaned/` folder
The full `outputs/cleaned/` folder is **not included** in this repository because the cleaned intermediate files were too large to upload to GitHub.

As a result, this GitHub repository does **not** contain the full raw datasets or the full intermediate cleaned outputs.

---

## Sample Files Included

To keep the repository lightweight and reviewable, only small sample prediction files are included in:

```text
outputs/tables/
```

Current sample files:
- `predictions_drugsComTrain_sample.csv`
- `predictions_drugsComTest_sample.csv`

These files are included only for **demonstration and repository readability**.

---

## Main Scripts

### Text pipeline
- `preprocess_drugsComTrain_raw.py`
- `preprocess_drugsComTest_raw.py`
- `baseline_drugsComTrain_raw.py`
- `baseline_drugsComTest_raw.py`

### CDC aggregate trend pipeline
- `preprocess_cdc2.py`
- `plot_cdc2.py`

### CDC specific-drug regional pipeline
- `preprocess_cdc1.py`
- `plot_cdc1.py`

### Utility
- `make_cleaned_sample.py`

---

## Dashboard Components

### A. Text Risk Signals
- risk distribution
- monthly trend analysis
- train vs test comparison
- topic discovery
- example posts

### B. CDC Aggregate Trends
- latest values
- 12-month rolling total drug overdose deaths
- percent with drugs specified over time

### C. CDC Regional Drug Trends
- multi-drug trend in selected region
- regional comparison for selected drug

---

## Local Reproduction Notes

To fully reproduce the project locally, the missing raw data and missing cleaned intermediate files must be stored on the local machine before running the scripts.

A typical local workflow is:

1. prepare the required raw data locally  
2. run the preprocessing scripts in `src/`  
3. run the baseline / plotting scripts in `src/`  
4. launch the Streamlit dashboard  

Example:

```bash
python src/preprocess_drugsComTrain_raw.py
python src/baseline_drugsComTrain_raw.py
python src/preprocess_drugsComTest_raw.py
python src/baseline_drugsComTest_raw.py
python src/preprocess_cdc1.py
python src/plot_cdc1.py
python src/preprocess_cdc2.py
python src/plot_cdc2.py
streamlit run app/streamlit_app.py
```

---

## Repository Purpose

This repository is designed to show:
- the end-to-end project structure
- the analytical workflow
- the integrated dashboard
- the final visual outputs
- the logic that connects raw data to interpretable insights

It is a **submission-ready GitHub version** focused on clarity, presentation, and project reproducibility at the code level.

---

## Final Note

Because the large raw-data folder and large cleaned intermediate outputs were removed from the repository, this GitHub version emphasizes:
- readable code
- lightweight sample outputs
- final figures
- dashboard presentation
- a clear data-to-output pipeline
