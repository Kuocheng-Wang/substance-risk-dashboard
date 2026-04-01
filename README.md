# Substance Risk Dashboard

A lightweight NLP pipeline for detecting **substance**, **distress**, and **relapse** signals from public text data, with **monthly trend analysis** and an **interactive Streamlit dashboard**.

## Team Members
- Yuchen Zhang
- Kuocheng Wang

## Overview
This project was developed for **UMKC NSF NRT Challenge 1 (AI)**, **Track B: Data Intelligence and Decision Support**.

The system:
- preprocesses raw text data
- applies rule-based baseline labeling
- analyzes monthly risk trends
- visualizes results in an interactive dashboard

## Risk Labels
- **Substance**: substance-use-related language in an abuse, dependency, or recovery context
- **Distress**: emotional or psychological distress signals
- **Relapse**: cravings, withdrawal, relapse, or recovery-difficulty signals

## Project Structure
```text
substance-risk-dashboard/
├── app/
│   └── streamlit_app.py
├──── outputs/
│   ├── cleanned/
│       ├── cleaned_drugsComTrain_sample.csv
│       └── cleaned_drugsComTest_sample.csv
│   └── tables/
│       ├── predictions_baseline.csv
│       └── predictions_baseline_sample.csv
└── src/
    ├── baseline_drugsComTest_raw.py
    ├── baseline_drugsComTrain_raw.py
    ├── preprocess_drugsComTest_raw.py
    └── preprocess_drugsComTrain_raw.py
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
py src/preprocess.py
```

Run baseline labeling:

```bash
py src/baseline.py
```

Launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Sample Files
This repository includes sample files for demonstration:

- `outputs/cleanned/cleaned_drugsComTrain.csv`
- `outputs/tables/predictions_drugsComTrain.csv`
- `outputs/cleanned/cleaned_drugsComTest.csv`
- `outputs/tables/predictions_drugsComTest.csv`

Due to GitHub file size limits, the full dataset and full prediction files are kept locally and are not included in this repository.

## Dashboard Features
- Top metrics
- Risk distribution
- Monthly trend analysis
- Topic discovery
- Example posts by label

## Notes
This project is intended for **population-level monitoring** and **decision support**.  
The labels represent **detected risk-related language patterns**, not clinical diagnoses.
