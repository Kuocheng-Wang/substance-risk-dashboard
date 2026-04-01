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
substance-risk-dashboard/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── drugsComTrain_raw.csv
│   └── processed/
│       └── cleaned_data.csv
├── outputs/
│   ├── figures/
│   └── tables/
│       ├── predictions_baseline.csv
│       └── predictions_baseline_sample.csv
└── src/
    ├── preprocess.py
    └── baseline.py

## How to Run

Install dependencies:

pip install -r requirements.txt

Run preprocessing:

py src/preprocess.py

Run baseline labeling:

py src/baseline.py

Launch the dashboard:

streamlit run app/streamlit_app.py

## Sample Files
This repository includes sample files for demonstration:

- data/processed/cleaned_data_sample.csv
- outputs/tables/predictions_baseline_sample.csv

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
