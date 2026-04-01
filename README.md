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
