# Fraud Detection Project - 10 Academy Week 8&9

## Overview
This project develops fraud detection models for e-commerce and bank transactions for Adey Innovations Inc., addressing class imbalance and incorporating geolocation and transaction pattern analysis. This repository contains code, notebooks, and reports for the 10 Academy Week 8&9 Challenge.

## Setup
1. Clone the repository: `git clone https://github.com/Yhenew21/Fraud-Detection-ML.git`
2. Create and activate a virtual environment:
   - Windows: `python -m venv .venv && .venv\Scripts\activate`
   - Linux/MacOS: `python -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Place datasets (`Fraud_Data.csv`, `IpAddress_to_Country.csv`, `creditcard.csv`) in `data/raw/`.

## Project Structure
- `data/raw/`: Raw datasets.
- `data/processed/`: Processed datasets after cleaning and feature engineering.
- `notebooks/`: Jupyter notebooks (e.g., `EDA.ipynb` for Task 1).
- `src/`: Modular scripts (`data_utils.py`, `feature_engineering.py`).
- `plots/`: EDA visualizations (e.g., `purchase_value_hist.png`).
- `reports/`: Submission reports (e.g., `interim_1_report.md`).
- `requirements.txt`: Project dependencies.
- `.gitignore`: Ignores virtual environment and temporary files.

## Running the Code
Run `notebooks/EDA.ipynb` in Jupyter Lab to perform Task 1 (Data Analysis and Preprocessing).

## Status
Task 1 (Interim-1): Completed data cleaning, EDA, feature engineering, and class imbalance analysis.