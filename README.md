# Fraud Detection Project - 10 Academy Week 8&9

## Overview
This project develops fraud detection models for e-commerce and bank transactions for Adey Innovations Inc., addressing class imbalance and incorporating geolocation and transaction pattern analysis. This repository contains code, notebooks, and reports for the 10 Academy Week 8&9 Challenge, covering Task 1 (Data Analysis and Preprocessing) and Task 2 (Model Building and Training).

## Setup
1. Clone the repository: `git clone https://github.com/Yihenew21/Fraud-Detection-ML.git`
2. Create and activate a virtual environment:
   - Windows: `python -m venv .venv && .venv\Scripts\activate`
   - Linux/MacOS: `python -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Place datasets (`Fraud_Data.csv`, `IpAddress_to_Country.csv`, `creditcard.csv`) in `data/raw/`.

## Project Structure
- `data/raw/`: Raw datasets.
- `data/processed/`: Processed datasets after cleaning and feature engineering (e.g., `processed_ecommerce_with_features.csv`, `processed_creditcard.csv`).
- `notebooks/`: Jupyter notebooks (e.g., `EDA.ipynb` for Task 1, `model_training.ipynb` for Task 2).
- `src/`: Modular scripts (`data_utils.py`, `feature_engineering.py`).
- `plots/`: EDA and model visualizations (e.g., `purchase_value_hist.png`, `ecommerce_roc_curve.png`, `ecommerce_pr_curve.png`, `creditcard_roc_curve.png`, `creditcard_pr_curve.png`).
- `reports/`: Submission reports (e.g., `interim_1_report.md`, `interim_2_report.md`).
- `requirements.txt`: Project dependencies.
- `.gitignore`: Ignores virtual environment and temporary files.

## Running the Code
- Run `notebooks/EDA.ipynb` in Jupyter Lab to perform Task 1 (Data Analysis and Preprocessing).
- Run `notebooks/model_training.ipynb` in Jupyter Lab to perform Task 2 (Model Building and Training) and generate model evaluations.

## Status
- Task 1 (Interim-1): Completed data cleaning, EDA, feature engineering, and class imbalance analysis.
- Task 2 (Interim-2): Completed model building and training with Logistic Regression and Random Forest on e-commerce and credit card datasets, including hyperparameter tuning, evaluation metrics (F1, ROC-AUC, AUC-PR), confusion matrices, and visualizations.