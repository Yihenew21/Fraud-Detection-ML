# Interim-1 Report: Fraud Detection Project

## Overview
This report summarizes Task 1 (Data Analysis and Preprocessing) for the 10 Academy Week 8&9 Challenge, focusing on fraud detection for e-commerce and bank transactions at Adey Innovations Inc. The work is implemented in `notebooks/interim_1.ipynb` and meets the Interim-1 submission requirements.

## Data Cleaning and Preprocessing
- **Datasets**: Loaded `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` using `src/data_utils.py`.
- **Duplicates**: Removed duplicates to ensure data integrity.
- **Missing Values**: Imputed numerical columns (`purchase_value`, `Amount`) with median and categorical columns (`source`, `browser`, `sex`) with mode to maintain data consistency.
- **Data Types**: Converted `signup_time` and `purchase_time` to datetime for temporal analysis.
- **Geolocation**: Merged `Fraud_Data.csv` with `IpAddress_to_Country.csv` to map IP addresses to countries, handling unmatched IPs as 'Unknown'.
- **Output**: Saved processed datasets to `data/processed/`.

**Rationale**: These steps ensure a clean, consistent dataset for analysis and modeling, addressing errors and enabling geolocation-based insights critical for fraud detection.

## Exploratory Data Analysis (EDA)
- **Univariate Analysis**:
  - Histogram of `purchase_value` (`plots/purchase_value_hist.png`) shows a right-skewed distribution, with high values often associated with fraud (Class=1).
  - Histogram of `Amount` in `creditcard.csv` reveals outliers potentially linked to fraud.
- **Bivariate Analysis**:
  - Time series plot (`plots/time_series.png`) shows transaction counts by hour, with fraud cases spiking at unusual times (e.g., midnight to 4 AM).
- **Insights**:
  - High `purchase_value` and transactions at odd hours are strong indicators of fraud.
  - Geolocation analysis (via `country`) may reveal fraud hotspots in specific regions.

## Feature Engineering
- **Time-Based Features** (in `src/feature_engineering.py`):
  - `time_since_signup`: Duration (in hours) between `signup_time` and `purchase_time`. Hypothesis: Fraudsters make rapid purchases post-signup to exploit accounts.
  - `hour_of_day`: Hour of `purchase_time`. Hypothesis: Fraud occurs at unusual hours.
  - `day_of_week`: Day of `purchase_time`. Hypothesis: Fraud patterns may vary by day.
- **Transaction Frequency**:
  - `trans_freq`: Number of transactions per `user_id`. Hypothesis: High frequency may indicate automated fraud attempts.
- **Output**: Saved updated e-commerce dataset to `data/processed/processed_ecommerce_with_features.csv`.
- **Rationale**: These features capture temporal and behavioral patterns essential for identifying fraudulent activities.

## Class Imbalance Analysis
- **E-commerce Data**: `class` distribution shows ~95% non-fraud (0) and ~5% fraud (1).
- **Credit Card Data**: `Class` distribution is highly imbalanced, with <1% fraud (1).
- **Strategy**: Propose SMOTE (Synthetic Minority Oversampling Technique) for training data only to balance classes. This prioritizes detecting fraud (high business cost of false negatives) while avoiding data leakage.
- **Rationale**: SMOTE generates synthetic fraud cases, improving model performance on rare events critical for financial security.

## Repository
- GitHub: `https://github.com/<your-username>/Fraud-Detection-Adey-Innovations`
- Structure: `data/`, `notebooks/`, `src/`, `plots/`, `reports/`.
- Code: `interim_1.ipynb` contains all analysis, supported by modular functions in `src/`.
- README: Includes setup instructions, project overview, and execution steps.

## Next Steps
- Task 2: Build and evaluate Logistic Regression and an ensemble model (e.g., Random Forest).
- Incorporate SMOTE, normalization, and categorical encoding for modeling.