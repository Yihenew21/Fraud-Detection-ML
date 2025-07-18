# Interim-1 Report: Fraud Detection Project

## Overview
This report summarizes Task 1 (Data Analysis and Preprocessing) for the 10 Academy Week 8&9 Challenge, focusing on fraud detection for e-commerce and bank transactions at Adey Innovations Inc. The work is implemented in `notebooks/interim_1.ipynb` and meets the Interim-1 submission requirements.

## Data Cleaning and Preprocessing
- **Datasets**: Loaded `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv`.
- **Duplicates**: Removed duplicates to ensure data integrity.
- **Missing Values**: Imputed numerical columns (`purchase_value`, `Amount`) with median and categorical columns (`source`, `browser`, `sex`) with mode.
- **Data Types**: Converted `signup_time` and `purchase_time` to datetime for temporal analysis.
- **Geolocation**: Merged `Fraud_Data.csv` with `IpAddress_to_Country.csv` to map IP addresses to countries, handling cases where no match is found (labeled as 'Unknown').

**Rationale**: These steps ensure a clean, consistent dataset ready for analysis and modeling, addressing potential errors in data entry and enabling geolocation-based insights.

## Exploratory Data Analysis (EDA)
- **Univariate Analysis**:
  - Histogram of `purchase_value` (saved as `plots/purchase_value_hist.png`) shows a right-skewed distribution, with a tail of high values often associated with fraud (Class=1).
  - Similar analysis for `Amount` in `creditcard.csv` reveals outliers that may indicate fraudulent transactions.
- **Bivariate Analysis**:
  - Time series plot (saved as `plots/time_series.png`) shows transaction counts by hour, with fraud cases spiking at unusual times (e.g., midnight to 4 AM).
  - Planned: Analyze `class` vs. categorical features (`source`, `browser`, `country`).
- **Insights**:
  - High `purchase_value` and transactions at odd hours are potential fraud indicators.
  - Geolocation may reveal fraud patterns in specific countries.

## Feature Engineering
- **Time-Based Features**:
  - `time_since_signup`: Duration (in hours) between `signup_time` and `purchase_time`. Hypothesis: Fraudsters make rapid purchases post-signup to exploit accounts.
  - `hour_of_day`: Hour of `purchase_time`. Hypothesis: Fraud occurs at unusual hours.
  - `day_of_week`: Day of `purchase_time`. Hypothesis: Fraud patterns may vary by day.
- **Transaction Frequency**:
  - `trans_freq`: Number of transactions per `user_id`. Hypothesis: High frequency may indicate automated fraud attempts.
- **Implementation**: Added in `src/feature_engineering.py` and applied in `interim_1.ipynb`.
- **Rationale**: These features capture temporal and behavioral patterns critical for fraud detection.

## Class Imbalance Analysis
- **E-commerce Data**: `class` distribution shows ~95% non-fraud (0) and ~5% fraud (1).
- **Credit Card Data**: `Class` distribution is highly imbalanced, with <1% fraud (1).
- **Strategy**: Propose SMOTE (Synthetic Minority Oversampling Technique) for training data only to balance classes. This prioritizes detecting fraud (high business cost of false negatives) while avoiding data leakage.
- **Rationale**: SMOTE generates synthetic fraud cases, improving model performance on rare events without overfitting to the test set.

## Repository
- GitHub: `https://github.com/<your-username>/Fraud-Detection-Adey-Innovations`
- Structure: `data/`, `notebooks/`, `src/`, `plots/`, `reports/`.
- Code: `interim_1.ipynb` contains all analysis, with modular functions in `src/data_utils.py` and `src/feature_engineering.py`.
- README: Includes setup instructions and project overview.

## Next Steps
- Task 2: Build and evaluate Logistic Regression and an ensemble model (e.g., Random Forest).
- Incorporate SMOTE, normalization, and categorical encoding for modeling.